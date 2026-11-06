"""Compare trained postfilters, NDAC, and optionally released FlowDec on fixed crops."""

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import wandb

from .codec import BLOCK_SAMPLES, SAMPLE_RATE, SCRATCH, FlowCodec, file_hash
from .data import read_manifest
from .metrics import si_sdr, spectral_loss
from .postfilter import Postfilter
from .train_postfilter import AudioCrops


@torch.inference_mode()
def released_enhance(codec, coarse, nfe, seed, solver):
    condition, _, (scale, frames, length) = codec.preprocess(coarse)
    generator = torch.Generator(device=coarse.device).manual_seed(seed)
    noise = torch.randn(condition.shape, dtype=condition.dtype, device=coarse.device, generator=generator)
    state = condition + codec.sigma * noise
    midpoint = solver == "midpoint" and nfe != 1
    steps = nfe // 2 if midpoint else nfe
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full((len(coarse),), index * dt, device=coarse.device)
        velocity = codec.velocity(state, condition, t)
        if midpoint:
            velocity = codec.velocity(state + 0.5 * dt * velocity, condition, t + 0.5 * dt)
        state = state + dt * velocity
    return codec.features.invert(state[..., :frames], orig_length=length) * scale


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--name", default="postfilter-comparison")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--per-domain", type=int)
    parser.add_argument("--released", action="store_true", help="also measure the released FlowDec-75m")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("--name must be a folder name")
    first = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    config = first["hyper_parameters"]["config"]
    expected_manifest = first["hyper_parameters"]["provenance"]["manifest_sha256"]
    del first
    if file_hash(config["manifest"]) != expected_manifest:
        raise ValueError("The training manifest changed; restore its original split definitions")
    rows = read_manifest(config["manifest"])
    per_domain = args.per_domain or config["test_per_domain"]
    if per_domain < 1:
        raise ValueError("--per-domain must be positive")
    duration = config.get("evaluation_seconds", 2) if args.split == "test" else 2
    if duration < 2 or duration % 2:
        raise ValueError("Evaluation duration must be a positive multiple of two seconds")
    data = AudioCrops(rows, args.split, per_domain=per_domain, seed=0, seconds=duration)
    output = (Path(config["output"]) / args.name).resolve()
    if not output.is_relative_to(SCRATCH.resolve()):
        raise ValueError(f"Write reports under {SCRATCH}")
    cache, audio_dir = output / "measurements", output / "audio"
    cache.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    all_rows, models = [], []
    labels = [path.parent.name for path in args.checkpoints]
    if len(set(labels)) != len(labels):
        raise ValueError("Checkpoint parent directories must have distinct run names")
    for model_index, (checkpoint_path, label) in enumerate(zip(args.checkpoints, labels)):
        checkpoint_hash = file_hash(checkpoint_path)
        model = Postfilter.load_from_checkpoint(checkpoint_path, map_location="cpu").eval().cuda()
        settings = model.hparams.config
        if settings != config or model.hparams.provenance["manifest_sha256"] != expected_manifest:
            raise ValueError("Compare models trained with the same data, bitrate and settings")
        parameters = sum(p.numel() for p in model.network.parameters())
        models.append({"name": label, "geometry": model.hparams.geometry, "seed": model.hparams.seed,
                       "checkpoint": str(checkpoint_path), "sha256": checkpoint_hash,
                       "trainable_parameters": parameters, "fit_seconds": model.fit_seconds})
        released = FlowCodec(config["checkpoints"]).eval().cuda() if args.released and model_index == 0 else None
        if released is not None and released.identity != model.codec.identity:
            raise ValueError("Released reference must use the same NDAC and feature/noise source")
        examples = set()
        for index, row in enumerate(data.rows):
            stat = Path(row["path"]).stat()
            domain = row.get("domain") or "audio"
            export_audio = domain not in examples
            examples.add(domain)
            identity = {"checkpoint": checkpoint_hash, "model": label, "index": index, "source": row,
                        "source_size": stat.st_size, "source_mtime": stat.st_mtime_ns,
                        "split": args.split, "data_seed": 0, "seconds": duration, "released": released is not None,
                        "nfe": config["evaluation_nfe"], "noise_seeds": config["evaluation_seeds"]}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
            record_path = cache / f"{key}.json"
            if record_path.exists():
                all_rows.extend(json.loads(record_path.read_text())["rows"])
                continue
            audio = data[index][None].cuda()
            with torch.inference_mode():
                coarse_blocks = [model.prepare(block)[0] for block in audio.split(BLOCK_SAMPLES, dim=-1)]
                coarse = torch.cat(coarse_blocks, dim=-1)
            values = []
            baseline = {"path": row["path"], "domain": domain, "group": row["group"],
                        "dataset": row.get("dataset", "audio"), "source_sample_rate": row.get("sample_rate", ""),
                        "seconds": duration,
                        "codebooks": config["codebooks"], "nominal_kbps": config["codebooks"] * .75}
            if model_index == 0:
                values.append({**baseline, "model": "ndac", "nfe": 0,
                               "spectral": spectral_loss(audio, coarse).item(),
                               "si_sdr": si_sdr(audio, coarse).item(), "postfilter_seconds": 0.0})
                if export_audio:
                    for tag, waveform in (("original", audio), ("ndac", coarse)):
                        sf.write(audio_dir / f"{index:05d}_{tag}.wav", waveform[0, 0].cpu().numpy(),
                                 SAMPLE_RATE, subtype="FLOAT")
            methods = [(label, model.enhance)]
            if released is not None:
                methods.append(("released_flowdec", lambda coarse, nfe, seed, solver:
                                released_enhance(released, coarse, nfe, seed, solver)))
            for method_name, enhance in methods:
                # Warm up on each source before collecting decoder timings.
                enhance(coarse_blocks[0], 1, 0, config["solver"])
                for nfe in config["evaluation_nfe"]:
                    errors, scores, seconds = [], [], []
                    for noise_index, sample_seed in enumerate(config["evaluation_seeds"]):
                        seed = 1000000 + 100 * index + sample_seed
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        reconstruction = torch.cat([enhance(block, nfe, seed + block_index * 10007, config["solver"])
                                                    for block_index, block in enumerate(coarse_blocks)], dim=-1)
                        torch.cuda.synchronize()
                        seconds.append(time.perf_counter() - start)
                        errors.append(spectral_loss(audio, reconstruction).item())
                        scores.append(si_sdr(audio, reconstruction).item())
                        if export_audio and noise_index == 0:
                            sf.write(audio_dir / f"{index:05d}_{method_name}_nfe{nfe}.wav",
                                     reconstruction[0, 0].cpu().numpy(), SAMPLE_RATE, subtype="FLOAT")
                    values.append({**baseline, "model": method_name, "nfe": nfe,
                                   "spectral": float(np.mean(errors)), "si_sdr": float(np.mean(scores)),
                                   "postfilter_seconds": float(np.mean(seconds))})
            save_json(record_path, {"identity": identity, "rows": values})
            all_rows.extend(values)
            print(f"{label}: {index + 1}/{len(data)} saved", flush=True)
        del model, released
        torch.cuda.empty_cache()
    temporary = output / "metrics.csv.part"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    temporary.replace(output / "metrics.csv")
    summary, by_dataset = [], []
    for field, destination in (("domain", summary), ("dataset", by_dataset)):
        grouped = defaultdict(lambda: defaultdict(list))
        for row in all_rows:
            grouped[(row[field], row["model"], row["nfe"])][row["group"]].append(row)
        for (label, method, nfe), groups in sorted(grouped.items()):
            point = {field: label, "model": method, "nfe": nfe, "groups": len(groups)}
            for metric in ("spectral", "si_sdr", "postfilter_seconds"):
                point[metric] = float(np.mean([np.mean([r[metric] for r in records]) for records in groups.values()]))
            destination.append(point)
    domains = sorted({row["domain"] for row in summary})
    fig, axes = plt.subplots(2, len(domains), figsize=(5 * len(domains), 7), squeeze=False)
    for column, domain in enumerate(domains):
        for row_index, (metric, ylabel) in enumerate((("spectral", "Spectral error (lower is better)"),
                                                     ("si_sdr", "SI-SDR, dB (higher is better)"))):
            ax = axes[row_index, column]
            for method in sorted({row["model"] for row in summary}):
                points = sorted((row for row in summary if row["domain"] == domain and row["model"] == method),
                                key=lambda row: row["nfe"])
                if method == "ndac":
                    ax.axhline(points[0][metric], color="gray", linestyle="--", label="NDAC")
                else:
                    ax.plot([row["nfe"] for row in points], [row[metric] for row in points], "o-", label=method)
            ax.set(title=domain, xlabel="Network evaluations (NFE)", ylabel=ylabel)
            ax.set_xscale("log", base=2)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "quality_vs_nfe.png", dpi=160)
    plt.close(fig)
    save_json(output / "summary.json", {"config": config, "split": args.split, "models": models,
              "results": summary, "by_dataset": by_dataset,
              "notes": ["Means first average recordings within source groups, then average groups.",
                        "NFE=1 uses Euler; other budgets use the configured solver.",
                        "Timing covers the postfilter including STFT/inverse, excluding NDAC and metrics.",
                        "Ten-second test excerpts are decoded in five independent two-second blocks; waveform metrics score the concatenation.",
                        "Released FlowDec has a different backbone and training history; use the trained Cartesian arm to isolate geometry.",
                        "Spectral error and SI-SDR do not establish perceptual quality; listen to audio/."]})
    run_id = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    with wandb.init(project=config["wandb_project"], name=args.name, id=run_id, resume="allow", dir=config["wandb_dir"]):
        wandb.log({"evaluation": wandb.Table(columns=list(summary[0]), data=[list(row.values()) for row in summary]),
                   "quality_vs_nfe": wandb.Image(str(output / "quality_vs_nfe.png"))})
    print(f"Saved {output / 'metrics.csv'}, summary.json, quality_vs_nfe.png and audio/")


if __name__ == "__main__":
    main()
