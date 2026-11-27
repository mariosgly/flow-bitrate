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
from .data import matched_seed, read_manifest
from .jamendo_comparison import artist_reports, blind_packet
from .metrics import si_sdr, spectral_loss
from .postfilter import Postfilter
from .sweep import save_audio
from .train_postfilter import AudioCrops, JamendoCrops


@torch.inference_mode()
def released_enhance(codec, coarse, nfe, seed, solver, *, matched_noise=False):
    solver = "euler" if nfe == 1 else solver
    steps = nfe // 2 if solver == "midpoint" else nfe
    return codec.enhance(coarse, seed=seed, steps=steps, solver=solver, matched_noise=matched_noise)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoints", type=Path, nargs="+")
    inputs.add_argument("--score-plan", type=Path, help="Jamendo plan.json with frozen checkpoints and cohort")
    parser.add_argument("--name", default="postfilter-comparison")
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--per-domain", type=int)
    parser.add_argument("--nfe", type=int, nargs="+", help="Evaluate these network-call budgets")
    parser.add_argument("--require-update", type=int, help="Require every checkpoint to contain this optimizer update")
    parser.add_argument("--cohort-seed", type=int, help="Fixed Jamendo cohort, including across training seeds")
    parser.add_argument("--listening-tracks", type=int, default=0,
                        help="Prepare a blind A/B packet for the first N Jamendo excerpts and two checkpoints")
    parser.add_argument("--export-audio-tracks", type=int,
                        help="Save labeled WAVs for the first N excerpts for every model (default: one per domain)")
    parser.add_argument("--released", action="store_true", help="also measure the released FlowDec-75m")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("--name must be a folder name")
    if args.export_audio_tracks is not None and args.export_audio_tracks < 0:
        parser.error("--export-audio-tracks must be nonnegative")
    score_plan = None
    if args.score_plan is not None:
        plan_path = args.score_plan.resolve()
        score_plan = json.loads(plan_path.read_text())
        if any(model.get("released_flowdec") for model in score_plan["models"]):
            parser.error("For audio with released FlowDec, pass local postfilter checkpoints and --released")
        snapshots = plan_path.parent.parent.parent / "snapshots"
        checkpoints = [snapshots / model["snapshot"] for model in score_plan["models"]]
        labels = [f"{Path(model['source']).parent.name}-{model['sha256'][:12]}" for model in score_plan["models"]]
        if not checkpoints:
            parser.error("The scoring plan contains no frozen checkpoints")
    else:
        checkpoints = args.checkpoints
        labels = [path.parent.name for path in checkpoints]
    plan_split = score_plan.get("split", "test") if score_plan is not None else "test"
    args.split = args.split or plan_split
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    config = first["hyper_parameters"]["config"]
    provenance = first["hyper_parameters"]["provenance"]
    expected_manifest = provenance["manifest_sha256"]
    training_seed = first["hyper_parameters"]["seed"]
    del first
    jamendo = config.get("dataset") == "jamendo"
    solver = config["solver"]
    noise_seeds = config["evaluation_seeds"]
    default_budgets = config["evaluation_nfe"]
    if score_plan is not None:
        if not jamendo or args.split != plan_split or score_plan["manifest_sha256"] != expected_manifest:
            parser.error("--score-plan requires Jamendo checkpoints on the plan's manifest and split")
        solver = score_plan["solver"]
        noise_seeds = score_plan["seeds"]
        default_budgets = [score_plan["nfe"]]
    budgets = args.nfe if args.nfe is not None else default_budgets
    if len(set(budgets)) != len(budgets) or any(n < 1 or (solver == "midpoint" and n != 1 and n % 2) for n in budgets):
        parser.error("Use distinct positive NFE budgets; midpoint accepts one or even budgets")
    if args.listening_tracks < 0 or (args.listening_tracks and (not jamendo or len(checkpoints) != 2)):
        parser.error("--listening-tracks requires two Jamendo checkpoints and a nonnegative count")
    if args.cohort_seed is not None and not jamendo:
        parser.error("--cohort-seed is for Jamendo")
    if file_hash(config["manifest"]) != expected_manifest:
        raise ValueError("The training manifest changed; restore its original split definitions")
    rows = score_plan["tracks"] if score_plan is not None else read_manifest(config["manifest"], dataset=config.get("dataset"))
    count_key = "validation_per_domain" if jamendo and args.split == "val" else "test_per_domain"
    per_domain = args.per_domain if args.per_domain is not None else config[count_key]
    if per_domain < 1:
        raise ValueError("--per-domain must be positive")
    duration = config.get("evaluation_seconds", 2) if jamendo or args.split == "test" else 2
    if duration < 2 or duration % 2:
        raise ValueError("Evaluation duration must be a positive multiple of two seconds")
    data_seed = training_seed if jamendo and score_plan is None else 0
    if args.cohort_seed is not None:
        data_seed = args.cohort_seed
    crop_type = JamendoCrops if jamendo else AudioCrops
    data = crop_type(rows, args.split, per_domain=per_domain, seed=data_seed, seconds=duration)
    output = (Path(config["output"]) / args.name).resolve()
    if not output.is_relative_to(SCRATCH.resolve()):
        raise ValueError(f"Write reports under {SCRATCH}")
    cache, audio_dir = output / "measurements", output / "audio"
    cache.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    all_rows, models = [], []
    blind_sources = {}
    if len(set(labels)) != len(labels):
        raise ValueError("Checkpoints must have distinct run labels")
    arm_settings = ({"backbone", "ema", "batch_size", "accumulate", "phase_loss", "phase_calibration", "tau_multiplier",
                     "width", "time_embedding_dim", "optimizer_recipe", "max_updates", "ablation_arm", "architecture",
                     "schedule", "start_noise", "noise_calibration", "ema_decay", "effective_batch", "validate_every",
                     "band_width", "highband_loss_weight"}
                    if jamendo else {"ema", "ema_decay"})
    expected_settings = {key: value for key, value in config.items() if key not in arm_settings}
    if jamendo:
        expected_batch = config["batch_size"] * config["accumulate"] * provenance["world_size"]
    for model_index, (checkpoint_path, label) in enumerate(zip(checkpoints, labels)):
        checkpoint_hash = file_hash(checkpoint_path)
        if score_plan is not None and checkpoint_hash != score_plan["models"][model_index]["sha256"]:
            raise ValueError(f"Frozen checkpoint does not match the scoring plan: {checkpoint_path}")
        model = Postfilter.load_from_checkpoint(checkpoint_path, map_location="cpu").eval().cuda()
        if args.require_update is not None and model.checkpoint_step != args.require_update:
            raise ValueError(f"{label} is at optimizer update {model.checkpoint_step}; requested {args.require_update}")
        settings = model.hparams.config
        shared_settings = {key: value for key, value in settings.items() if key not in arm_settings}
        if shared_settings != expected_settings or model.hparams.provenance["manifest_sha256"] != expected_manifest:
            raise ValueError("Compare models from the same dataset, rate and evaluation protocol")
        if jamendo and ((args.cohort_seed is None and model.hparams.seed != training_seed) or any(
                model.hparams.provenance[key] != provenance[key] for key in ("sources", "world_size"))):
            raise ValueError("Match sources and GPU counts. To compare different training seeds, set --cohort-seed.")
        if models and model.codec.identity != models[0]["codec"]:
            raise ValueError("Compare checkpoints with the same frozen codec and noise assets")
        models.append({"name": label, "backbone": model.backbone_name,
                       "geometry": model.hparams.geometry, "seed": model.hparams.seed,
                       "phase_loss": model.phase_loss, "tau_multiplier": settings.get("tau_multiplier"),
                       "phase_weight_mean": settings.get("phase_calibration", {}).get("mean_q"),
                       "codec": model.codec.identity, "step": model.checkpoint_step,
                       "checkpoint": str(checkpoint_path), "sha256": checkpoint_hash,
                       "evaluation_weights": "ema" if model.ema_network is not None else "raw",
                       "ema_decay": settings.get("ema_decay") if model.ema_network is not None else None,
                       "ema_updates": model.ema_updates.item() if model.ema_updates is not None else 0,
                       "trainable_parameters": model.trainable_parameters, "fit_seconds": model.fit_seconds})
        if jamendo:
            effective_batch = settings["batch_size"] * settings["accumulate"] * provenance["world_size"]
            if effective_batch != expected_batch and not (
                    config.get("ablation_arm") and settings.get("ablation_arm")):
                raise ValueError("Jamendo checkpoints must have the same effective training batch size")
            architecture = settings.get("architecture", "unet" if model.backbone_name == "cyfm" else "ncsnpp")
            cyfm = model.backbone_name == "cyfm" and architecture not in {"band-tcn", "band-attention", "ncsnpp"}
            models[-1].update(batch_size_per_gpu=settings["batch_size"], accumulate=settings["accumulate"],
                              effective_batch_size=effective_batch, optimizer_recipe=model.optimizer_recipe,
                              architecture=architecture, start_noise=settings.get("start_noise", "original"),
                              band_width=settings.get("band_width", 256) if architecture.startswith("band-") else None,
                              highband_loss_weight=settings.get("highband_loss_weight", 0.0),
                              schedule=settings.get("schedule"), crop_slots=model.checkpoint_step * effective_batch,
                              cyfm_width=settings["width"] if cyfm else None,
                              cyfm_time_width=settings.get("time_embedding_dim", 4 * settings["width"]) if cyfm else None)
        released = FlowCodec(config["checkpoints"]).eval().cuda() if args.released and model_index == 0 else None
        if released is not None:
            if released.identity["ndac"] != model.codec.identity["ndac"]:
                raise ValueError("Released reference must use the same NDAC weights")
            if not jamendo and released.identity != model.codec.identity:
                raise ValueError("Released reference must use the same feature/noise source")
        examples = set()
        for index, row in enumerate(data.rows):
            stat = Path(row["path"]).stat()
            if jamendo and row.get("file_size") and (
                    stat.st_size != int(row["file_size"]) or stat.st_mtime_ns != int(row["mtime_ns"])):
                raise ValueError(f"Audio changed since its decode audit: {row['path']}")
            domain = row.get("domain") or "audio"
            export_audio = domain not in examples
            if args.export_audio_tracks is not None:
                export_audio = index < args.export_audio_tracks
            examples.add(domain)
            identity = {"checkpoint": checkpoint_hash, "model": label, "index": index, "source": row,
                        "source_size": stat.st_size, "source_mtime": stat.st_mtime_ns,
                        "split": args.split, "data_seed": data_seed, "seconds": duration,
                        "released": released.identity["flow"] if released is not None else False,
                        "whole_excerpt": jamendo, "include_ndac": model_index == 0, "solver": solver,
                        "nfe": budgets, "noise_seeds": noise_seeds}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
            record_path = cache / f"{key}.json"
            listening = index < args.listening_tracks
            required_wavs = []
            if export_audio:
                exported_models = (label, "released_flowdec") if released is not None else (label,)
                required_wavs.extend(audio_dir / f"{index:05d}_{name}_nfe{n}.wav"
                                     for name in exported_models for n in budgets)
                if model_index == 0:
                    required_wavs.extend(audio_dir / f"{index:05d}_{tag}.wav" for tag in ("original", "ndac"))
            if listening:
                clip_dir = cache / f"{key}-audio"
                blind_sources[index, label] = clip_dir
                required_wavs.extend(clip_dir / f"nfe{n}.wav" for n in budgets)
                if model_index == 0:
                    required_wavs.extend([clip_dir / "reference.wav", clip_dir / "ndac.wav"])
            if record_path.exists() and all(path.is_file() for path in required_wavs):
                all_rows.extend(json.loads(record_path.read_text())["rows"])
                continue
            sample = data[index]
            waveform, track = sample if jamendo else (sample, None)
            audio = waveform[None].cuda()
            with torch.inference_mode():
                blocks = (audio,) if jamendo else audio.split(BLOCK_SAMPLES, dim=-1)
                coarse_blocks = [model.coarse_audio(block) for block in blocks]
                coarse = torch.cat(coarse_blocks, dim=-1)
            values = []
            baseline = {"path": row["path"], "domain": domain, "group": row["group"],
                        "dataset": row.get("dataset", "audio"), "source_sample_rate": row.get("sample_rate", ""),
                        "seconds": duration,
                        "codebooks": config["codebooks"], "nominal_kbps": config["codebooks"] * .75}
            if model_index == 0:
                if listening:
                    save_audio(clip_dir / "reference.wav", audio.flatten().cpu())
                    save_audio(clip_dir / "ndac.wav", coarse.flatten().cpu())
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
                                released_enhance(released, coarse, nfe, seed, solver, matched_noise=jamendo)))
            for method_name, enhance in methods:
                # Warm up on each source before collecting decoder timings.
                enhance(coarse_blocks[0], 1, 0, solver)
                for nfe in budgets:
                    errors, scores, seconds = [], [], []
                    for noise_index, sample_seed in enumerate(noise_seeds):
                        seed = matched_seed("evaluation", track, sample_seed) if jamendo else 1000000 + 100 * index + sample_seed
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        reconstruction = torch.cat([enhance(block, nfe, seed + block_index * 10007, solver)
                                                    for block_index, block in enumerate(coarse_blocks)], dim=-1)
                        torch.cuda.synchronize()
                        seconds.append(time.perf_counter() - start)
                        errors.append(spectral_loss(audio, reconstruction).item())
                        scores.append(si_sdr(audio, reconstruction).item())
                        if listening and noise_index == 0:
                            save_audio(clip_dir / f"nfe{nfe}.wav", reconstruction.flatten().cpu())
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
    if jamendo:
        paired = (dict(method=r["model"], artist=r["group"], track=r["path"], nfe=r["nfe"], seed=None,
                       spectral=r["spectral"], si_sdr=r["si_sdr"]) for r in all_rows)
        artist_reports(output, paired, ("spectral", "si_sdr"))
    if args.listening_tracks:
        packet = blind_packet(output, data.rows[:args.listening_tracks], blind_sources, models, budgets,
                              noise_seeds[0])
        print(f"Blind listening packet: {packet / 'listener'}")
    summary, by_dataset = [], []
    for field, destination in (("domain", summary), ("dataset", by_dataset)):
        grouped = defaultdict(lambda: defaultdict(list))
        for row in all_rows:
            grouped[(row[field], row["model"], row["nfe"])][row["group"]].append(row)
        for (label, method, nfe), groups in sorted(grouped.items()):
            point = {field: label, "model": method, "nfe": nfe, "groups": len(groups)}
            for metric in ("spectral", "si_sdr", "postfilter_seconds"):
                if jamendo:
                    values = [record[metric] for records in groups.values() for record in records]
                else:
                    values = [np.mean([record[metric] for record in records]) for records in groups.values()]
                point[metric] = float(np.mean(values))
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
    duration_note = ("Each ten-second excerpt is encoded and postfiltered in one pass." if jamendo else
                     "Ten-second test excerpts are decoded in five independent two-second blocks; waveform metrics score the concatenation.")
    comparison_note = ("Compare geometry within one CyFM capacity and optimizer. See each model's size, optimizer and step."
                       if jamendo else "Released FlowDec has a different backbone and training history; use the trained Cartesian arm to isolate geometry.")
    save_json(output / "summary.json", {"config": expected_settings if jamendo else config,
              "split": args.split, "cohort_seed": data_seed, "nfe": budgets, "models": models,
              "results": summary, "by_dataset": by_dataset,
              "notes": ["Means average tracks equally." if jamendo else
                        "Means first average recordings within source groups, then average groups.",
                        "NFE=1 uses Euler; other budgets use the configured solver.",
                        "Timing covers the postfilter including STFT/inverse, excluding NDAC and metrics.",
                        duration_note, comparison_note,
                        "Spectral error and SI-SDR do not establish perceptual quality; listen to audio/."]})
    run_id = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    with wandb.init(project=config["wandb_project"], name=args.name, id=run_id, resume="allow", dir=config["wandb_dir"]):
        wandb.log({"evaluation": wandb.Table(columns=list(summary[0]), data=[list(row.values()) for row in summary]),
                   "quality_vs_nfe": wandb.Image(str(output / "quality_vs_nfe.png"))})
    print(f"Saved {output / 'metrics.csv'}, summary.json, quality_vs_nfe.png and audio/")


if __name__ == "__main__":
    main()
