"""Single-process held-out evaluation. Allocate within each recording, never across files."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import wandb

from .allocation import allocate, pack_codes, stream_bits, uniform_choices, unpack_codes
from .codec import BLOCK_SAMPLES, CODEBOOKS, ROOT, SAMPLE_RATE, FlowCodec
from .data import check_manifest, load_cache, load_config, read_audio
from .prepare import block_seed
from .train import RatePredictor


def confidence_interval(values, groups, seed=42):
    # Resample independent groups (songs/speakers), not correlated two-second blocks.
    group_values = defaultdict(list)
    for value, group in zip(values, groups):
        group_values[group].append(value)
    means = np.array([np.mean(x) for x in group_values.values()])
    if len(means) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(len(means), size=(2000, len(means)))].mean(1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/experiment.json")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--flow", required=True, type=Path)
    parser.add_argument("--name", default="evaluation")
    parser.add_argument("--audio", type=int, default=0, help="also save actual bitstreams and WAVs for N files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    cache = load_cache(config["cache"])
    check_manifest(cache, config["manifest"])
    features = torch.cat((cache["base"], cache["flow"]), 1)
    predictions, inference_seconds = {}, {}
    for label, checkpoint, use_flow in (("learned_baseline", args.baseline, False), ("learned_flow", args.flow, True)):
        model = RatePredictor.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False).eval()
        if model.hparams.provenance != cache["metadata"] or model.hparams.use_flow != use_flow:
            raise ValueError(f"Wrong cache or feature ablation for {checkpoint}")
        started = time.perf_counter()
        with torch.inference_mode():
            predictions[label] = torch.cat([model(x).expm1().clamp_min(0) for x in features.split(1024)]).numpy()
        inference_seconds[label] = time.perf_counter() - started
    output = Path(config["output"]) / args.name
    output.mkdir(parents=True, exist_ok=True)
    recordings = defaultdict(list)
    for index, row in enumerate(cache["records"]):
        if row["split"] == "test":
            recordings[row["path"]].append(index)
    if not recordings:
        raise ValueError("No test recordings")
    codec = None
    if args.audio:
        codec = FlowCodec(config["checkpoints"], config.get("flow_checkpoint")).to(args.device).requires_grad_(False)
        if codec.identity != cache["metadata"]["codec"]:
            raise ValueError("Audio decoder differs from the cached decoder")
    records, choices_to_save = [], []
    rng = np.random.default_rng(config["seed"])
    for file_number, (path, indices) in enumerate(sorted(recordings.items())):
        indices.sort(key=lambda i: cache["records"][i]["start"])
        count = len(indices)
        true_loss = cache["loss"][indices].numpy()
        probes = cache["flow"][indices].numpy().reshape(count, 4, 10)
        costs = {
            "latent_residual": cache["base"][indices, -8:-4].numpy(),
            "coarse_error": cache["base"][indices, -4:].numpy(),
            "flow_error": probes[:, :, [0, 3, 6]].mean(2),
            "flow_sensitivity": probes[:, :, 9],
            "random": rng.random((count, 4)), "oracle": true_loss,
            **{key: value[indices] for key, value in predictions.items()},
        }
        selections = {key: allocate(value, config["budget_kbps"]) for key, value in costs.items()}
        selections["uniform"] = uniform_choices(count, config["budget_kbps"])
        row = {"path": path, "group": cache["records"][indices[0]]["group"], "blocks": count}
        for method, choice in selections.items():
            row[method] = {"spectral_loss": float(true_loss[np.arange(count), choice].mean()),
                           "si_sdr": float(cache["sisdr"][indices].numpy()[np.arange(count), choice].mean()),
                           "kbps": stream_bits(choice) / (1000 * count * 2)}
            choices_to_save.append({"path": path, "method": method, "rate_indices": choice.tolist()})
        records.append(row)
        if codec is not None and file_number < args.audio:
            original = read_audio(path)[..., :count * BLOCK_SAMPLES]
            directory = output / f"audio_{file_number:03d}"
            directory.mkdir(exist_ok=True)
            sf.write(directory / "original.wav", original[0].numpy(), SAMPLE_RATE, subtype="FLOAT")
            with torch.inference_mode():
                codes = [codec.encode(block[None].to(args.device))[1][0].cpu().numpy()
                         for block in original.split(BLOCK_SAMPLES, dim=-1)]
                for method in ("uniform", "learned_baseline", "learned_flow", "oracle"):
                    file_seed = block_seed(path, 0, config["sample_seeds"][0])
                    stream = pack_codes(np.stack(codes), selections[method], seed=file_seed)
                    assert len(stream) * 8 == stream_bits(selections[method])
                    (directory / f"{method}.fbr").write_bytes(stream)
                    transmitted, choices, file_seed = unpack_codes(stream)
                    decoded = []
                    for index, (code, choice) in enumerate(zip(transmitted, choices)):
                        code = torch.from_numpy(code)[None].long().to(args.device)
                        coarse = codec.coarse(code, CODEBOOKS[choice])[0]
                        seed = block_seed("", index * BLOCK_SAMPLES, file_seed)
                        decoded.append(codec.enhance(coarse, seed, config["steps"])[0, 0].cpu())
                    waveform = torch.cat(decoded).numpy()
                    sf.write(directory / f"{method}.wav", waveform, SAMPLE_RATE, subtype="FLOAT")
                    row[method]["boundary_jump"] = float(np.mean(np.abs(
                        waveform[BLOCK_SAMPLES::BLOCK_SAMPLES] - waveform[BLOCK_SAMPLES - 1:-1:BLOCK_SAMPLES]
                    ))) if count > 1 else 0.0
    methods = list(records[0].keys())[3:]
    summary = {}
    groups = [row["group"] for row in records]
    for method in methods:
        values = [row[method]["spectral_loss"] for row in records]
        gains = [row["uniform"]["spectral_loss"] - row[method]["spectral_loss"] for row in records]
        by_group = defaultdict(list)
        for value, gain, group in zip(values, gains, groups):
            by_group[group].append((value, gain))
        group_means = np.stack([np.mean(x, axis=0) for x in by_group.values()])
        summary[method] = {"spectral_loss_group_mean": float(group_means[:, 0].mean()),
                           "improvement_over_uniform": float(group_means[:, 1].mean()),
                           "improvement_95ci": confidence_interval(gains, groups, config["seed"]),
                           "kbps_file_mean": float(np.mean([row[method]["kbps"] for row in records]))}
    paired = [row["learned_baseline"]["spectral_loss"] - row["learned_flow"]["spectral_loss"] for row in records]
    paired_groups = defaultdict(list)
    for gain, group in zip(paired, groups):
        paired_groups[group].append(gain)
    summary["learned_flow"]["improvement_over_learned_baseline"] = float(np.mean([np.mean(x) for x in paired_groups.values()]))
    summary["learned_flow"]["baseline_improvement_95ci"] = confidence_interval(paired, groups, config["seed"])
    report = {"summary": summary, "files": records, "config": config,
              "metadata": cache["metadata"], "head_cpu_seconds_all_splits": inference_seconds,
              "note": "Cached losses average sampling seeds. WAVs use only the first seed. Blocks reset independently."}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "choices.json").write_text(json.dumps(choices_to_save, indent=2) + "\n")
    with wandb.init(project=config["wandb_project"], name=args.name, job_type="evaluation", config=config) as run:
        run.log({f"{method}/{metric}": value for method, values in summary.items()
                 for metric, value in values.items() if isinstance(value, (int, float))})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
