"""Read the diagnostic cache, compare bits/steps, and measure allocation headroom."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import wandb

from .allocation import allocate, stream_bits, uniform_choices
from .codec import CODEBOOKS, ROOT
from .data import load_config
from .sweep import commit_file, diagnostic_rows, sweep_settings


def group_values(rows, key):
    groups = defaultdict(list)
    for row in rows:
        if row[key] is not None:
            groups[(row["domain"], row["group"])].append(row[key])
    return np.array([np.mean(values) for values in groups.values()])


def interval(values, seed):
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    return np.quantile(bootstrap, [0.025, 0.975]).tolist()


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".part")
    if rows:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        temporary.write_text("")
    commit_file(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/diagnostic.json")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = sweep_settings(config)
    rows = diagnostic_rows(config)
    if config["min_oracle_blocks"] < 2 or config["shuffle_repeats"] < 1:
        raise ValueError("Need at least two blocks per oracle recording and positive shuffle_repeats")
    if not config["budgets_kbps"] or any(not np.isfinite(b) or b <= 0 for b in config["budgets_kbps"]):
        raise ValueError("Need positive finite bitrate budgets")
    output = Path(config["output"]) / config["name"]
    output.mkdir(parents=True, exist_ok=True)
    nfe_values = [0] + config["nfe"] + [config["reference_nfe"]]
    selection_count = len(config["selection_seeds"])
    grid, measurements, effects, oracles, excluded = [], [], [], [], []
    metadata = None
    for index, row in enumerate(rows):
        path = Path(config["cache"]) / f"record_{index:05d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}; finish all sweep jobs before analysis")
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record["record"] != row or record["metadata"]["settings"] != settings:
            raise ValueError(f"Different manifest or sweep settings in {path}")
        if metadata is not None and metadata != record["metadata"]:
            raise ValueError("Mixed checkpoints/settings in diagnostic cache")
        metadata = record["metadata"]
        count = record["blocks"]
        identity = {"record_index": index, "path": row["path"], "domain": row["domain"], "group": row["group"]}
        if not count:
            excluded.append({**identity, "reason": "no full two-second block"})
            continue
        # Axis 2 now includes NDAC-only at NFE=0. It has no sampling randomness.
        selected = np.concatenate((record["coarse_loss"].numpy()[:, :, None],
                                   record["loss"][..., :selection_count].mean(-1).numpy()), axis=2)
        evaluated = np.concatenate((record["coarse_loss"].numpy()[:, :, None],
                                    record["loss"][..., selection_count:].mean(-1).numpy()), axis=2)
        sisdr = np.concatenate((record["coarse_sisdr"].numpy()[:, :, None],
                               record["sisdr"][..., selection_count:].mean(-1).numpy()), axis=2)
        seconds = np.concatenate((record["coarse_seconds"].numpy()[:, :, None],
                                  record["seconds"][..., selection_count:].mean(-1).numpy()
                                  + record["coarse_seconds"].numpy()[:, :, None]), axis=2)
        error = record["nmse_reference"][..., selection_count:].mean(-1).numpy()
        for point, nfe in enumerate(nfe_values):
            for rate, depth in enumerate(CODEBOOKS):
                kbps = stream_bits(np.full(count, rate)) / (2000 * count)
                grid.append({**identity, "blocks": count, "codebooks": depth, "nfe": nfe,
                             "kbps": kbps, "loss": float(evaluated[:, rate, point].mean()),
                             "si_sdr": float(sisdr[:, rate, point].mean()),
                             "nmse_reference": float(error[:, rate, point - 1].mean()) if point else None,
                             "decoder_rtf": float(seconds[:, rate, point].mean() / 2)})
                for block in range(count):
                    measurements.append({**identity, "block": block,
                                         "start_seconds": row["start_seconds"] + 2 * block,
                                         "codebooks": depth, "nfe": nfe,
                                         "selection_loss": float(selected[block, rate, point]),
                                         "evaluation_loss": float(evaluated[block, rate, point]),
                                         "si_sdr": float(sisdr[block, rate, point]),
                                         "nmse_reference": float(error[block, rate, point - 1]) if point else None,
                                         "decoder_seconds": float(seconds[block, rate, point])})
            for rate in range(len(CODEBOOKS) - 1):
                gain = evaluated[:, rate, point] - evaluated[:, rate + 1, point]
                effects.append({**identity, "kind": "add_codebooks", "from_value": CODEBOOKS[rate],
                                "to_value": CODEBOOKS[rate + 1], "fixed_nfe": nfe, "fixed_codebooks": None,
                                "loss_reduction": float(gain.mean()), "fraction_blocks_improved": float((gain > 0).mean())})
            # NFE=0 changes the decoder, rather than the accuracy of one ODE solve.
            if point > 1:
                for rate, depth in enumerate(CODEBOOKS):
                    gain = evaluated[:, rate, point - 1] - evaluated[:, rate, point]
                    effects.append({**identity, "kind": "add_evaluations", "from_value": nfe_values[point - 1],
                                    "to_value": nfe, "fixed_nfe": None, "fixed_codebooks": depth,
                                    "loss_reduction": float(gain.mean()), "fraction_blocks_improved": float((gain > 0).mean())})
            if count < config["min_oracle_blocks"]:
                continue
            for budget in config["budgets_kbps"]:
                uniform = uniform_choices(count, budget)
                feasible = [level for level in range(4) if stream_bits(np.full(count, level)) <= budget * count * 2000]
                best_level = min(feasible, key=lambda level: selected[:, level, point].mean())
                best_uniform = np.full(count, best_level)
                choices = allocate(selected[:, :, point], budget)
                indices = np.arange(count)
                oracle_loss = float(evaluated[indices, choices, point].mean())
                rng = np.random.default_rng(config["seed"] + index)
                shuffled_loss = np.mean([evaluated[indices, rng.permutation(choices), point].mean()
                                         for _ in range(config["shuffle_repeats"])])
                uniform_loss = float(evaluated[indices, uniform, point].mean())
                best_loss = float(evaluated[indices, best_uniform, point].mean())
                oracles.append({**identity, "blocks": count, "nfe": nfe, "budget_kbps": budget,
                                "uniform_loss": uniform_loss, "best_uniform_loss": best_loss,
                                "oracle_selection_loss": float(selected[indices, choices, point].mean()),
                                "oracle_evaluation_loss": oracle_loss,
                                "uniform_gain": uniform_loss - oracle_loss,
                                "best_uniform_gain": best_loss - oracle_loss,
                                "placement_gain": float(shuffled_loss - oracle_loss),
                                "uniform_kbps": stream_bits(uniform) / (count * 2000),
                                "oracle_kbps": stream_bits(choices) / (count * 2000),
                                "rate_indices": json.dumps(choices.tolist())})
        if count < config["min_oracle_blocks"]:
            excluded.append({**identity, "reason": "too few blocks for allocation", "blocks": count})
    if not grid:
        raise ValueError("No full blocks were measured")
    summary = {"grid": [], "effects": [], "oracle": []}
    for domain in ["all"] + sorted({row["domain"] for row in grid}):
        for section, values, keys, metrics in (
            ("grid", grid, ("codebooks", "nfe"), ("kbps", "loss", "si_sdr", "nmse_reference", "decoder_rtf")),
            ("effects", effects, ("kind", "from_value", "to_value", "fixed_nfe", "fixed_codebooks"),
             ("loss_reduction", "fraction_blocks_improved")),
            ("oracle", oracles, ("nfe", "budget_kbps"),
             ("uniform_loss", "best_uniform_loss", "oracle_evaluation_loss", "uniform_gain", "best_uniform_gain",
              "placement_gain", "uniform_kbps", "oracle_kbps")),
        ):
            buckets = defaultdict(list)
            for value in values:
                if domain == "all" or value["domain"] == domain:
                    buckets[tuple(value[key] for key in keys)].append(value)
            for key, bucket in buckets.items():
                entry = {"domain": domain, **dict(zip(keys, key)), "files": len(bucket),
                         "groups": len({(r["domain"], r["group"]) for r in bucket})}
                for metric in metrics:
                    estimates = group_values(bucket, metric)
                    entry[metric] = float(estimates.mean()) if len(estimates) else None
                    if metric in {"uniform_gain", "best_uniform_gain", "placement_gain", "loss_reduction"}:
                        entry[metric + "_95ci"] = interval(estimates, config["seed"])
                summary[section].append(entry)
    report = {"config": config, "metadata": metadata, "summary": summary, "exclusions": excluded,
              "notes": [
                  "Positive gain means lower multi-resolution log-spectral error. SI-SDR is secondary.",
                  "Oracle choices see the original and selection seeds; evaluation uses disjoint noise seeds.",
                  "Cross-seed oracle gains may be negative. This is not a strict lower bound on evaluation error.",
                  "Placement gain compares against shuffled assignments with the exact same rate histogram.",
                  "Means and bootstrap intervals use independent groups, not individual blocks. All is group-weighted.",
                  "Reference is a higher-NFE solve, not ground truth. Check 48-to-96 (or configured last-to-reference) changes.",
                  "NMSE to reference uses original waveform power; reference-to-itself is zero by construction.",
                  "Decoder RTF includes RVQ reconstruction and flow, excludes encoder/I/O/metrics; GPU warmup excluded.",
                  "Independent mono two-second blocks; tails dropped. Metrics do not establish perceptual quality.",
                  "NDAC output is right-zero-padded/cropped to 96000 samples before flow and metrics; policy is in codec identity.",
              ]}
    for name, values in (("grid", grid), ("measurements", measurements), ("effects", effects), ("oracle", oracles)):
        write_csv(output / f"{name}.csv", values)
    temporary = output / "summary.json.part"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    commit_file(temporary, output / "summary.json")
    with wandb.init(project=config["wandb_project"], name=f"{config['name']}-oracle", group=config["name"],
                    job_type="diagnostic-analysis", dir=config["wandb_dir"], config=config) as run:
        for name, values in summary.items():
            if values:
                columns = list(values[0])
                run.log({name: wandb.Table(columns=columns, data=[[r[column] for column in columns] for r in values])})
        for domain in ["all"] + sorted({row["domain"] for row in grid}):
            curves = [[next(row["loss"] for row in summary["grid"]
                            if row["domain"] == domain and row["codebooks"] == depth and row["nfe"] == nfe)
                       for nfe in nfe_values] for depth in CODEBOOKS]
            run.log({f"{domain}/rate_compute": wandb.plot.line_series(
                xs=nfe_values, ys=curves, keys=[f"{depth} codebooks" for depth in CODEBOOKS],
                title=f"{domain}: spectral loss vs evaluations (0 = NDAC)", xname="NFE")})
    print(f"Saved {output / 'summary.json'} and grid/measurements/effects/oracle.csv")
    if not oracles:
        print("No allocation results: need recordings with enough full blocks for min_oracle_blocks.")
    for row in summary["oracle"]:
        if row["domain"] == "all":
            print(f"NFE={row['nfe']:>2}, budget={row['budget_kbps']:.2f} kbps: "
                  f"uniform gain={row['uniform_gain']:.4f}, placement gain={row['placement_gain']:.4f}")


if __name__ == "__main__":
    main()
