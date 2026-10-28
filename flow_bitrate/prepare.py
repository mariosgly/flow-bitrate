"""Run the expensive, frozen-codec measurements once, one independent shard per GPU."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .codec import BLOCK_SAMPLES, CODEBOOKS, ROOT, FlowCodec, file_hash
from .data import load_config, read_audio, read_manifest
from .metrics import si_sdr, signal_features, spectral_loss


def block_seed(path, start, seed):
    key = f"{path}:{start}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/experiment.json")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("shard must be in [0, num-shards)")
    config = load_config(args.config)
    if not config["sample_seeds"] or config["steps"] < 1:
        raise ValueError("Need at least one sampling seed and a positive number of steps")
    rows = read_manifest(config["manifest"])
    codec = FlowCodec(config["checkpoints"], config.get("flow_checkpoint")).to(args.device).requires_grad_(False)
    metadata = {
        "schema": 1, "num_shards": args.num_shards, "codec": codec.identity,
        "manifest_sha256": file_hash(config["manifest"]), "block_samples": BLOCK_SAMPLES,
        "steps": config["steps"], "sample_seeds": config["sample_seeds"],
        "source_stats": [{"path": r["path"], "size": Path(r["path"]).stat().st_size,
                          "mtime": Path(r["path"]).stat().st_mtime_ns} for r in rows],
        "base_features": "latent mean/std, log RMS, flux, four latent errors, four coarse spectral errors",
        "flow_features": "per rate: (CFM error, relative CFM error, speed) at .2/.5/.8; sensitivity at .5",
    }
    output = Path(config["cache"])
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"shard_{args.shard:05d}.pt"
    if destination.exists():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        if existing["metadata"] != metadata:
            raise ValueError("Existing shard has different settings. Choose a new cache directory")
        print(f"Already complete: {destination}")
        return
    result = {key: [] for key in ("base", "flow", "loss", "sisdr", "records")}
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for row in tqdm(rows[args.shard::args.num_shards], desc=f"shard {args.shard}"):
            audio = read_audio(row["path"])
            for start in range(0, audio.shape[-1] - BLOCK_SAMPLES + 1, BLOCK_SAMPLES):
                original = audio[..., start:start + BLOCK_SAMPLES][None].to(args.device)
                latent, codes = codec.encode(original)
                if codes.shape[-1] != 150:
                    raise ValueError("Expected 150 RVQ frames per two-second block")
                errors, coarse_errors, flow, losses, scores = [], [], [], [], []
                for depth in CODEBOOKS:
                    coarse, quantized = codec.coarse(codes, depth)
                    errors.append((latent - quantized).square().mean((1, 2)).log1p())
                    coarse_errors.append(spectral_loss(original, coarse))
                    flow.append(codec.probes(original, coarse, block_seed(row["path"], start, "probe")))
                    samples, sample_scores = [], []
                    for seed in config["sample_seeds"]:
                        file_seed = block_seed(row["path"], 0, seed)
                        decoded = codec.enhance(coarse, block_seed("", start, file_seed), config["steps"])
                        samples.append(spectral_loss(original, decoded))
                        sample_scores.append(si_sdr(original, decoded))
                    losses.append(torch.stack(samples).mean(0))
                    scores.append(torch.stack(sample_scores).mean(0))
                base = torch.cat((latent.mean(-1), latent.std(-1, correction=0), signal_features(original),
                                  torch.stack(errors, 1), torch.stack(coarse_errors, 1)), 1)
                result["base"].append(base.cpu())  # [1, 2058]
                result["flow"].append(torch.cat(flow, 1).cpu())  # [1, 40]
                result["loss"].append(torch.stack(losses, 1).cpu())
                result["sisdr"].append(torch.stack(scores, 1).cpu())
                result["records"].append({**row, "start": start})
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    for key in ("base", "flow", "loss", "sisdr"):
        result[key] = torch.cat(result[key]) if result[key] else torch.empty(0)
        if not torch.isfinite(result[key]).all():
            raise FloatingPointError(f"Non-finite {key}; shard was not saved")
    result.update(metadata=metadata, shard=args.shard, elapsed_seconds=time.perf_counter() - started)
    temporary = destination.with_suffix(".pt.part")
    torch.save(result, temporary)
    temporary.replace(destination)
    print(json.dumps({"saved": str(destination), "blocks": len(result["records"]),
                      "seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
