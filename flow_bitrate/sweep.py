"""Frozen FlowDec, four RVQ depths, several midpoint budgets. No training or probes."""

import argparse
import json
import math
import time
from pathlib import Path

import soundfile as sf
import torch
import wandb
from tqdm import tqdm

from .codec import BLOCK_SAMPLES, CODEBOOKS, ROOT, SAMPLE_RATE, FlowCodec, file_hash
from .data import load_config, read_audio, read_manifest
from .metrics import si_sdr, spectral_loss
from .prepare import block_seed


def diagnostic_rows(config):
    rows = [row for row in read_manifest(config["manifest"]) if row["split"] == config["split"]]
    if not rows:
        raise ValueError(f"No {config['split']} recordings in the manifest")
    for row in rows:
        if row.get("domain") not in {"speech", "music", "sound"}:
            raise ValueError("Diagnostic manifest needs domain=speech, music, or sound")
        row["start_seconds"] = float(row.get("start_seconds") or 0)
        if not math.isfinite(row["start_seconds"]) or row["start_seconds"] < 0:
            raise ValueError("start_seconds must be finite and nonnegative")
    return rows


def sweep_settings(config):
    nfe = config["nfe"]
    reference = config["reference_nfe"]
    if not nfe or nfe != sorted(set(nfe)) or any(type(n) is not int or n < 2 or n % 2 for n in nfe + [reference]):
        raise ValueError("NFE must be positive even integers; nfe must be sorted and unique")
    if reference <= max(nfe):
        raise ValueError("reference_nfe must exceed the sweep budgets")
    selection, evaluation = config["selection_seeds"], config["evaluation_seeds"]
    if not selection or not evaluation or set(selection) & set(evaluation):
        raise ValueError("Need nonempty, disjoint selection/evaluation seeds")
    if len(set(selection + evaluation)) != len(selection + evaluation):
        raise ValueError("Sampling seeds must be unique")
    if any(type(seed) is not int or seed < 0 for seed in selection + evaluation):
        raise ValueError("Sampling seeds must be nonnegative integers")
    if not math.isfinite(config["max_seconds_per_file"]) or config["max_seconds_per_file"] < 2:
        raise ValueError("max_seconds_per_file must be at least two seconds")
    return {
        "schema": 1, "manifest_sha256": file_hash(config["manifest"]), "split": config["split"],
        "nfe": nfe, "reference_nfe": reference, "selection_seeds": selection,
        "evaluation_seeds": evaluation, "max_seconds_per_file": config["max_seconds_per_file"],
        "codebooks": list(CODEBOOKS), "sample_rate": SAMPLE_RATE, "block_samples": BLOCK_SAMPLES,
        "solver": "midpoint", "dtype": "float32", "audio_files_per_domain": config["audio_files_per_domain"],
    }


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/diagnostic.json")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("shard must be in [0, num-shards)")
    config = load_config(args.config)
    settings = sweep_settings(config)
    rows = diagnostic_rows(config)
    codec = FlowCodec(config["checkpoints"], config.get("flow_checkpoint")).to(args.device).requires_grad_(False)
    metadata = {"settings": settings, "codec": codec.identity}
    directory = Path(config["cache"])
    directory.mkdir(parents=True, exist_ok=True)
    nfe_values = config["nfe"] + [config["reference_nfe"]]
    seeds = config["selection_seeds"] + config["evaluation_seeds"]
    audio_indices, counts = set(), {domain: 0 for domain in ("speech", "music", "sound")}
    for index, row in enumerate(rows):
        if counts[row["domain"]] < config["audio_files_per_domain"]:
            audio_indices.add(index)
        counts[row["domain"]] += 1
    warmed_up = False
    with wandb.init(project=config["wandb_project"], name=f"{config['name']}-shard{args.shard}",
                    group=config["name"], job_type="diagnostic-sweep", config=config) as run:
        for index in tqdm(range(args.shard, len(rows), args.num_shards), desc="recordings"):
            row = rows[index]
            source = Path(row["path"])
            source_info = {"sha256": file_hash(source), "bytes": source.stat().st_size}
            destination = directory / f"record_{index:05d}.pt"
            if destination.exists():
                previous = torch.load(destination, map_location="cpu", weights_only=False)
                if previous["metadata"] != metadata or previous["source"] != source_info or previous["record"] != row:
                    raise ValueError(f"Stale cache: {destination}. Use a fresh cache directory")
                continue
            audio = read_audio(source)
            start = round(row["start_seconds"] * SAMPLE_RATE)
            audio = audio[..., start:start + int(config["max_seconds_per_file"] * SAMPLE_RATE)]
            blocks = audio.shape[-1] // BLOCK_SAMPLES
            result = {"metadata": metadata, "source": source_info, "record": row, "index": index,
                      "blocks": blocks, "dropped_tail_samples": audio.shape[-1] % BLOCK_SAMPLES}
            if blocks == 0:
                result["status"] = "no_full_block"
            else:
                # [blocks, rates, NFE values including reference, sampling seeds]
                shape = (blocks, len(CODEBOOKS), len(nfe_values), len(seeds))
                result.update({key: torch.empty(shape) for key in ("loss", "sisdr", "nmse_reference", "seconds")})
                result["coarse_loss"] = torch.empty(blocks, len(CODEBOOKS))
                result["coarse_sisdr"] = torch.empty(blocks, len(CODEBOOKS))
                result["coarse_seconds"] = torch.empty(blocks, len(CODEBOOKS))
                result["encode_seconds"] = torch.empty(blocks)
                saved_audio = {}
                with torch.inference_mode():
                    for block in range(blocks):
                        original = audio[..., block * BLOCK_SAMPLES:(block + 1) * BLOCK_SAMPLES][None].to(args.device)
                        if not warmed_up:
                            _, warm_codes = codec.encode(original)
                            warm_coarse = codec.coarse(warm_codes, CODEBOOKS[0])[0]
                            codec.enhance(warm_coarse, seed=0, steps=3)
                            warmed_up = True
                        synchronize(args.device)
                        started = time.perf_counter()
                        _, codes = codec.encode(original)
                        synchronize(args.device)
                        result["encode_seconds"][block] = time.perf_counter() - started
                        if codes.shape[-1] != 150:
                            raise ValueError("Expected 150 RVQ frames per two-second block")
                        energy = original.square().mean().clamp_min(1e-8)
                        for rate, depth in enumerate(CODEBOOKS):
                            synchronize(args.device)
                            started = time.perf_counter()
                            coarse = codec.coarse(codes, depth)[0]
                            synchronize(args.device)
                            result["coarse_seconds"][block, rate] = time.perf_counter() - started
                            result["coarse_loss"][block, rate] = spectral_loss(original, coarse).item()
                            result["coarse_sisdr"][block, rate] = si_sdr(original, coarse).item()
                            if index in audio_indices:
                                saved_audio.setdefault(f"ndac_k{depth}", []).append(coarse[0, 0].cpu())
                            for sample, seed in enumerate(seeds):
                                file_seed = block_seed(row["path"], 0, seed)
                                noise_seed = block_seed("", start + block * BLOCK_SAMPLES, file_seed)
                                synchronize(args.device)
                                started = time.perf_counter()
                                reference = codec.enhance(coarse, noise_seed, config["reference_nfe"] // 2)
                                synchronize(args.device)
                                reference_seconds = time.perf_counter() - started
                                for point, nfe in enumerate(nfe_values):
                                    if nfe == config["reference_nfe"]:
                                        decoded, seconds = reference, reference_seconds
                                    else:
                                        synchronize(args.device)
                                        started = time.perf_counter()
                                        decoded = codec.enhance(coarse, noise_seed, nfe // 2)
                                        synchronize(args.device)
                                        seconds = time.perf_counter() - started
                                    location = (block, rate, point, sample)
                                    result["loss"][location] = spectral_loss(original, decoded).item()
                                    result["sisdr"][location] = si_sdr(original, decoded).item()
                                    result["nmse_reference"][location] = ((decoded - reference).square().mean() / energy).item()
                                    result["seconds"][location] = seconds
                                    if index in audio_indices and seed == config["evaluation_seeds"][0]:
                                        saved_audio.setdefault(f"flow_k{depth}_nfe{nfe}", []).append(decoded[0, 0].cpu())
                if any(not torch.isfinite(result[key]).all() for key in
                       ("loss", "sisdr", "nmse_reference", "seconds", "coarse_loss", "coarse_sisdr")):
                    raise FloatingPointError(f"Non-finite measurements: {row['path']}")
                if index in audio_indices:
                    audio_dir = directory / "audio" / f"record_{index:05d}"
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(audio_dir / "original.wav", audio[0, :blocks * BLOCK_SAMPLES].numpy(), SAMPLE_RATE, subtype="FLOAT")
                    for name, waveforms in saved_audio.items():
                        sf.write(audio_dir / f"{name}.wav", torch.cat(waveforms).numpy(), SAMPLE_RATE, subtype="FLOAT")
                    (audio_dir / "source.json").write_text(json.dumps(row, indent=2) + "\n")
                result["status"] = "complete"
                run.log({"record_index": index, "blocks": blocks,
                         "loss/nfe6_or_first": result["loss"][:, :, config["nfe"].index(6) if 6 in config["nfe"] else 0,
                                                             len(config["selection_seeds"]):].mean().item(),
                         "loss/reference": result["loss"][:, :, -1, len(config["selection_seeds"]):].mean().item()})
            temporary = destination.with_suffix(".pt.part")
            torch.save(result, temporary)
            temporary.replace(destination)
            print(f"Saved {destination.name}: {blocks} blocks", flush=True)


if __name__ == "__main__":
    main()
