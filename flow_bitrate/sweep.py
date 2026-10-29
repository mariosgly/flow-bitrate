"""Frozen FlowDec, four RVQ depths, several midpoint budgets. No training or probes."""

import argparse
import fcntl
import json
import math
import os
import signal
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


def commit_file(temporary, destination):
    # Finish the write before publishing it. A kill leaves the old file or a .part.
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_progress(result, destination):
    temporary = destination.with_suffix(destination.suffix + ".part")
    torch.save(result, temporary)
    commit_file(temporary, destination)


def save_audio(path, waveform):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".wav.part")
    sf.write(temporary, waveform.numpy(), SAMPLE_RATE, subtype="FLOAT", format="WAV")
    commit_file(temporary, path)


def measure_record(codec, audio, row, index, source_info, metadata, config, directory,
                   device, listening, warmed_up, should_stop):
    destination = directory / f"record_{index:05d}.pt"
    progress = directory / f"record_{index:05d}.progress.pt"
    start = round(row["start_seconds"] * SAMPLE_RATE)
    audio = audio[..., start:start + int(config["max_seconds_per_file"] * SAMPLE_RATE)]
    blocks = audio.shape[-1] // BLOCK_SAMPLES
    nfe_values = config["nfe"] + [config["reference_nfe"]]
    seeds = config["selection_seeds"] + config["evaluation_seeds"]
    if progress.exists():
        result = torch.load(progress, map_location="cpu", weights_only=False)
        if (result["metadata"] != metadata or result["source"] != source_info
                or result["record"] != row or result["blocks"] != blocks):
            raise ValueError(f"Stale progress: {progress}. Use a fresh cache directory")
        print(f"Resuming {destination.name}: {result['done'].sum().item()}/{result['done'].numel()} solves saved",
              flush=True)
    else:
        result = {"metadata": metadata, "source": source_info, "record": row, "index": index,
                  "blocks": blocks, "dropped_tail_samples": audio.shape[-1] % BLOCK_SAMPLES,
                  "status": "in_progress"}
        # [blocks, rates, NFE values including reference, sampling seeds]
        shape = (blocks, len(CODEBOOKS), len(nfe_values), len(seeds))
        result.update({key: torch.full(shape, float("nan"))
                       for key in ("loss", "sisdr", "nmse_reference", "seconds")})
        result["done"] = torch.zeros(shape, dtype=torch.bool)
        for key in ("coarse_loss", "coarse_sisdr", "coarse_seconds"):
            result[key] = torch.full((blocks, len(CODEBOOKS)), float("nan"))
        result["encode_seconds"] = torch.full((blocks,), float("nan"))
    audio_dir = directory / "audio" / f"record_{index:05d}"
    with torch.inference_mode():
        for block in range(blocks):
            if result["done"][block].all():
                continue
            if should_stop():
                return False, warmed_up
            original = audio[..., block * BLOCK_SAMPLES:(block + 1) * BLOCK_SAMPLES][None].to(device)
            if not warmed_up:
                _, warm_codes = codec.encode(original)
                warm_coarse = codec.coarse(warm_codes, CODEBOOKS[0])[0]
                codec.enhance(warm_coarse, seed=0, steps=3)
                warmed_up = True
            synchronize(device)
            started = time.perf_counter()
            _, codes = codec.encode(original)
            synchronize(device)
            if not torch.isfinite(result["encode_seconds"][block]):
                result["encode_seconds"][block] = time.perf_counter() - started
            if codes.shape[-1] != 150:
                raise ValueError("Expected 150 RVQ frames per two-second block")
            energy = original.square().mean().clamp_min(1e-8)
            for rate, depth in enumerate(CODEBOOKS):
                if result["done"][block, rate].all():
                    continue
                if should_stop():
                    return False, warmed_up
                synchronize(device)
                started = time.perf_counter()
                coarse = codec.coarse(codes, depth)[0]
                synchronize(device)
                if not torch.isfinite(result["coarse_seconds"][block, rate]):
                    result["coarse_seconds"][block, rate] = time.perf_counter() - started
                    result["coarse_loss"][block, rate] = spectral_loss(original, coarse).item()
                    result["coarse_sisdr"][block, rate] = si_sdr(original, coarse).item()
                    if listening:
                        save_audio(audio_dir / "blocks" / f"ndac_k{depth}_block{block:03d}.wav", coarse[0, 0].cpu())
                    save_progress(result, progress)
                for sample, seed in enumerate(seeds):
                    if result["done"][block, rate, :, sample].all():
                        continue
                    if should_stop():
                        return False, warmed_up
                    reference_key = (block, rate, sample)
                    if result["done"][block, rate, -1, sample]:
                        cached = result.get("reference")
                        if cached is None or cached["key"] != reference_key:
                            raise ValueError(f"Missing reference in {progress}; keep the progress file for diagnosis")
                        reference = cached["waveform"].to(device)
                        reference_seconds = cached["seconds"]
                    else:
                        file_seed = block_seed(row["path"], 0, seed)
                        noise_seed = block_seed("", start + block * BLOCK_SAMPLES, file_seed)
                        synchronize(device)
                        started = time.perf_counter()
                        reference = codec.enhance(coarse, noise_seed, config["reference_nfe"] // 2)
                        synchronize(device)
                        reference_seconds = time.perf_counter() - started
                        # Keep only the active reference, not every generated waveform.
                        result["reference"] = {"key": reference_key, "waveform": reference.cpu(),
                                               "seconds": reference_seconds}
                    file_seed = block_seed(row["path"], 0, seed)
                    noise_seed = block_seed("", start + block * BLOCK_SAMPLES, file_seed)
                    # Commit the reference first so resuming does not rerun its expensive solve.
                    for point in [len(nfe_values) - 1] + list(range(len(nfe_values) - 1)):
                        location = (block, rate, point, sample)
                        if result["done"][location]:
                            continue
                        nfe = nfe_values[point]
                        if point == len(nfe_values) - 1:
                            decoded, elapsed = reference, reference_seconds
                        else:
                            if should_stop():
                                return False, warmed_up
                            synchronize(device)
                            started = time.perf_counter()
                            decoded = codec.enhance(coarse, noise_seed, nfe // 2)
                            synchronize(device)
                            elapsed = time.perf_counter() - started
                        result["loss"][location] = spectral_loss(original, decoded).item()
                        result["sisdr"][location] = si_sdr(original, decoded).item()
                        result["nmse_reference"][location] = ((decoded - reference).square().mean() / energy).item()
                        result["seconds"][location] = elapsed
                        if any(not torch.isfinite(result[key][location])
                               for key in ("loss", "sisdr", "nmse_reference", "seconds")):
                            raise FloatingPointError(f"Non-finite measurement: {row['path']} {location}")
                        if listening and seed == config["evaluation_seeds"][0]:
                            save_audio(audio_dir / "blocks" / f"flow_k{depth}_nfe{nfe}_block{block:03d}.wav",
                                       decoded[0, 0].cpu())
                        result["done"][location] = True
                        save_progress(result, progress)
            print(f"record {index}: block {block + 1}/{blocks} saved", flush=True)
    if blocks:
        if not result["done"].all() or any(not torch.isfinite(result[key]).all() for key in
                ("loss", "sisdr", "nmse_reference", "seconds", "coarse_loss", "coarse_sisdr",
                 "coarse_seconds", "encode_seconds")):
            raise FloatingPointError(f"Incomplete/non-finite measurements: {row['path']}")
        if listening:
            save_audio(audio_dir / "original.wav", audio[0, :blocks * BLOCK_SAMPLES])
            names = [f"ndac_k{depth}" for depth in CODEBOOKS]
            names += [f"flow_k{depth}_nfe{nfe}" for depth in CODEBOOKS for nfe in nfe_values]
            for name in names:
                if should_stop():
                    return False, warmed_up
                waveforms = [torch.from_numpy(sf.read(audio_dir / "blocks" / f"{name}_block{block:03d}.wav",
                                                     dtype="float32")[0]) for block in range(blocks)]
                save_audio(audio_dir / f"{name}.wav", torch.cat(waveforms))
            temporary = audio_dir / "source.json.part"
            temporary.write_text(json.dumps(row, indent=2) + "\n")
            commit_file(temporary, audio_dir / "source.json")
    result.pop("reference", None)
    result["status"] = "complete" if blocks else "no_full_block"
    save_progress(result, destination)
    progress.unlink(missing_ok=True)
    return True, warmed_up


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/diagnostic.json")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=float,
                        help="Stop the sweep cleanly near this runtime; exit 75 when work remains")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("shard must be in [0, num-shards)")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error("max-seconds must be positive and finite")
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    def should_stop():
        return stopping or (args.max_seconds is not None and time.monotonic() - started >= args.max_seconds)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    config = load_config(args.config)
    settings = sweep_settings(config)
    rows = diagnostic_rows(config)
    codec = FlowCodec(config["checkpoints"], config.get("flow_checkpoint")).to(args.device).requires_grad_(False)
    metadata = {"settings": settings, "codec": codec.identity}
    directory = Path(config["cache"])
    directory.mkdir(parents=True, exist_ok=True)
    audio_indices, counts = set(), {domain: 0 for domain in ("speech", "music", "sound")}
    for index, row in enumerate(rows):
        if counts[row["domain"]] < config["audio_files_per_domain"]:
            audio_indices.add(index)
        counts[row["domain"]] += 1
    warmed_up = False
    with wandb.init(project=config["wandb_project"], name=f"{config['name']}-shard{args.shard}",
                    group=config["name"], job_type="diagnostic-sweep", dir=config["wandb_dir"], config=config) as run:
        for index in tqdm(range(args.shard, len(rows), args.num_shards), desc="recordings"):
            if should_stop():
                print("Sweep stopped cleanly. Rerun the same job to continue.", flush=True)
                return 75
            # Kernel-held locks disappear on a kill; lock files themselves can remain.
            with (directory / f"record_{index:05d}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(f"Record {index} is being measured by another job") from error
                row = rows[index]
                source = Path(row["path"])
                source_info = {"sha256": file_hash(source), "bytes": source.stat().st_size}
                destination = directory / f"record_{index:05d}.pt"
                if destination.exists():
                    previous = torch.load(destination, map_location="cpu", weights_only=False)
                    if previous["metadata"] != metadata or previous["source"] != source_info or previous["record"] != row:
                        raise ValueError(f"Stale cache: {destination}. Use a fresh cache directory")
                    continue
                complete, warmed_up = measure_record(
                    codec, read_audio(source), row, index, source_info, metadata, config, directory,
                    args.device, index in audio_indices, warmed_up, should_stop)
                if not complete:
                    print("Sweep stopped cleanly. Rerun the same job to continue.", flush=True)
                    return 75
                result = torch.load(destination, map_location="cpu", weights_only=False)
                if result["blocks"]:
                    run.log({"record_index": index, "blocks": result["blocks"],
                             "loss/nfe6_or_first": result["loss"][:, :, config["nfe"].index(6) if 6 in config["nfe"] else 0,
                                                                 len(config["selection_seeds"]):].mean().item(),
                             "loss/reference": result["loss"][:, :, -1, len(config["selection_seeds"]):].mean().item()})
                print(f"Saved {destination.name}: {result['blocks']} blocks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
