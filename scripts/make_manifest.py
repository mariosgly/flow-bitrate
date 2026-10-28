"""Pick a small pilot from the shared LibriTTS-R, MedleyDB and FSD50K folders."""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/scratch/mg8253/flow-bitrate")


def scan(root, domain):
    if not root.is_dir():
        raise FileNotFoundError(f"Cannot access {root}; check the path on Torch")
    audio, metadata, visited = set(), [], set()

    def walk_error(error):
        raise error

    # Shared collections can contain directory symlinks; avoid following cycles.
    for directory, folders, names in os.walk(root, followlinks=True, onerror=walk_error):
        stat = Path(directory).stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in visited:
            folders[:] = []
            continue
        visited.add(identity)
        folders.sort()
        for name in sorted(names):
            path = Path(directory) / name
            if name in {"dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"}:
                metadata.append(path)
            if path.suffix.lower() not in {".wav", ".flac"}:
                continue
            if domain == "music" and not path.stem.lower().endswith("_mix"):
                continue  # MedleyDB full mixes only, never stems or raw microphones.
            audio.add(path.resolve())
    print(f"{domain}: found {len(audio)} audio candidates in {root}", flush=True)
    return sorted(audio), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnostic.json")
    parser.add_argument("--speech", type=Path, default=Path(
        "/scratch/mk9649/shared/dcase2026_task5/dataset_copy/libritts/LibriTTS_R/train-clean-100"))
    parser.add_argument("--music", type=Path, default=Path(
        "/projects/work/marl/datasets/mir_datasets/medleydb_melody"))
    parser.add_argument("--sounds", type=Path, default=Path(
        "/projects/work/marl/datasets/sound_datasets/fsd50k"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = (SCRATCH / config["manifest"]).resolve()
    if not output.is_relative_to(SCRATCH.resolve()):
        raise ValueError(f"Manifest must stay under {SCRATCH}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; use --overwrite to replace it")
    seconds = 2 * config["min_oracle_blocks"]
    if seconds <= 0 or config["max_seconds_per_file"] < seconds:
        raise ValueError("max_seconds_per_file must cover min_oracle_blocks two-second blocks")
    rng = random.Random(config["seed"])
    rows, seen_paths = [], set()
    for domain, root in (("speech", args.speech), ("music", args.music), ("sound", args.sounds)):
        candidates, metadata = scan(root, domain)
        if not candidates:
            hint = " (expected *_MIX.wav full mixes)" if domain == "music" else ""
            raise RuntimeError(f"No extracted WAV/FLAC files in {root}{hint}; archives are not unpacked")
        uploaders = {}
        if domain == "sound":
            for path in metadata:
                for clip, info in json.loads(path.read_text()).items():
                    uploader = info.get("username") or info.get("uploader")
                    if uploader:
                        uploaders[Path(str(clip)).stem] = str(uploader)
            if not uploaders:
                raise RuntimeError(
                    f"Need extracted dev/eval_clips_info_FSD50K.json under {root} "
                    "to select different uploaders; no manifest was written")
        rng.shuffle(candidates)
        groups, selected = set(), []
        skipped = {"short": 0, "unreadable": 0, "empty_or_nonfinite": 0, "unknown_group": 0}
        for path in candidates:
            if path in seen_paths:
                continue
            if domain == "speech":
                speaker = path.stem.split("_", 1)[0]
                group = f"speech:{speaker}" if speaker.isdigit() else None
            elif domain == "music":
                group = f"music:{path.stem[:-4].split('_', 1)[0]}"  # Artist_Track_MIX
            else:
                uploader = uploaders.get(path.stem)
                group = f"sound:{uploader}" if uploader else None
            if group is None:
                skipped["unknown_group"] += 1
                continue
            if group in groups:
                continue
            try:
                with sf.SoundFile(path) as source:
                    duration = source.frames / source.samplerate
                    if duration < seconds + 0.02:
                        skipped["short"] += 1
                        continue
                    # Integer-second offset near the middle; a small duration margin
                    # keeps the oracle eligible after sample-rate conversion.
                    start = math.floor((duration - seconds) / 2)
                    source.seek(start * source.samplerate)
                    audio = source.read(seconds * source.samplerate, dtype="float32", always_2d=True)
                    if len(audio) != seconds * source.samplerate:
                        skipped["unreadable"] += 1
                        continue
                mono = audio.mean(axis=1)
                if not np.isfinite(audio).all() or np.max(np.abs(mono)) <= 1e-5:
                    skipped["empty_or_nonfinite"] += 1
                    continue
            except (OSError, RuntimeError, ValueError):
                skipped["unreadable"] += 1
                continue
            selected.append(dict(path=str(path), split=config["split"], group=group,
                                 domain=domain, start_seconds=start))
            seen_paths.add(path)
            groups.add(group)
            print(f"  {len(selected):2d}/10  {group}  offset={start}s  {path}", flush=True)
            if len(selected) == 10:
                break
        print(f"{domain}: selected {len(selected)}/10; skipped {skipped}", flush=True)
        if len(selected) != 10:
            raise RuntimeError(f"Not enough eligible {domain} sources in {root}; no manifest was written")
        rows.extend(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "split", "group", "domain", "start_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    print(f"Wrote {len(rows)} recordings to {output}. No audio was copied.")


if __name__ == "__main__":
    main()
