"""Exercise the actual CPU dataloader for each source/rate/channel/format profile."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .codec import SAMPLE_RATE
from .data import read_manifest
from .train_postfilter import AudioCrops


def check_loaders(manifest, evaluation_seconds=10, workers=4):
    rows = read_manifest(manifest)
    profiles = {}
    for row in rows:
        key = (row["split"], row.get("dataset", "audio"), row.get("sample_rate"),
               row.get("channels"), Path(row["path"]).suffix.lower())
        profiles.setdefault(key, row)
    for split in ("train", "val", "test"):
        seconds = evaluation_seconds if split == "test" else 2
        datasets = [AudioCrops([row], split, samples=1, per_domain=1, seconds=seconds)
                    for key, row in profiles.items() if key[0] == split]
        if not datasets:
            raise ValueError(f"No {split} profiles")
        loader = DataLoader(ConcatDataset(datasets), batch_size=1, num_workers=workers)
        for audio in loader:
            if audio.shape != (1, 1, seconds * SAMPLE_RATE) or not torch.isfinite(audio).all():
                raise ValueError(f"Invalid {split} dataloader output: {audio.shape}")
        print(f"Loader passed: {split}, {len(datasets)} source/rate/channel/format profiles, {seconds}s", flush=True)
    return [list(key) for key in profiles]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluation-seconds", type=int, default=10)
    args = parser.parse_args()
    check_loaders(args.manifest, args.evaluation_seconds, args.workers)
