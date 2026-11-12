"""A manifest in, small feature shards out. Splits stay at recording/group level."""

import csv
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

from .codec import BLOCK_SAMPLES, SAMPLE_RATE, SCRATCH, file_hash


def load_config(path):
    config = json.loads(Path(path).read_text())
    config.setdefault("wandb_dir", str(SCRATCH / "wandb"))
    for key in ("manifest", "checkpoints", "cache", "output", "flow_checkpoint", "wandb_dir"):
        if config.get(key):
            resolved = (SCRATCH / config[key]).resolve()
            if key in {"checkpoints", "cache", "output", "wandb_dir"} and not resolved.is_relative_to(SCRATCH.resolve()):
                raise ValueError(f"{key} must be under {SCRATCH}")
            config[key] = str(resolved)
    Path(config["wandb_dir"]).mkdir(parents=True, exist_ok=True)
    return config


def matched_seed(*parts):
    digest = hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (1 << 63)


def read_manifest(path, dataset=None):
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"path", "split", "group"} <= rows[0].keys():
        raise ValueError("Manifest needs path,split,group columns and at least one recording")
    if dataset is not None:
        rows = [row for row in rows if row.get("dataset") == dataset]
        if not rows:
            raise ValueError(f"No {dataset} tracks in {path}")
    groups, seen = {}, set()
    for row in rows:
        row["path"] = str((path.parent / row["path"]).resolve())
        if row["split"] not in {"train", "val", "test"} or not row["group"]:
            raise ValueError(f"Bad split/group: {row}")
        if row["path"] in seen:
            raise ValueError(f"Repeated recording: {row['path']}")
        seen.add(row["path"])
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError(f"Group leaks across splits: {row['group']}")
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    return rows


def read_audio(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio).mean(1)[None]  # [1, samples], mono
    if rate != SAMPLE_RATE:
        audio = AF.resample(audio, rate, SAMPLE_RATE)
    if not torch.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {path}")
    return audio


def load_cache(directory):
    files = sorted(Path(directory).glob("shard_*.pt"))
    if not files:
        raise FileNotFoundError(f"No cache shards in {directory}; run flow_bitrate.prepare")
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    metadata = shards[0]["metadata"]
    expected = metadata["num_shards"]
    if len(files) != expected or {s["shard"] for s in shards} != set(range(expected)):
        raise ValueError("Cache is incomplete; finish all preparation shards")
    if any(s["metadata"] != metadata for s in shards):
        raise ValueError("Mixed cache settings/checkpoints. Use a fresh cache directory")
    nonempty = [s for s in shards if s["records"]]
    if not nonempty:
        raise ValueError("No full two-second blocks in the dataset")
    cache = {key: torch.cat([s[key] for s in nonempty]) for key in ("base", "flow", "loss", "sisdr")}
    cache["records"] = [r for s in nonempty for r in s["records"]]
    cache["metadata"] = metadata
    for key in ("base", "flow", "loss", "sisdr"):
        if not torch.isfinite(cache[key]).all():
            raise ValueError(f"Non-finite cached {key}")
    return cache


class FeatureDataset(Dataset):
    def __init__(self, cache, split):
        indices = [i for i, row in enumerate(cache["records"]) if row["split"] == split]
        if not indices:
            raise ValueError(f"No {split} blocks")
        self.x = torch.cat((cache["base"], cache["flow"]), 1)[indices]
        self.y = cache["loss"][indices].log1p()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class AudioDataset(Dataset):
    def __init__(self, cache, split):
        self.records = [r for r in cache["records"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"No {split} blocks")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        info = sf.info(row["path"])
        if info.samplerate == SAMPLE_RATE:
            audio, _ = sf.read(row["path"], start=row["start"], frames=BLOCK_SAMPLES,
                               dtype="float32", always_2d=True)
            audio = torch.from_numpy(audio).mean(1)[None]
        else:
            audio = read_audio(row["path"])[..., row["start"]:row["start"] + BLOCK_SAMPLES]
        if audio.shape != (1, BLOCK_SAMPLES) or not torch.isfinite(audio).all():
            raise ValueError(f"Audio changed since preparation: {row['path']}")
        return audio


def check_manifest(cache, path):
    if file_hash(path) != cache["metadata"]["manifest_sha256"]:
        raise ValueError("Manifest changed since preparation. Make a new cache")
    for source in cache["metadata"]["source_stats"]:
        stat = Path(source["path"]).stat()
        if stat.st_size != source["size"] or stat.st_mtime_ns != source["mtime"]:
            raise ValueError(f"Audio changed since preparation: {source['path']}")
