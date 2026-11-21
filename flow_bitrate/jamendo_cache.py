"""Plan, build and read the fixed-rate Jamendo training crop cache."""

import argparse
import fcntl
import json
import math
import os
import signal
import time
from collections import defaultdict, deque
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .codec import BLOCK_SAMPLES, ROOT, SCRATCH, FlowCodec, file_hash
from .data import jamendo_track_order, load_config, matched_seed, read_manifest
from .jamendo_ablation import ARMS, CACHE_LOOKAHEAD, CALIBRATION_FILE, FIRST_CROPS, real_crops

DEFAULT_ROOT = SCRATCH / "jamendo-ablation-cache" / "k6-seed0-c192000-v1"
CACHE_VERSION = 1
BLOCK_CROPS = 64
CODEBOOKS = 6


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_plan(root):
    plan = json.loads((root / "plan.json").read_text())
    if (plan.get("version") != CACHE_VERSION or plan.get("codebooks") != CODEBOOKS
            or plan.get("samples") != BLOCK_SAMPLES or plan.get("torch") != str(torch.__version__)
            or min(plan["blocks"], plan["block_crops"], plan["track_count"]) < 1
            or plan["blocks"] * plan["block_crops"] != plan["stored_crops"]):
        raise ValueError(f"Incompatible Jamendo crop plan in {root}")
    return plan


def block_path(root, block):
    return root / "blocks" / f"{block:07d}.npy"


def open_block(path, plan):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (plan["block_crops"], 2, BLOCK_SAMPLES):
        raise ValueError(f"Unexpected cache block format: {path}")
    return array


def check_source(row):
    stat = Path(row["path"]).stat()
    if (not row.get("file_size") or not row.get("mtime_ns")
            or stat.st_size != int(row["file_size"]) or stat.st_mtime_ns != int(row["mtime_ns"])):
        raise ValueError(f"Source does not match its full decode audit: {row['path']}")


def audit(config):
    manifest = Path(config["manifest"])
    manifest_hash = file_hash(manifest)
    report = json.loads(manifest.with_suffix(".summary.json").read_text())
    if (report.get("manifest_sha256") != manifest_hash or report.get("audit") != "full_decode_and_seek_v1"
            or not report.get("loader_check") or report["config"].get("evaluation_seconds") != 10):
        raise ValueError("Run the Jamendo data audit before planning this cache")
    return manifest_hash


def create_plan(root, config, crops, seed, extend):
    manifest = audit(config)
    rows = read_manifest(config["manifest"], dataset="jamendo")
    track_ids = [row["source_id"] for row in rows]
    if "" in track_ids or len(track_ids) != len(set(track_ids)):
        raise ValueError("The audited Jamendo manifest needs one row per track")
    training = [row for row in rows if row["split"] == "train"]
    batches = sorted({arm.batch for arm in ARMS.values()})
    if not training or any(crops % batch for batch in batches) or crops <= 0:
        raise ValueError("The crop budget must be positive and divisible by every effective batch")
    counts = {str(batch): real_crops(len(training), batch, crops // batch) for batch in batches}
    blocks = math.ceil((max(counts.values()) + CACHE_LOOKAHEAD) / BLOCK_CROPS)
    codec = FlowCodec(config["checkpoints"], load_backbone=False, configured_noise=True)
    value = dict(version=CACHE_VERSION, codebooks=CODEBOOKS, samples=BLOCK_SAMPLES, seed=seed,
                 torch=str(torch.__version__), manifest_sha256=manifest, codec=codec.identity,
                 track_count=len(training), budget_crops=crops, real_crops=counts,
                 blocks=blocks, block_crops=BLOCK_CROPS, stored_crops=blocks * BLOCK_CROPS)
    (root / "blocks").mkdir(parents=True, exist_ok=True)
    with (root / ".plan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "plan.json"
        if path.exists():
            previous = read_plan(root)
            budget_keys = {"budget_crops", "real_crops", "blocks", "stored_crops"}
            if any(previous[key] != value[key] for key in value.keys() - budget_keys):
                raise ValueError(f"Existing plan has different inputs. Use a different --root: {root}")
            if previous != value:
                if not extend or crops <= previous["budget_crops"]:
                    raise ValueError("Use plan --extend with a larger crop budget to append to this cache")
                (root / "ready.json").unlink(missing_ok=True)
                write_json(path, value)
        else:
            write_json(path, value)
    gib = value["stored_crops"] * 2 * BLOCK_SAMPLES * 4 / 2 ** 30
    print(f"{len(training):,} train tracks, real crops by batch {counts}")
    print(f"{value['stored_crops']:,} paired crops in {blocks:,} blocks, {gib:.1f} GiB at {root}")


def verify_identity(root, plan, *, manifest, codec, seed, track_count):
    actual = (manifest, codec, seed, track_count)
    expected = (plan["manifest_sha256"], plan["codec"], plan["seed"], plan["track_count"])
    if actual != expected:
        raise ValueError(f"Cache plan and training inputs disagree: {root}")


class CachedJamendoCrops(Dataset):
    """Read the same clean crop, coarse crop and training seed for a sampler key."""

    def __init__(self, dataset, root, manifest, codec, batch, updates):
        self.rows = dataset.rows
        self.seed = dataset.seed
        self.training = True
        self.root = root
        self.plan = read_plan(root)
        verify_identity(root, self.plan, manifest=manifest, codec=codec, seed=self.seed,
                        track_count=len(self.rows))
        if self.plan["stored_crops"] < real_crops(len(self.rows), batch, updates) + CACHE_LOOKAHEAD:
            raise ValueError(f"The cache at {root} is too short for the requested training budget")
        if not (root / "ready.json").is_file():
            raise ValueError(f"Run cache status --require-complete before training: {root}")
        marker = json.loads((root / "ready.json").read_text())
        if marker.get("plan_sha256") != file_hash(root / "plan.json"):
            raise ValueError(f"Run cache status --require-complete before training: {root}")
        self.epoch = None
        self.block = None
        self.array = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        index, epoch = key
        if index < 0:
            zeros = torch.zeros(1, BLOCK_SAMPLES)
            return zeros, zeros, -1
        if epoch != self.epoch:
            order = jamendo_track_order(len(self), self.seed, epoch)
            self.inverse = order.argsort().numpy()
            self.epoch = epoch
        position = epoch * len(self) + int(self.inverse[index])
        block, offset = divmod(position, self.plan["block_crops"])
        if block != self.block:
            self.array = open_block(block_path(self.root, block), self.plan)
            self.block = block
        pair = torch.from_numpy(self.array[offset].copy())
        seed = matched_seed("training", self.seed, epoch, self.rows[index]["source_id"])
        return pair[:1], pair[1:], seed


def valid_block(path, plan):
    try:
        open_block(path, plan)
    except (OSError, ValueError, EOFError):
        return False
    return True


def status(root, plan, require_complete):
    with (root / ".plan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if read_plan(root) != plan:
            raise RuntimeError("Cache plan changed; rerun status for its current budget")
        missing = [block for block in range(plan["blocks"]) if not valid_block(block_path(root, block), plan)]
        if not missing:
            write_json(root / "ready.json", {"plan_sha256": file_hash(root / "plan.json")})
        else:
            (root / "ready.json").unlink(missing_ok=True)
    total = plan["blocks"]
    print(f"{total - len(missing):,}/{total:,} blocks complete "
          f"({(total - len(missing)) * plan['block_crops']:,}/{plan['stored_crops']:,} crop pairs)")
    if not missing:
        return True
    if require_complete:
        raise RuntimeError(f"{len(missing):,} blocks pending; first missing block IDs: {missing[:12]}")
    print(f"First missing block IDs: {missing[:12]}")
    return False


def training_dataset(config, plan):
    from .train_postfilter import JamendoCrops

    if file_hash(config["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("Manifest changed after planning")
    rows = read_manifest(config["manifest"], dataset="jamendo", check_files=False)
    return JamendoCrops(rows, "train", plan["seed"])


def build(root, config, plan, shard, shards, batch):
    if not 0 <= shard < shards or batch < 1:
        raise ValueError("Use 0 <= shard < shards and a positive codec batch")
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    end = float(os.environ.get("SLURM_JOB_END_TIME", "inf")) - 30

    def stop_requested():
        return stopping or time.time() >= end

    ids = range(shard, plan["blocks"], shards)
    pending = [block for block in ids if not valid_block(block_path(root, block), plan)]
    if not pending:
        print(f"Shard {shard}/{shards}: all {len(ids)} assigned blocks already exist")
        return 0
    dataset = training_dataset(config, plan)
    codec = FlowCodec(config["checkpoints"], load_backbone=False, configured_noise=True)
    verify_identity(root, plan, manifest=file_hash(config["manifest"]), codec=codec.identity,
                    seed=dataset.seed, track_count=len(dataset))
    block_crops = plan["block_crops"]
    order_by_epoch = {}
    built = 0
    started = time.monotonic()
    with torch.inference_mode(), tqdm(total=len(ids), initial=len(ids) - len(pending),
                                      desc=f"Shard {shard}/{shards}", unit="block", mininterval=15) as progress:
        for block in pending:
            if stop_requested():
                break
            path = block_path(root, block)
            with path.with_suffix(".lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if not valid_block(path, plan):
                    pairs = np.empty((block_crops, 2, BLOCK_SAMPLES), dtype=np.float32)
                    start = block * block_crops
                    for offset in range(0, block_crops, batch):
                        if stop_requested():
                            break
                        crops = []
                        for within in range(offset, min(offset + batch, block_crops)):
                            epoch, position = divmod(start + within, len(dataset))
                            if epoch not in order_by_epoch:
                                order_by_epoch[epoch] = jamendo_track_order(len(dataset), dataset.seed, epoch).tolist()
                            index = order_by_epoch[epoch][position]
                            check_source(dataset.rows[index])
                            crops.append(dataset[index, epoch][0])
                        clean = torch.stack(crops)
                        _, codes = codec.encode(clean)
                        coarse, _ = codec.coarse(codes, CODEBOOKS)
                        if coarse.shape != clean.shape or not torch.isfinite(coarse).all():
                            raise ValueError(f"Invalid NDAC output in block {block} at offset {offset}")
                        pairs[offset:offset + len(crops)] = torch.cat((clean, coarse), dim=1).numpy()
                    else:
                        staging = path.with_suffix(".part")
                        with staging.open("wb") as handle:
                            np.save(handle, pairs, allow_pickle=False)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(staging, path)
                        built += 1
                    if not valid_block(path, plan):
                        break
                progress.update()
    remaining = sum(not valid_block(block_path(root, block), plan) for block in ids)
    elapsed = time.monotonic() - started
    rate = built * block_crops / max(elapsed, 0.001)
    print(f"Shard {shard}/{shards}: wrote {built} blocks at {rate:.2f} crop pairs/s; "
          f"{remaining} assigned blocks still pending")
    return 75 if remaining else 0


def calibration_positions(dataset, plan, limit=4096):
    # Choose a training excerpt per shuffled artist before using another track from that artist.
    order = jamendo_track_order(len(dataset), dataset.seed, 0).tolist()[:plan["stored_crops"]]
    by_group = defaultdict(deque)
    for position, index in enumerate(order):
        by_group[dataset.rows[index]["group"]].append(position)
    groups = deque(by_group.values())
    selected = []
    while groups and len(selected) < limit:
        positions = groups.popleft()
        selected.append(positions.popleft())
        if positions:
            groups.append(positions)
    return sorted(selected)


def calibrate(root, config, plan):
    from scipy.ndimage import gaussian_filter1d

    status(root, plan, require_complete=True)
    dataset = training_dataset(config, plan)
    codec = FlowCodec(config["checkpoints"], load_backbone=False, configured_noise=True)
    verify_identity(root, plan, manifest=file_hash(config["manifest"]), codec=codec.identity,
                    seed=dataset.seed, track_count=len(dataset))
    selected = calibration_positions(dataset, plan)
    rows = []
    latest_block = -1
    with torch.inference_mode():
        for position in tqdm(selected, desc="Calibrating residual sigma", unit="crop", mininterval=15):
            block, offset = divmod(position, plan["block_crops"])
            if block != latest_block:
                pairs = open_block(block_path(root, block), plan)
                latest_block = block
            sample = torch.from_numpy(pairs[offset].copy())
            condition, target, (_, frames, _) = codec.preprocess(sample[1:].unsqueeze(0), sample[:1].unsqueeze(0))
            rms = (target[..., :frames] - condition[..., :frames]).abs().square().mean(-1).sqrt()
            rows.append(rms.reshape(-1).numpy())
    raw = gaussian_filter1d(np.quantile(np.stack(rows), 0.997, axis=0) / 3, 3, mode="nearest")
    rms = np.sqrt(np.mean(raw ** 2))
    if not np.isfinite(raw).all() or rms <= 0 or (raw <= 0).any():
        raise ValueError("Invalid calibrated noise curve")
    matched = raw * (float(codec.sigma.square().mean().sqrt()) / rms)
    write_json(root / CALIBRATION_FILE, {
        "version": 1, "method": "jamendo-training-residual-rms-q997-div3-gaussian3-v1",
        "codec": codec.identity, "manifest_sha256": plan["manifest_sha256"], "seed": plan["seed"],
        "codebooks": CODEBOOKS, "crops": len(selected), "raw": raw.tolist(), "matched": matched.tolist(),
    })
    print(f"Calibrated from {len(selected):,} training crops. Wrote {root / CALIBRATION_FILE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs/jamendo.json", type=Path)
    parser.add_argument("--root", default=DEFAULT_ROOT, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--crops", default=FIRST_CROPS, type=int)
    plan_parser.add_argument("--seed", default=0, type=int)
    plan_parser.add_argument("--extend", action="store_true")
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    build_parser.add_argument("--shards", type=int, required=True)
    build_parser.add_argument("--codec-batch", type=int, default=2)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--require-complete", action="store_true")
    commands.add_parser("calibrate")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    config = load_config(args.config)
    if config.get("dataset") != "jamendo" or config.get("codebooks") != CODEBOOKS:
        parser.error("This cache is only for Jamendo at six codebooks (4.5 kbps)")
    if args.command == "plan":
        create_plan(args.root, config, args.crops, args.seed, args.extend)
        return
    plan = read_plan(args.root)
    if args.command == "build":
        return build(args.root, config, plan, args.shard, args.shards, args.codec_batch)
    if args.command == "status":
        status(args.root, plan, args.require_complete)
    if args.command == "calibrate":
        calibrate(args.root, config, plan)


if __name__ == "__main__":
    raise SystemExit(main())
