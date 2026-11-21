"""Train the copied CyFM U-Net on NDAC reconstructions, with Lightning + W&B."""

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio.functional as AF
from pytorch_lightning.callbacks import Callback, Checkpoint, LearningRateMonitor, ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from .codec import BLOCK_SAMPLES, ROOT, SAMPLE_RATE, SCRATCH, file_hash
from .data import jamendo_track_order, load_config, matched_seed, read_manifest
from .jamendo_ablation import ARMS, CALIBRATION_FILE, FIRST_CROPS, real_crops
from .jamendo_cache import CachedJamendoCrops, write_json
from .postfilter import Postfilter


class AudioCrops(Dataset):
    def __init__(self, rows, split, samples=8192, per_domain=32, seed=0, seconds=2):
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} recordings in the manifest")
        self.training = split == "train"
        self.seconds = seconds
        self.samples, self.seed, self.generator = samples, seed, None
        # v2 content labels describe recordings; sampling still uses the three broad source pools.
        def sampling_domain(row):
            return row.get("sampling_domain") or row.get("domain") or "audio"
        self.domains = sorted({sampling_domain(row) for row in self.rows})
        self.by_domain = [[row for row in self.rows if sampling_domain(row) == domain]
                          for domain in self.domains]
        self.by_source = [[[row for row in candidates if (row.get("sampling_dataset") or row.get("dataset", "audio")) == source]
                           for source in sorted({row.get("sampling_dataset") or row.get("dataset", "audio") for row in candidates})]
                          for candidates in self.by_domain]
        if not self.training:
            # Spread fixed evaluation excerpts across source groups before reusing one.
            selected = []
            for candidates in self.by_domain:
                groups = {}
                for row in candidates:
                    groups.setdefault(row["group"], []).append(row)
                sources = {}
                for group, values in groups.items():
                    sources.setdefault(values[0].get("dataset", "audio"), []).append(group)
                source_groups = [sorted(values, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
                                 for _, values in sorted(sources.items())]
                ordered = [values[offset] for offset in range(max(map(len, source_groups)))
                           for values in source_groups if offset < len(values)]
                for offset in range(max(map(len, groups.values()))):
                    for group in ordered:
                        if offset < len(groups[group]):
                            selected.append(groups[group][offset])
                            if sum(sampling_domain(r) == sampling_domain(candidates[0])
                                   for r in selected) >= per_domain:
                                break
                    else:
                        continue
                    break
            buckets = [[row for row in selected if sampling_domain(row) == domain]
                       for domain in self.domains]
            self.rows = [bucket[offset] for offset in range(max(map(len, buckets)))
                         for bucket in buckets if offset < len(bucket)]

    def __len__(self):
        return self.samples if self.training else len(self.rows)

    def __getitem__(self, index):
        if self.generator is None:
            worker = get_worker_info()
            self.generator = torch.Generator().manual_seed(worker.seed if worker else self.seed)
        if self.training:
            sources = self.by_source[index % len(self.by_source)]
            candidates = sources[torch.randint(len(sources), (), generator=self.generator).item()]
            row = candidates[torch.randint(len(candidates), (), generator=self.generator).item()]
        else:
            row = self.rows[index]
        return self.read_crop(row, self.generator)

    def read_crop(self, row, generator=None):
        if row.get("sample_rate") and row.get("frames"):
            rate, frames = int(row["sample_rate"]), int(row["frames"])
        else:
            info = sf.info(row["path"])
            rate, frames = info.samplerate, info.frames
        needed = math.ceil(self.seconds * rate)
        first = round(float(row.get("start_seconds") or 0) * rate)
        last = frames - needed
        if first < 0 or last < first:
            raise ValueError(f"Need {self.seconds} seconds after start_seconds: {row['path']}")
        if self.training:
            start = torch.randint(first, last + 1, (), generator=generator).item()
        else:
            start = first if row.get("start_seconds") else (first + last) // 2
        value, _ = sf.read(row["path"], start=start, frames=needed, dtype="float32", always_2d=True)
        audio = torch.from_numpy(value).mean(1)[None]
        if rate != SAMPLE_RATE:
            audio = AF.resample(audio, rate, SAMPLE_RATE)
        target_samples = round(self.seconds * SAMPLE_RATE)
        if audio.shape[-1] < target_samples or not torch.isfinite(audio).all():
            raise ValueError(f"Short or non-finite crop: {row['path']}")
        return audio[..., :target_samples]


class JamendoCrops(AudioCrops):
    def __init__(self, rows, split, seed, per_domain=32, seconds=2):
        if split != "train":
            rows = [row for row in rows if row["split"] == split
                    and int(row["frames"]) - round(float(row.get("start_seconds") or 0) * int(row["sample_rate"]))
                    >= math.ceil(seconds * int(row["sample_rate"]))]
        super().__init__(rows, split, per_domain=per_domain, seed=seed, seconds=seconds)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        if index < 0:
            return torch.zeros(1, round(self.seconds * SAMPLE_RATE)), -1 if self.training else ""
        row = self.rows[index]
        track = row["source_id"]
        if self.training:
            generator = torch.Generator().manual_seed(matched_seed("crop", self.seed, epoch, track))
            return self.read_crop(row, generator), matched_seed("training", self.seed, epoch, track)
        return self.read_crop(row), track


class TrackSampler(Sampler):
    """Shard a permutation exactly once. Empty slots align the final DDP update."""

    def __init__(self, dataset, rank, world_size, window_size=1):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.window_size = window_size
        self.epoch = 0
        self.skip = 0

    def __len__(self):
        # Align epoch boundaries to optimizer steps so Lightning's batch-based
        # validation trigger stays aligned to global_step across epochs.
        return math.ceil(len(self.dataset) / (self.world_size * self.window_size)) * self.window_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        epoch = self.epoch
        size = len(self.dataset)
        if self.dataset.training:
            indices = jamendo_track_order(size, self.dataset.seed, epoch).tolist()
        else:
            indices = list(range(size))
        indices.extend([-1] * (len(self) * self.world_size - size))
        offset = self.skip * self.world_size + self.rank
        self.skip = 0
        return ((index, epoch) for index in indices[offset::self.world_size])


class TrackDataLoader(DataLoader):
    batches_seen = 0

    def state_dict(self):
        return {"epoch": self.sampler.epoch, "batches_seen": self.batches_seen}

    def load_state_dict(self, state_dict):
        self.batches_seen = state_dict["batches_seen"]
        self.sampler.set_epoch(state_dict["epoch"])
        self.sampler.skip = self.batches_seen * self.batch_size


class RollingCheckpoint(Checkpoint):
    def __init__(self, path):
        self.path = path
        self.saved_step = 0

    def on_train_start(self, trainer, model):
        saved_step = 0
        if trainer.is_global_zero and self.path.is_file():
            if trainer.ckpt_path and Path(trainer.ckpt_path).resolve() == self.path.resolve():
                saved_step = trainer.global_step
            else:
                checkpoint = torch.load(self.path, map_location="cpu", weights_only=False, mmap=True)
                saved_step = checkpoint["global_step"]
        self.saved_step = trainer.strategy.broadcast(saved_step)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        trainer.train_dataloader.batches_seen = batch_idx + 1

    def on_validation_end(self, trainer, model):
        # A timed stop triggers validation before Lightning closes the partial epoch.
        if trainer.should_stop or trainer.global_step % model.hparams.config["validate_every"] == 0:
            self._save_if_newer(trainer)

    def on_train_end(self, trainer, model):
        self._save_if_newer(trainer)

    def _save_if_newer(self, trainer):
        step = trainer.global_step
        if step <= self.saved_step:
            return
        staging_path = self.path.with_suffix(".ckpt.tmp")
        trainer.save_checkpoint(staging_path)
        if trainer.is_global_zero:
            os.replace(staging_path, self.path)
            print(f"Saved {self.path} at optimizer update {step}", flush=True)
        self.saved_step = step


class SubmissionTimer(Timer):
    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if not model.jamendo or (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
            super().on_train_batch_end(trainer, model, outputs, batch, batch_idx)

    def load_state_dict(self, state_dict):
        # A resumed submission has a new wall-time allocation, not the old timer's elapsed budget.
        pass


class SmokeCheck(Callback):
    def __init__(self):
        self.epochs = 0
        self.validations = 0
        self.gradient_steps = 0

    def on_train_start(self, trainer, model):
        if trainer.world_size != 2:
            raise RuntimeError("The smoke test expects two DDP ranks on two GPUs")
        self.initial_weight = next(model.network.parameters()).detach().clone()
        print(f"Smoke rank={trainer.global_rank}, local_rank={trainer.local_rank}, "
              f"device={model.device}, gpu={torch.cuda.get_device_name(model.device)}", flush=True)

    def on_before_optimizer_step(self, trainer, model, optimizer):
        gradients = [parameter.grad for parameter in model.network.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError("Missing or non-finite postfilter gradients")
        norm = torch.stack([value.detach().float().square().sum() for value in gradients]).sum().sqrt()
        if norm <= 0:
            raise RuntimeError("No gradient reaches the postfilter")
        model.log("smoke/gradient_norm", norm, sync_dist=True)
        self.gradient_steps += 1

    def on_train_epoch_end(self, trainer, model):
        self.epochs += 1

    def on_validation_epoch_end(self, trainer, model):
        if not trainer.sanity_checking:
            self.validations += 1

    def on_train_end(self, trainer, model):
        steps = model.hparams.config.get("jamendo_smoke_steps")
        if steps is not None:
            metric = trainer.callback_metrics.get("val/spectral")
            if (trainer.global_step != steps or self.gradient_steps != steps or self.validations != 1
                    or metric is None or not math.isfinite(float(metric))):
                raise RuntimeError("Jamendo smoke needs all optimizer steps and one finite scheduled validation")
        elif self.epochs < 2 or self.validations < 2 or self.gradient_steps < 4:
            raise RuntimeError("Smoke training stopped before two complete train/validation epochs")
        if torch.equal(self.initial_weight, next(model.network.parameters()).detach()):
            raise RuntimeError("Postfilter weights did not update")
        if model.ema_network is not None:
            if model.ema_updates.item() != self.gradient_steps:
                raise RuntimeError("EMA did not update with the optimizer")
            if not all(torch.isfinite(parameter).all() for parameter in model.ema_network.parameters()):
                raise RuntimeError("Non-finite EMA weights")
        if steps is not None:
            peak = torch.cuda.max_memory_allocated(model.device) / 1024 ** 3
            print(f"Smoke rank {trainer.global_rank}: peak allocated {peak:.2f} GiB", flush=True)


def tiny_rows(rows):
    selected, groups = [], {}
    for row in sorted(rows, key=lambda item: (float(item.get("duration_seconds") or 0), item["path"])):
        key = row["split"], row.get("dataset", row.get("domain", "audio"))
        chosen = groups.setdefault(key, set())
        if len(chosen) < 2 and row["group"] not in chosen:
            chosen.add(row["group"])
            selected.append(row)
    return selected


def main():
    launched_at = float(os.environ.get("FLOW_ALLOCATION_STARTED_AT", time.time()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs/postfilter.json")
    parser.add_argument("--geometry", choices=("cartesian", "disk", "cylindrical", "warped-disk"),
                        help="disk is the Euclidean Cartesian control; warped-disk changes the training bridge")
    parser.add_argument("--phase-loss", choices=("unit", "disk", "state-weighted"), default="unit")
    parser.add_argument("--phase-calibration", type=Path, help="Training-split calibration.json for state-weighted loss")
    parser.add_argument("--tau-multiplier", type=float, help="Transition scale in multiples of FlowDec's noise scale")
    parser.add_argument("--backbone", choices=("flowdec", "cyfm"))
    parser.add_argument("--flowdec-match", choices=("capacity", "optimizer"),
                        help="Fresh 20,000-update Jamendo comparison with FlowDec Adam and EMA. "
                             "Capacity also scales the CyFM U-Net to the FlowDec parameter budget.")
    parser.add_argument("--batch-size", type=int, help="Jamendo crops per GPU per forward pass")
    parser.add_argument("--accumulate", type=int, help="Jamendo batches per optimizer update")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=None,
                        help="--ema enables EMA evaluation; --no-ema uses ordinary weights")
    parser.add_argument("--name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--ablation", type=int, choices=ARMS, metavar="ARM")
    parser.add_argument("--campaign", help="Stable ablation directory suffix for automatic resume")
    parser.add_argument("--target-crops", type=int, help="Total scheduled crop slots for an ablation arm")
    parser.add_argument("--train-cache", type=Path, help="Ready six-codebook Jamendo crop cache")
    parser.add_argument("--noise-calibration", type=Path, help="Jamendo start-noise calibration JSON")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="short two-GPU check of training and validation")
    args = parser.parse_args()
    config = load_config(args.config)
    jamendo = config.get("dataset") == "jamendo"
    recipe = ARMS.get(args.ablation)
    target_updates = None
    if recipe:
        if not jamendo or args.devices * args.nodes != 2 or args.smoke:
            parser.error("Ablations use the Jamendo config and two GPUs")
        if (any(value is not None for value in (args.geometry, args.backbone, args.flowdec_match, args.ema,
                                                args.batch_size, args.accumulate))
                or args.phase_loss != "unit"):
            parser.error("The ablation arm supplies geometry, phase loss, optimizer, EMA and batching")
        if args.campaign and (args.resume or args.name or Path(args.campaign).name != args.campaign
                              or args.campaign in {".", ".."}):
            parser.error("--campaign needs a directory suffix and manages the name and checkpoint automatically")
        crops = FIRST_CROPS if args.target_crops is None else args.target_crops
        if crops <= 0 or crops % 32_000:
            parser.error("The ablation target must be a positive multiple of the 32,000-crop validation interval")
        target_updates = crops // recipe.batch
        args.geometry, args.phase_loss = recipe.geometry, recipe.phase_loss
        args.name = args.name or f"jamendo-a{args.ablation:02d}-{recipe.name}-k6-seed{args.seed}"
        memory_heavy = {"frequency-first", "convnext", "ncsnpp", "band-attention"}
        batch_size = 2 if recipe.architecture in memory_heavy else 4
        config.update(ablation_arm=args.ablation, width=recipe.width, architecture=recipe.architecture,
                      start_noise=recipe.noise, schedule=recipe.schedule, ema=True,
                      ema_decay=0.999 ** (recipe.batch / 64), batch_size=batch_size,
                      accumulate=recipe.batch // (2 * batch_size), effective_batch=recipe.batch,
                      validate_every=32_000 // recipe.batch)
        if recipe.noise.startswith("jamendo-"):
            calibration = args.noise_calibration or (args.train_cache / CALIBRATION_FILE if args.train_cache else None)
            if calibration is None:
                parser.error("This arm needs --noise-calibration or a calibrated --train-cache")
            config["noise_calibration"] = json.loads(calibration.read_text())
    elif args.geometry is None:
        parser.error("Specify --geometry or --ablation")
    elif any(value is not None for value in (args.campaign, args.target_crops, args.noise_calibration)):
        parser.error("--campaign, --target-crops and --noise-calibration require --ablation")
    if args.geometry == "disk":
        args.geometry = "cartesian"
    if args.phase_loss != "unit" and (not jamendo or args.geometry != "cylindrical"):
        parser.error("--phase-loss disk/state-weighted requires the cylindrical Jamendo arm")
    if args.geometry == "warped-disk" and not jamendo:
        parser.error("--geometry warped-disk requires the Jamendo config")
    if args.phase_calibration and args.phase_loss != "state-weighted":
        parser.error("--phase-calibration is used with --phase-loss state-weighted")
    if args.tau_multiplier is not None and (not math.isfinite(args.tau_multiplier) or args.tau_multiplier <= 0
            or (args.geometry != "warped-disk" and args.phase_loss != "state-weighted")):
        parser.error("--tau-multiplier must be positive and is used with warped-disk or state-weighted")
    if args.phase_loss != "unit":
        config["phase_loss"] = args.phase_loss
    if args.geometry == "warped-disk" or args.phase_loss == "state-weighted":
        config["tau_multiplier"] = args.tau_multiplier if args.tau_multiplier is not None else 1.0
    if args.phase_loss == "state-weighted":
        if args.resume and not args.phase_calibration:
            saved = torch.load(args.resume, map_location="cpu", weights_only=False)
            config["phase_calibration"] = saved["hyper_parameters"]["config"]["phase_calibration"]
            del saved
        else:
            calibration_path = args.phase_calibration or SCRATCH / "jamendo-geometry" / "calibration.json"
            config["phase_calibration"] = json.loads(calibration_path.read_text())
        if args.tau_multiplier is None:
            config["tau_multiplier"] = config["phase_calibration"]["tau_multiplier"]
    legacy_smoke = args.smoke and not jamendo
    if not jamendo and any(value is not None for value in (args.backbone, args.batch_size, args.accumulate, args.flowdec_match)):
        parser.error("--backbone, --batch-size, --accumulate and --flowdec-match are for the Jamendo experiment")
    if jamendo:
        config["backbone"] = args.backbone or "cyfm"
        if args.flowdec_match:
            config.update(optimizer_recipe="flowdec", max_updates=20_000, ema=True, batch_size=4, accumulate=8)
            if args.flowdec_match == "capacity" and config["backbone"] == "cyfm":
                # Closest shared time width at spatial width 160 for the pinned NCSN++ count.
                config.update(width=160, time_embedding_dim=910)
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.accumulate is not None:
            config["accumulate"] = args.accumulate
        if config["codebooks"] != 6 or config["evaluation_seconds"] != 10:
            parser.error("Jamendo runs use six codebooks and ten-second evaluation")
        if min(config[key] for key in ("batch_size", "accumulate", "max_epochs", "validate_every")) < 1:
            parser.error("Jamendo needs positive batch size, accumulation, epoch budget and validation interval")
        if config["backbone"] == "flowdec" and (args.geometry != "cartesian" or args.ema is False):
            parser.error("The FlowDec baseline uses Cartesian geometry and EMA")
    if args.ema is not None:
        config["ema"] = args.ema
    if jamendo and config["backbone"] == "flowdec":
        config["ema"] = True
    if args.smoke:
        if args.devices * args.nodes != 2:
            raise ValueError("Use two GPUs for --smoke")
        if args.resume is not None:
            parser.error("Start smoke runs from fresh weights")
    if jamendo and args.smoke:
        config.update(jamendo_smoke_steps=3, validate_every=3, validation_per_domain=2)
    max_steps = target_updates if target_updates is not None else config.get(
        "jamendo_smoke_steps", config.get("max_updates", -1)
    )
    if args.train_cache and (not jamendo or max_steps < 1):
        parser.error("--train-cache requires a Jamendo run with a finite update budget")
    if legacy_smoke:
        if os.environ.get("WANDB_MODE", "online") not in {"online", "run"}:
            raise ValueError("This smoke test checks online W&B; unset WANDB_MODE=offline/disabled")
        config.update(smoke=True, max_steps=-1, max_epochs=2,
                      samples_per_epoch=2 * args.devices * args.nodes * config["batch_size"] * config["accumulate"],
                      workers=2, validation_per_domain=2, test_per_domain=2, metric_batches=2,
                      checkpoint_every=1, milestone_every=2, checkpoint_nfe=2,
                      validation_nfe=[1, 2], evaluation_nfe=[2], evaluation_seeds=[2])
    if (config["checkpoint_nfe"] not in config["validation_nfe"]
            or (jamendo and not config["evaluation_seeds"])
            or (not jamendo and config["metric_batches"] < 1)):
        raise ValueError("Validation must measure checkpoint_nfe with at least one batch and noise draw")
    if config["solver"] not in {"euler", "midpoint"} or any(
        nfe < 1 or (config["solver"] == "midpoint" and nfe != 1 and nfe % 2)
        for nfe in config["validation_nfe"] + config["evaluation_nfe"]
    ):
        raise ValueError("Use positive NFE; midpoint budgets must be even, with NFE=1 using Euler")
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    if legacy_smoke:
        # Select before filesystem checks: a smoke job should not stat the whole corpus.
        with Path(config["manifest"]).open(newline="") as handle:
            rows = tiny_rows(list(csv.DictReader(handle)))
        if not rows:
            raise ValueError("The prepared manifest is empty")
    else:
        rows = read_manifest(config["manifest"], dataset=config.get("dataset"))
    if config.get("require_data_audit", True):
        report_path = Path(config["manifest"]).with_suffix(".summary.json")
        if not report_path.is_file():
            raise FileNotFoundError("Run slurm/postfilter_data.slurm before training: missing data audit")
        report = json.loads(report_path.read_text())
        if (report.get("manifest_sha256") != file_hash(config["manifest"])
                or report.get("audit") != "full_decode_and_seek_v1" or not report.get("loader_check")):
            raise ValueError("Dataset audit is incomplete or stale; rerun the data preparation job")
        if report["config"]["evaluation_seconds"] != config.get("evaluation_seconds", 2):
            raise ValueError("Data preparation and training configs disagree on test duration")
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0"))))
    crop_seed = args.seed + 10000 * rank
    if jamendo:
        track_ids = {row.get("source_id") for row in rows}
        if None in track_ids or "" in track_ids or len(track_ids) != len(rows):
            raise ValueError("The audited Jamendo manifest must contain one row per track ID")
        train = JamendoCrops(rows, "train", args.seed)
        valid = JamendoCrops(rows, "val", args.seed, per_domain=config["validation_per_domain"],
                            seconds=config["evaluation_seconds"])
    else:
        train = AudioCrops(rows, "train", samples=config["samples_per_epoch"], seed=crop_seed)
        valid = AudioCrops(rows, "val", per_domain=config["validation_per_domain"], seed=args.seed)
    # Fingerprint the source list and file metadata without reading whole audio files.
    sources = hashlib.sha256()
    for row in rows:
        stat = Path(row["path"]).stat()
        if row.get("file_size") and (stat.st_size != int(row["file_size"]) or stat.st_mtime_ns != int(row["mtime_ns"])):
            raise ValueError(f"Audio changed since its decode audit: {row['path']}")
        sources.update(f"{row['path']}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    provenance = {"manifest_sha256": file_hash(config["manifest"]), "sources": sources.hexdigest(),
                  "geometry": args.geometry, "seed": args.seed,
                  "world_size": args.devices * args.nodes, "settings": config}
    model = Postfilter(config, args.geometry, args.seed, provenance)
    effective_batch = config["batch_size"] * args.devices * args.nodes * config["accumulate"]
    if args.train_cache:
        train = CachedJamendoCrops(train, args.train_cache, provenance["manifest_sha256"],
                                   model.codec.identity, effective_batch, max_steps)
    averaging = "ema" if config.get("ema", False) else "noema"
    prefix = ("smoke-" if args.smoke else "") + (f"jamendo-{config['backbone']}-" if jamendo else "")
    arm = args.geometry if args.phase_loss == "unit" else f"{args.geometry}-{args.phase_loss}"
    if args.flowdec_match:
        model_size = f"w{config['width']}" if config["backbone"] == "cyfm" else "ncsnpp"
        arm = f"{arm}-{model_size}-adam-20k"
    label = args.name or f"{prefix}{arm}-{averaging}-k{config['codebooks']}-seed{args.seed}"
    if Path(label).name != label or label in {".", ".."}:
        raise ValueError("--name must be a folder name")
    resume = args.resume
    # Slurm ranks share the job ID; local DDP children inherit this generated token.
    token = args.campaign or os.environ.setdefault(
        "FLOW_POSTFILTER_RUN_ID", os.environ.get("SLURM_JOB_ID") or uuid4().hex[:12]
    )
    run_dir = resume.resolve().parent if resume else Path(config["output"]) / f"{label}-{token}"
    run_dir.mkdir(parents=True, exist_ok=True)
    requeued = int(os.environ.get("SLURM_RESTART_COUNT", "0")) > 0
    if args.campaign and (run_dir / "last.ckpt").is_file():
        resume = run_dir / "last.ckpt"
    if resume is None and requeued:
        hpc = list(run_dir.glob("hpc_ckpt_*.ckpt"))
        if hpc:
            resume = max(hpc, key=lambda item: int(item.stem.rsplit("_", 1)[1]))
        elif (run_dir / "last.ckpt").exists():
            resume = run_dir / "last.ckpt"
        elif jamendo and (run_dir / "best.ckpt").exists():
            resume = run_dir / "best.ckpt"
    if resume is None and ((run_dir / "last.ckpt").exists() or (jamendo and (run_dir / "best.ckpt").exists())):
        raise FileExistsError("This run directory already has a checkpoint; use --resume or a fresh submission")
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)
    if recipe and resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False, mmap=True)
        model.on_load_checkpoint(checkpoint)
        saved_step = checkpoint["global_step"]
        del checkpoint
        if saved_step >= max_steps:
            print(f"{run_dir} already reached update {saved_step:,} (target {max_steps:,})", flush=True)
            return
    run_id = hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]
    logger = WandbLogger(project=config["wandb_project"], name=run_dir.name, id=run_id,
                         resume="allow" if resume or args.campaign else "never",
                         group=args.campaign or f"{'smoke-' if args.smoke else ''}{arm}-{averaging}",
                         save_dir=config["wandb_dir"], log_model=False)
    logger.log_hyperparams({"experiment": config, "geometry": args.geometry, "seed": args.seed,
                           "codec": model.codec.identity,
                           "trainable_parameters": model.trainable_parameters, "optimizer_recipe": model.optimizer_recipe,
                           "global_batch_size": effective_batch})
    callbacks = [RollingCheckpoint(run_dir / "last.ckpt")] if jamendo else [
        ModelCheckpoint(dirpath=run_dir, filename="best", monitor="val/spectral", mode="min",
                        save_top_k=1, enable_version_counter=False),
        ModelCheckpoint(dirpath=run_dir, save_top_k=0, save_last=True,
                        every_n_train_steps=config["checkpoint_every"]),
        ModelCheckpoint(dirpath=run_dir, filename="step-{step:08d}", save_top_k=-1,
                        auto_insert_metric_name=False, every_n_train_steps=config["milestone_every"]),
    ]
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    smoke = SmokeCheck() if args.smoke else None
    if smoke is not None:
        callbacks.append(smoke)
    deadline = None
    if not args.smoke:
        deadline = launched_at + config["allocation_hours"] * 3600
        if os.environ.get("SLURM_JOB_END_TIME"):
            deadline = min(deadline, float(os.environ["SLURM_JOB_END_TIME"]))
        deadline -= config["shutdown_margin_minutes"] * 60
        remaining = int(deadline - time.time())
        if remaining <= 0:
            raise RuntimeError("No training time remains after startup and checkpoint-saving margin")
        callbacks.append(SubmissionTimer(duration=timedelta(seconds=remaining), interval="step"))
    loader_options = dict(num_workers=config["workers"], pin_memory=True, persistent_workers=config["workers"] > 0)
    if jamendo:
        world_size = args.devices * args.nodes
        accumulate = config["accumulate"]
        window_size = config["batch_size"] * accumulate if accumulate > 1 else 1
        loaders = [
            TrackDataLoader(train, batch_size=config["batch_size"],
                            sampler=TrackSampler(train, rank, world_size, window_size),
                            **loader_options),
            DataLoader(valid, batch_size=1, sampler=TrackSampler(valid, rank, world_size), **loader_options),
        ]
        if max_steps < 1 and len(loaders[0]) // accumulate * config["max_epochs"] < config["validate_every"]:
            raise ValueError("The epoch budget must reach at least one scheduled validation")
    else:
        loaders = [DataLoader(dataset, batch_size=config["batch_size"], shuffle=index == 0, drop_last=index == 0,
                              generator=torch.Generator().manual_seed(crop_seed + index), **loader_options)
                   for index, dataset in enumerate((train, valid))]
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices, num_nodes=args.nodes,
        strategy=DDPStrategy(find_unused_parameters=False) if args.devices * args.nodes > 1 else "auto",
        precision=config["precision"], max_steps=max_steps,
        max_epochs=config["max_epochs"] if (jamendo or args.smoke) and target_updates is None
        and "max_updates" not in config else -1,
        accumulate_grad_batches=config["accumulate"], gradient_clip_val=None if jamendo else 1.0,
        val_check_interval=1.0 if legacy_smoke else config["validate_every"] * config["accumulate"],
        check_val_every_n_epoch=1 if legacy_smoke else None,
        num_sanity_val_steps=0 if jamendo else 1, use_distributed_sampler=not jamendo,
        logger=logger, callbacks=callbacks, default_root_dir=run_dir,
        plugins=[SLURMEnvironment(auto_requeue=False)] if "SLURM_JOB_ID" in os.environ else None,
        enable_progress_bar="SLURM_JOB_ID" not in os.environ, log_every_n_steps=1 if args.smoke else 20,
    )
    if trainer.is_global_zero:
        (run_dir / "run.json").write_text(json.dumps({"name": run_dir.name, "wandb_id": run_id,
                "config": config, "geometry": args.geometry, "resume": str(resume) if resume else None,
                "training_cache": str(args.train_cache) if args.train_cache else None,
                "target_updates": max_steps if max_steps > 0 else None,
                "training_deadline_unix": deadline}, indent=2))
        if legacy_smoke:
            with (run_dir / "smoke_manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Run directory: {run_dir}", flush=True)
        if deadline is not None:
            if max_steps > 0:
                budget = f"target {max_steps:,} optimizer updates"
            else:
                budget = f"up to {config['max_epochs']} full track passes" if jamendo else "no step or epoch cap"
            print(f"{int(deadline - time.time())} seconds remain before the wall-time margin; "
                  f"{budget}, no automatic requeue.", flush=True)
    trainer.fit(model, *loaders, ckpt_path=str(resume) if resume else None)
    if recipe and trainer.is_global_zero:
        progress = {
            "global_step": trainer.global_step, "target_updates": max_steps,
            "real_crops": real_crops(len(train), effective_batch, trainer.global_step),
            "target_reached": trainer.global_step >= max_steps,
            "fit_seconds": model.fit_seconds + (time.monotonic() - model.started if model.started else 0),
        }
        write_json(run_dir / "training-progress.json", progress)
        print(f"{run_dir.name}: {trainer.global_step:,}/{max_steps:,} optimizer updates", flush=True)
    if not jamendo:
        trainer.save_checkpoint(run_dir / "last.ckpt")
    if args.smoke:
        trainer.strategy.barrier()
        if not jamendo and not (run_dir / "best.ckpt").is_file():
            raise RuntimeError("Smoke test did not create best.ckpt")
        if not (run_dir / "last.ckpt").is_file():
            raise RuntimeError("Smoke test did not create last.ckpt")
    if jamendo and args.smoke and trainer.is_global_zero:
        print(f"SMOKE PASSED: {trainer.global_step} optimizer updates; checkpoints in {run_dir}", flush=True)
    if legacy_smoke:
        test = AudioCrops(rows, "test", per_domain=2, seed=args.seed, seconds=config["evaluation_seconds"])
        test_loader = DataLoader(test, batch_size=config["batch_size"], num_workers=2, pin_memory=True)
        # This reloads the saved weights through Lightning, then runs distributed testing.
        results = trainer.test(model, dataloaders=test_loader, ckpt_path=str(run_dir / "best.ckpt"))
        if not results or not results[0] or not all(math.isfinite(float(value)) for value in results[0].values()):
            raise RuntimeError("Missing or non-finite test metrics")
        if trainer.is_global_zero:
            run = logger.experiment
            if run.settings.mode != "online":
                raise RuntimeError("W&B did not start an online run")
            url = run.url
            run.log({"smoke/passed": 1, "smoke/epochs": smoke.epochs,
                     "smoke/validation_cycles": smoke.validations,
                     "smoke/optimizer_steps_per_rank": smoke.gradient_steps})
            run.finish()
            (run_dir / "smoke_success.json").write_text(json.dumps({"passed": True,
                "world_size": trainer.world_size, "epochs": smoke.epochs, "validation_cycles": smoke.validations,
                "optimizer_steps_per_rank": smoke.gradient_steps, "checkpoint_reloaded": str(best),
                "evaluation_weights": averaging,
                "ema_updates_in_reloaded_checkpoint": model.ema_updates.item() if model.ema_updates is not None else 0,
                "test_metrics": results[0], "wandb_url": url}, indent=2, allow_nan=False))
            print(f"SMOKE PASSED: {run_dir / 'smoke_success.json'}\nW&B: {url}", flush=True)
        trainer.strategy.barrier()


if __name__ == "__main__":
    main()
