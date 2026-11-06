"""Train the copied CyFM U-Net on NDAC reconstructions, with Lightning + W&B."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4

import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio.functional as AF
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .codec import BLOCK_SAMPLES, ROOT, SAMPLE_RATE, file_hash
from .data import load_config, read_manifest
from .postfilter import Postfilter


class AudioCrops(Dataset):
    def __init__(self, rows, split, samples=8192, per_domain=32, seed=0, seconds=2):
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} recordings in the manifest")
        self.training = split == "train"
        self.seconds = seconds
        self.samples, self.seed, self.generator = samples, seed, None
        self.domains = sorted({row.get("domain") or "audio" for row in self.rows})
        self.by_domain = [[row for row in self.rows if (row.get("domain") or "audio") == domain]
                          for domain in self.domains]
        self.by_source = [[[row for row in candidates if row.get("dataset", "audio") == source]
                           for source in sorted({row.get("dataset", "audio") for row in candidates})]
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
                            if sum((r.get("domain") or "audio") == (candidates[0].get("domain") or "audio")
                                   for r in selected) >= per_domain:
                                break
                    else:
                        continue
                    break
            buckets = [[row for row in selected if (row.get("domain") or "audio") == domain]
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
            start = torch.randint(first, last + 1, (), generator=self.generator).item()
        else:
            start = first if row.get("start_seconds") else (first + last) // 2
        value, _ = sf.read(row["path"], start=start, frames=needed, dtype="float32", always_2d=True)
        audio = torch.from_numpy(value).mean(1)[None]
        if rate != SAMPLE_RATE:
            audio = AF.resample(audio, rate, SAMPLE_RATE)
        target_samples = round(self.seconds * SAMPLE_RATE)
        if audio.shape[-1] < target_samples or not torch.isfinite(audio).all():
            raise ValueError(f"Short or non-finite crop: {row['path']}")
        return audio[..., :target_samples]  # [1, 96000] during training; mono 48 kHz


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
        if self.epochs < 2 or self.validations < 2 or self.gradient_steps < 4:
            raise RuntimeError("Smoke training stopped before two complete train/validation epochs")
        if torch.equal(self.initial_weight, next(model.network.parameters()).detach()):
            raise RuntimeError("Postfilter weights did not update")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs/postfilter.json")
    parser.add_argument("--geometry", choices=("cartesian", "cylindrical"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke", action="store_true", help="two tiny epochs, checkpoint reload and DDP test; real W&B logging")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.smoke:
        if args.devices * args.nodes != 2:
            raise ValueError("Use two GPUs for --smoke")
        if os.environ.get("WANDB_MODE", "online") not in {"online", "run"}:
            raise ValueError("This smoke test checks online W&B; unset WANDB_MODE=offline/disabled")
        config.update(smoke=True, max_steps=-1, max_epochs=2, samples_per_epoch=32,
                      workers=2, validation_per_domain=2, test_per_domain=2, metric_batches=2,
                      checkpoint_every=1, milestone_every=2, checkpoint_nfe=2,
                      validation_nfe=[1, 2], evaluation_nfe=[2], evaluation_seeds=[2])
    if config["checkpoint_nfe"] not in config["validation_nfe"] or config["metric_batches"] < 1:
        raise ValueError("Validation must measure checkpoint_nfe on at least one batch")
    if config["solver"] not in {"euler", "midpoint"} or any(
        nfe < 1 or (config["solver"] == "midpoint" and nfe != 1 and nfe % 2)
        for nfe in config["validation_nfe"] + config["evaluation_nfe"]
    ):
        raise ValueError("Use positive NFE; midpoint budgets must be even, with NFE=1 using Euler")
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    if args.smoke:
        # Select before filesystem checks: a smoke job should not stat the whole corpus.
        with Path(config["manifest"]).open(newline="") as handle:
            rows = tiny_rows(list(csv.DictReader(handle)))
        if not rows:
            raise ValueError("The prepared manifest is empty")
    else:
        rows = read_manifest(config["manifest"])
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
    label = args.name or f"{'smoke-' if args.smoke else ''}{args.geometry}-k{config['codebooks']}-seed{args.seed}"
    if Path(label).name != label or label in {".", ".."}:
        raise ValueError("--name must be a folder name")
    resume = args.resume
    # Slurm ranks share the job ID; local DDP children inherit this generated token.
    token = os.environ.setdefault("FLOW_POSTFILTER_RUN_ID", os.environ.get("SLURM_JOB_ID") or uuid4().hex[:12])
    run_dir = resume.resolve().parent if resume else Path(config["output"]) / f"{label}-{token}"
    run_dir.mkdir(parents=True, exist_ok=True)
    requeued = int(os.environ.get("SLURM_RESTART_COUNT", "0")) > 0
    if resume is None and requeued:
        hpc = list(run_dir.glob("hpc_ckpt_*.ckpt"))
        if hpc:
            resume = max(hpc, key=lambda item: int(item.stem.rsplit("_", 1)[1]))
        elif (run_dir / "last.ckpt").exists():
            resume = run_dir / "last.ckpt"
    if resume is None and (run_dir / "last.ckpt").exists():
        raise FileExistsError("This run directory already has a checkpoint; use --resume or a fresh submission")
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)
    run_id = hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]
    logger = WandbLogger(project=config["wandb_project"], name=run_dir.name, id=run_id,
                         resume="allow" if resume else "never", group="smoke" if args.smoke else args.geometry,
                         save_dir=config["wandb_dir"], log_model=False)
    logger.log_hyperparams({"experiment": config, "geometry": args.geometry, "seed": args.seed,
                           "codec": model.codec.identity,
                           "trainable_parameters": sum(p.numel() for p in model.network.parameters()),
                           "global_batch_size": config["batch_size"] * args.devices * args.nodes * config["accumulate"]})
    callbacks = [
        ModelCheckpoint(dirpath=run_dir, filename="best", monitor="val/spectral", mode="min",
                        save_top_k=1, enable_version_counter=False),
        ModelCheckpoint(dirpath=run_dir, save_top_k=0, save_last=True,
                        every_n_train_steps=config["checkpoint_every"]),
        ModelCheckpoint(dirpath=run_dir, filename="step-{step:08d}", save_top_k=-1,
                        auto_insert_metric_name=False, every_n_train_steps=config["milestone_every"]),
        LearningRateMonitor(logging_interval="step"),
    ]
    smoke = SmokeCheck() if args.smoke else None
    if smoke is not None:
        callbacks.append(smoke)
    loaders = [DataLoader(dataset, batch_size=config["batch_size"], shuffle=index == 0,
                          num_workers=config["workers"], pin_memory=True, drop_last=index == 0,
                          persistent_workers=config["workers"] > 0,
                          generator=torch.Generator().manual_seed(crop_seed + index))
               for index, dataset in enumerate((train, valid))]
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices, num_nodes=args.nodes,
        strategy=DDPStrategy(find_unused_parameters=False) if args.devices * args.nodes > 1 else "auto",
        precision=config["precision"], max_steps=config["max_steps"], max_epochs=config.get("max_epochs", -1),
        accumulate_grad_batches=config["accumulate"], gradient_clip_val=1.0,
        val_check_interval=1.0 if args.smoke else config["validate_every"] * config["accumulate"],
        check_val_every_n_epoch=1 if args.smoke else None,
        num_sanity_val_steps=1, logger=logger, callbacks=callbacks, default_root_dir=run_dir,
        enable_progress_bar="SLURM_JOB_ID" not in os.environ, log_every_n_steps=1 if args.smoke else 20,
    )
    if trainer.is_global_zero:
        (run_dir / "run.json").write_text(json.dumps({"name": run_dir.name, "wandb_id": run_id,
                "config": config, "geometry": args.geometry, "resume": str(resume) if resume else None}, indent=2))
        if args.smoke:
            with (run_dir / "smoke_manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Run directory: {run_dir}", flush=True)
    trainer.fit(model, *loaders, ckpt_path=str(resume) if resume else None)
    trainer.save_checkpoint(run_dir / "last.ckpt")
    if args.smoke:
        trainer.strategy.barrier()
        best = run_dir / "best.ckpt"
        if not best.is_file() or not (run_dir / "last.ckpt").is_file():
            raise RuntimeError("Smoke test did not create both best.ckpt and last.ckpt")
        test = AudioCrops(rows, "test", per_domain=2, seed=args.seed, seconds=config["evaluation_seconds"])
        test_loader = DataLoader(test, batch_size=config["batch_size"], num_workers=2, pin_memory=True)
        # This reloads the saved weights through Lightning, then runs distributed testing.
        results = trainer.test(model, dataloaders=test_loader, ckpt_path=str(best))
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
                "test_metrics": results[0], "wandb_url": url}, indent=2, allow_nan=False))
            print(f"SMOKE PASSED: {run_dir / 'smoke_success.json'}\nW&B: {url}", flush=True)
        trainer.strategy.barrier()


if __name__ == "__main__":
    main()
