"""Two training jobs: predict rate/distortion curves, or fine-tune the flow backbone."""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader

from .codec import ROOT, FlowCodec
from .data import AudioDataset, FeatureDataset, check_manifest, load_cache, load_config


class RatePredictor(pl.LightningModule):
    def __init__(self, mean, std, base_dim, use_flow=True, lr=3e-4, provenance=None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-5))
        self.net = nn.Sequential(nn.Linear(len(mean), 256), nn.SiLU(), nn.Linear(256, 128),
                                 nn.SiLU(), nn.Linear(128, 4))

    def forward(self, features):
        normalized = (features - self.mean) / self.std
        if not self.hparams.use_flow:
            normalized = normalized.clone()
            normalized[:, self.hparams.base_dim:] = 0  # same MLP size for both ablations
        return self.net(normalized)  # [B, 4], predicted log1p distortion

    def training_step(self, batch, batch_idx):
        x, target = batch
        loss = (self(x) - target).square().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        loss = (self(x) - target).square().mean()
        self.log("val/loss", loss, sync_dist=True, batch_size=len(x))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)


class FlowFinetuner(pl.LightningModule):
    def __init__(self, checkpoint_dir, flow_checkpoint=None, lr=1e-5, provenance=None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.codec = FlowCodec(checkpoint_dir, flow_checkpoint)
        self.codec.backbone.requires_grad_(True)

    def loss(self, audio, seed=None):
        _, codes = self.codec.encode(audio)
        generator = None if seed is None else torch.Generator(device=audio.device).manual_seed(seed)
        rate = torch.randint(4, (len(audio),), device=audio.device, generator=generator)
        coarse = self.codec.coarse_at_rates(codes, rate)
        return self.codec.flow_loss(audio, coarse, seed=seed)[0].mean()

    def training_step(self, audio, batch_idx):
        loss = self.loss(audio)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=len(audio))
        return loss

    def validation_step(self, audio, batch_idx):
        loss = self.loss(audio, seed=1000 + batch_idx)
        self.log("val/loss", loss, sync_dist=True, batch_size=len(audio))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.codec.backbone.parameters(), lr=self.hparams.lr, weight_decay=0)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["flow_state"] = self.codec.export_flow()
        checkpoint["source_codec"] = self.codec.identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/experiment.json")
    parser.add_argument("--task", choices=("allocator", "flow"), default="allocator")
    parser.add_argument("--features", choices=("baseline", "flow"), default="flow")
    parser.add_argument("--name", required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "auto"),
                        default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("32-true", "bf16-mixed"), default="32-true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="one train/val batch per rank; no saved run")
    args = parser.parse_args()
    config = load_config(args.config)
    pl.seed_everything(config["seed"], workers=True)
    cache = load_cache(config["cache"])
    check_manifest(cache, config["manifest"])
    settings = config[args.task]
    if args.task == "allocator":
        train, val = FeatureDataset(cache, "train"), FeatureDataset(cache, "val")
        model = RatePredictor(train.x.mean(0).tolist(), train.x.std(0, correction=0).tolist(),
                              cache["base"].shape[1], args.features == "flow", settings["lr"], cache["metadata"])
    else:
        train, val = AudioDataset(cache, "train"), AudioDataset(cache, "val")
        model = FlowFinetuner(config["checkpoints"], config.get("flow_checkpoint"), settings["lr"], cache["metadata"])
        if model.codec.identity != cache["metadata"]["codec"]:
            raise ValueError("Flow checkpoint differs from the cache's source decoder")
    run_dir = Path(config["output"]) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    last = run_dir / "last.ckpt"
    requeued = int(os.environ.get("SLURM_RESTART_COUNT", "0")) > 0
    resuming = args.resume or requeued
    resume_path = last
    if requeued:
        hpc_checkpoints = list(run_dir.glob("hpc_ckpt_*.ckpt"))
        if hpc_checkpoints:
            resume_path = max(hpc_checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if args.resume and not resume_path.exists():
        raise FileNotFoundError(resume_path)
    if last.exists() and not resuming:
        raise FileExistsError(f"{last} exists; use --resume or a new --name")
    if resuming and resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)["hyper_parameters"]
        if saved.get("provenance") != cache["metadata"]:
            raise ValueError("Resume cache/checkpoints differ from the saved run")
        if args.task == "allocator" and saved.get("use_flow") != (args.features == "flow"):
            raise ValueError("Cannot change the experiment ablation while resuming")
    # All ranks get the same id; Lightning only starts W&B on rank zero.
    import hashlib
    run_id = hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]
    logger = False
    if not args.smoke:
        logger = WandbLogger(project=config["wandb_project"], name=args.name, id=run_id,
                             resume="allow", save_dir=config["wandb_dir"], log_model=False)
        logger.log_hyperparams({"experiment": config, "cli": vars(args)})
    callback = ModelCheckpoint(dirpath=run_dir, filename="best", monitor="val/loss", mode="min",
                               save_last=True, save_top_k=1)
    workers = settings["workers"]
    loaders = [DataLoader(dataset, batch_size=settings["batch_size"], shuffle=index == 0,
                          num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
               for index, dataset in enumerate((train, val))]
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=args.devices, num_nodes=args.nodes,
        strategy=DDPStrategy(find_unused_parameters=False) if args.devices * args.nodes > 1 else "auto",
        precision=args.precision, max_epochs=settings["epochs"], logger=logger, callbacks=[callback],
        default_root_dir=run_dir, log_every_n_steps=10, gradient_clip_val=1.0,
        accumulate_grad_batches=settings.get("accumulate", 1), enable_progress_bar="SLURM_JOB_ID" not in os.environ,
        fast_dev_run=args.smoke,
    )
    trainer.fit(model, *loaders, ckpt_path=str(resume_path) if resuming and resume_path.exists() else None)


if __name__ == "__main__":
    main()
