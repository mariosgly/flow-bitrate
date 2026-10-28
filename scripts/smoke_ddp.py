"""Two actual CPU processes; checks Lightning DDP and synchronized logging without weights."""

import tempfile
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from flow_bitrate.train import RatePredictor
from flow_bitrate.codec import SCRATCH


def main():
    torch.set_num_threads(1)
    pl.seed_everything(42)
    x = torch.randn(32, 12)
    loader = DataLoader(TensorDataset(x, x[:, :4].square().log1p()), batch_size=8)
    model = RatePredictor([0.] * 12, [1.] * 12, base_dim=8)
    (SCRATCH / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH / "tmp") as directory:
        trainer = pl.Trainer(accelerator="cpu", devices=2, strategy="ddp_spawn", max_epochs=1,
                             logger=False, default_root_dir=directory, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
        trainer.fit(model, loader, loader)
        trainer.save_checkpoint(Path(directory) / "ddp.ckpt")
        assert trainer.world_size == 2
    print("Two-rank CPU DDP training finished.")


if __name__ == "__main__":
    main()
