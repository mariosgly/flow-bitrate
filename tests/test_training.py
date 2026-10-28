import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from flow_bitrate.train import RatePredictor


def test_lightning_train_checkpoint_resume(tmp_path):
    torch.set_num_threads(2)
    pl.seed_everything(1)
    x = torch.randn(32, 12)
    target = torch.log1p(x[:, :4].square())
    loader = DataLoader(TensorDataset(x, target), batch_size=8)
    model = RatePredictor(x.mean(0).tolist(), x.std(0).tolist(), base_dim=8, use_flow=True)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0)
    trainer.fit(model, loader, loader)
    path = tmp_path / "last.ckpt"
    trainer.save_checkpoint(path)
    restored = RatePredictor.load_from_checkpoint(path, weights_only=False).eval()
    torch.testing.assert_close(model(x), restored(x))
    resumed = pl.Trainer(accelerator="cpu", devices=1, max_epochs=2, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0)
    resumed.fit(restored, loader, loader, ckpt_path=path)
    assert resumed.global_step == 8
    baseline = RatePredictor(x.mean(0).tolist(), x.std(0).tolist(), base_dim=8, use_flow=False)
    altered = x.clone()
    altered[:, 8:] += 100
    torch.testing.assert_close(baseline(x), baseline(altered))
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in baseline.parameters())
