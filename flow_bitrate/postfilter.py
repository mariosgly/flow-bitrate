"""Frozen NDAC, with the same conditional U-Net trained in either geometry."""

import time
from pathlib import Path

import pytorch_lightning as pl
import soundfile as sf
import torch

from .codec import BLOCK_SAMPLES, CODEBOOKS, SAMPLE_RATE, FlowCodec
from .cyfm_unet import CylindricalUNet
from .metrics import si_sdr, spectral_loss


class Postfilter(pl.LightningModule):
    def __init__(self, config, geometry, seed=0, provenance=None):
        super().__init__()
        if geometry not in {"cartesian", "cylindrical"}:
            raise ValueError("Choose cartesian or cylindrical")
        self.save_hyperparameters()
        self.codec = FlowCodec(config["checkpoints"], load_backbone=False)
        self.codec.requires_grad_(False)
        channels = 2 if geometry == "cartesian" else 3
        # State + coarse condition: [B, 4 or 6, 768, padded_frames].
        self.network = CylindricalUNet(base_channels=config["width"], in_channels=2 * channels,
                                       out_channels=2, velocity_bound=None)
        self.rate_index = CODEBOOKS.index(config["codebooks"])
        self.fit_seconds = 0.0
        self.started = None

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    @torch.no_grad()
    def prepare(self, audio):
        # Keep the frozen codec and complex STFT in float32 under mixed precision.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            _, codes = self.codec.encode(audio.float())
            coarse, _ = self.codec.coarse(codes, CODEBOOKS[self.rate_index])
            condition, target, info = self.codec.preprocess(coarse, audio.float())
        return coarse, condition, target, info

    def coordinates(self, value):
        if self.hparams.geometry == "cartesian":
            return torch.cat((value.real, value.imag), dim=1)
        return torch.cat((value.abs(), value.angle()), dim=1)

    def embedding(self, state):
        if self.hparams.geometry == "cartesian":
            return state
        amplitude, phase = state[:, :1], state[:, 1:]
        return torch.cat((amplitude, phase.cos(), phase.sin()), dim=1)

    def velocity(self, state, condition, time_value):
        value = torch.cat((self.embedding(state), condition), dim=1)
        return self.network(value, time_value).float()  # [B, 2, F, T]

    def start(self, condition, seed=None):
        generator = None if seed is None else torch.Generator(device=condition.device).manual_seed(seed)
        noise = torch.randn(condition.shape, dtype=condition.dtype, device=condition.device, generator=generator)
        return self.coordinates(condition + self.codec.sigma * noise), generator

    def flow_loss(self, condition, target, frames, seed=None):
        start, generator = self.start(condition, seed)
        end = self.coordinates(target)
        velocity = end - start
        if self.hparams.geometry == "cylindrical":
            difference = velocity[:, 1:]
            velocity = torch.cat((velocity[:, :1], torch.atan2(difference.sin(), difference.cos())), 1)
        time_value = torch.rand(len(start), device=start.device, generator=generator)
        state = start + time_value[:, None, None, None] * velocity
        prediction = self.velocity(state, self.embedding(self.coordinates(condition)), time_value)
        # Sum the two squared coordinate errors; padded STFT frames carry no loss.
        error = (prediction[..., :frames] - velocity[..., :frames]).square()
        return error.sum(1).mean(), error.mean((0, 2, 3))

    @torch.no_grad()
    def enhance(self, coarse, nfe, seed=0, solver="midpoint"):
        if nfe < 1 or solver not in {"euler", "midpoint"}:
            raise ValueError("Use positive NFE and euler or midpoint")
        # NFE=1 is Euler for both arms; midpoint otherwise needs an even budget.
        midpoint = solver == "midpoint" and nfe != 1
        if midpoint and nfe % 2:
            raise ValueError("Midpoint needs even NFE, except the one-call Euler case")
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            condition, _, (scale, frames, length) = self.codec.preprocess(coarse.float())
            state, _ = self.start(condition, seed)
            embedded = self.embedding(self.coordinates(condition))
            steps = nfe // 2 if midpoint else nfe
            dt = 1.0 / steps
            for index in range(steps):
                t = torch.full((len(state),), index * dt, device=state.device)
                first = self.velocity(state, embedded, t)
                if midpoint:
                    middle = state + 0.5 * dt * first
                    if self.hparams.geometry == "cylindrical":
                        middle = torch.cat((middle[:, :1].clamp_min(0), middle[:, 1:]), 1)
                    first = self.velocity(middle, embedded, t + 0.5 * dt)
                state = state + dt * first
                if self.hparams.geometry == "cylindrical":
                    phase = state[:, 1:]
                    state = torch.cat((state[:, :1].clamp_min(0), torch.atan2(phase.sin(), phase.cos())), 1)
            if self.hparams.geometry == "cartesian":
                spectrum = torch.complex(state[:, :1], state[:, 1:])
            else:
                spectrum = torch.polar(state[:, :1], state[:, 1:])
            waveform = self.codec.features.invert(spectrum[..., :frames], orig_length=length) * scale
        if not torch.isfinite(waveform).all():
            raise FloatingPointError("Non-finite postfilter reconstruction")
        return waveform

    def training_step(self, audio, batch_idx):
        _, condition, target, info = self.prepare(audio)
        # Separate RNG from model initialisation; both geometries see matched draws.
        seed = self.hparams.seed + self.current_epoch * 10000019 + batch_idx * self.trainer.world_size + self.global_rank
        loss, components = self.flow_loss(condition, target, info[1], seed)
        self.log("train/flow_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=len(audio))
        for index, value in enumerate(components):
            self.log(f"train/coordinate_{index}_mse", value, on_epoch=True, sync_dist=True, batch_size=len(audio))
        return loss

    def validation_step(self, audio, batch_idx):
        coarse, condition, target, info = self.prepare(audio)
        if batch_idx == 0 and self.started is not None:
            elapsed = self.fit_seconds + time.monotonic() - self.started
            self.log("timing/fit_seconds", elapsed, sync_dist=True, batch_size=1)
        seed = 1000000 + self.hparams.seed + batch_idx * self.trainer.world_size + self.global_rank
        loss, _ = self.flow_loss(condition, target, info[1], seed)
        self.log("val/flow_loss", loss, sync_dist=True, batch_size=len(audio))
        if batch_idx >= self.hparams.config["metric_batches"]:
            return
        self.log("val/ndac_spectral", spectral_loss(audio, coarse).mean(), sync_dist=True, batch_size=len(audio))
        for nfe in self.hparams.config["validation_nfe"]:
            output = self.enhance(coarse, nfe, seed, self.hparams.config["solver"])
            distortion = spectral_loss(audio, output).mean()
            self.log(f"val/spectral_nfe{nfe}", distortion, sync_dist=True, batch_size=len(audio))
            self.log(f"val/si_sdr_nfe{nfe}", si_sdr(audio, output).mean(), sync_dist=True, batch_size=len(audio))
            if nfe == self.hparams.config["checkpoint_nfe"]:
                self.log("val/spectral", distortion, sync_dist=True, batch_size=len(audio))
            if batch_idx == 0 and not self.trainer.sanity_checking:
                self.log_audio(audio, coarse, output, "val", nfe)

    def log_audio(self, original, coarse, output, stage, nfe):
        if self.global_rank != 0:
            return
        import wandb
        directory = Path(self.trainer.default_root_dir) / "audio" / f"{stage}_{self.global_step:08d}"
        directory.mkdir(parents=True, exist_ok=True)
        examples = {}
        for label, value in (("original", original), ("ndac", coarse), (f"nfe{nfe}", output)):
            waveform = value[0, 0].float().cpu().numpy()
            sf.write(directory / f"{label}.wav", waveform, SAMPLE_RATE, subtype="FLOAT")
            examples[f"audio/{stage}/{label}"] = wandb.Audio(waveform, sample_rate=SAMPLE_RATE)
        if self.logger:
            self.logger.experiment.log({**examples, "trainer/global_step": self.global_step})

    def test_step(self, audio, batch_idx):
        if audio.shape[-1] % BLOCK_SAMPLES:
            raise ValueError("Test audio must contain whole two-second blocks")
        coarse_blocks = [self.prepare(block)[0] for block in audio.split(BLOCK_SAMPLES, dim=-1)]
        coarse = torch.cat(coarse_blocks, dim=-1)
        self.log("test/ndac_spectral", spectral_loss(audio, coarse).mean(), sync_dist=True, batch_size=len(audio))
        for nfe in self.hparams.config["evaluation_nfe"]:
            seed = 2000000 + self.hparams.seed + batch_idx * self.trainer.world_size + self.global_rank
            output = torch.cat([self.enhance(block, nfe, seed + index * 10007, self.hparams.config["solver"])
                                for index, block in enumerate(coarse_blocks)], dim=-1)
            self.log(f"test/spectral_nfe{nfe}", spectral_loss(audio, output).mean(), sync_dist=True, batch_size=len(audio))
            self.log(f"test/si_sdr_nfe{nfe}", si_sdr(audio, output).mean(), sync_dist=True, batch_size=len(audio))
            if batch_idx == 0:
                self.log_audio(audio, coarse, output, "test", nfe)

    def on_train_epoch_end(self):
        peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
        self.log("gpu/peak_allocated_gb", peak, sync_dist=True, reduce_fx="max", batch_size=1)
        if self.hparams.config.get("smoke"):
            print(f"Smoke rank {self.global_rank}: peak allocated {peak:.2f} GiB", flush=True)

    def on_test_epoch_end(self):
        peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
        self.log("test/gpu_peak_allocated_gb", peak, sync_dist=True, reduce_fx="max", batch_size=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.network.parameters(), lr=self.hparams.config["lr"], weight_decay=0)

    def on_fit_start(self):
        self.started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(self.device)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["source_codec"] = self.codec.identity
        checkpoint["fit_seconds"] = self.fit_seconds + (time.monotonic() - self.started if self.started else 0)

    def on_load_checkpoint(self, checkpoint):
        if checkpoint["source_codec"] != self.codec.identity:
            raise ValueError("NDAC or source feature/noise checkpoint changed")
        if checkpoint["hyper_parameters"].get("provenance") != self.hparams.provenance:
            raise ValueError("Training data or experiment settings changed while resuming")
        self.fit_seconds = checkpoint.get("fit_seconds", 0.0)
