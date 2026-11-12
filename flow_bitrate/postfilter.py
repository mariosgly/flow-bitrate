"""Frozen NDAC, with the same conditional U-Net trained in either geometry."""

import copy
import time
from pathlib import Path

import pytorch_lightning as pl
import soundfile as sf
import torch

from .codec import BLOCK_SAMPLES, CODEBOOKS, SAMPLE_RATE, FlowCodec, flowdec_backbone
from .cyfm_unet import CylindricalUNet
from .data import matched_seed
from .metrics import si_sdr, spectral_loss


class Postfilter(pl.LightningModule):
    def __init__(self, config, geometry, seed=0, provenance=None):
        super().__init__()
        if geometry not in {"cartesian", "cylindrical"}:
            raise ValueError("Choose cartesian or cylindrical")
        self.save_hyperparameters()
        self.jamendo = config.get("dataset") == "jamendo"
        self.backbone_name = config.get("backbone", "cyfm")
        self.codec = FlowCodec(config["checkpoints"], load_backbone=False, configured_noise=self.jamendo)
        self.codec.requires_grad_(False)
        channels = 2 if geometry == "cartesian" else 3
        # State + coarse condition: [B, 4 or 6, 768, padded_frames].
        if self.backbone_name == "flowdec":
            if geometry != "cartesian":
                raise ValueError("The original FlowDec backbone requires Cartesian inputs")
            self.network = flowdec_backbone()
        else:
            self.network = CylindricalUNet(base_channels=config["width"], in_channels=2 * channels,
                                           out_channels=2, velocity_bound=None)
        self.ema_network = copy.deepcopy(self.network).requires_grad_(False) if config.get("ema", False) else None
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long) if self.ema_network is not None else None)
        if not 0 < config.get("ema_decay", 0.999) < 1:
            raise ValueError("ema_decay must be between zero and one")
        self.rate_index = CODEBOOKS.index(config["codebooks"])
        self.fit_seconds = 0.0
        self.started = None

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        if self.ema_network is not None:
            self.ema_network.eval()
        return self

    @torch.no_grad()
    def coarse_audio(self, audio):
        # Keep the frozen codec and complex STFT in float32 under mixed precision.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            samples = audio.shape[-1] if self.jamendo else BLOCK_SAMPLES
            _, codes = self.codec.encode(audio.float(), samples)
            coarse, _ = self.codec.coarse(codes, CODEBOOKS[self.rate_index], samples)
        return coarse

    @torch.no_grad()
    def prepare(self, audio):
        coarse = self.coarse_audio(audio)
        with torch.autocast(device_type=audio.device.type, enabled=False):
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
        network = self.ema_network if self.ema_network is not None and not self.training else self.network
        if self.backbone_name == "flowdec":
            state = torch.complex(state[:, :1], state[:, 1:])
            condition = torch.complex(condition[:, :1], condition[:, 1:])
            return self.coordinates(network(state, condition, time_value))
        value = torch.cat((self.embedding(state), condition), dim=1)
        return network(value, time_value).float()  # [B, 2, F, T]

    def start(self, condition, seed=None):
        generator = None if seed is None else torch.Generator(device=condition.device).manual_seed(seed)
        noise = torch.randn(condition.shape, dtype=condition.dtype, device=condition.device, generator=generator)
        return self.coordinates(condition + self.codec.sigma * noise), generator

    def matched_start(self, condition, seeds):
        # CPU draws match even when jobs use different CUDA devices or batch shapes.
        noise, times = [], []
        for seed in seeds:
            generator = torch.Generator().manual_seed(max(seed, 0))
            noise.append(torch.randn(condition.shape[1:], dtype=condition.dtype, generator=generator))
            times.append(torch.rand((), generator=generator))
        noise = torch.stack(noise).to(condition.device)
        return self.coordinates(condition + self.codec.sigma * noise), torch.stack(times).to(condition.device)

    def velocity_errors(self, condition, target, frames, seed=None, example_seeds=None):
        if example_seeds is None:
            start, generator = self.start(condition, seed)
            time_value = torch.rand(len(start), device=start.device, generator=generator)
        else:
            start, time_value = self.matched_start(condition, example_seeds)
        end = self.coordinates(target)
        velocity = end - start
        if self.hparams.geometry == "cylindrical":
            difference = velocity[:, 1:]
            velocity = torch.cat((velocity[:, :1], torch.atan2(difference.sin(), difference.cos())), 1)
        state = start + time_value[:, None, None, None] * velocity
        prediction = self.velocity(state, self.embedding(self.coordinates(condition)), time_value)
        # Match the original FlowDec reduction in all five Jamendo runs.
        if not self.jamendo:
            prediction, velocity = prediction[..., :frames], velocity[..., :frames]
        return (prediction - velocity).square().mean((2, 3))

    def flow_loss(self, condition, target, frames, seed=None):
        errors = self.velocity_errors(condition, target, frames, seed)
        return errors.sum(1).mean(), errors.mean(0)

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
            if self.jamendo:
                state, _ = self.matched_start(condition, [seed] * len(coarse))
            else:
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
        if self.jamendo:
            audio, seeds = audio
            _, condition, target, info = self.prepare(audio)
            errors = self.velocity_errors(condition, target, info[1], example_seeds=seeds.tolist())
            real = seeds >= 0
            count = self.trainer.strategy.reduce(real.sum(), reduce_op="sum")
            # DDP averages gradients across ranks. Divide by the real global batch size,
            # including when a final batch needs empty slots to keep the ranks in sync.
            components = errors.masked_fill(~real[:, None], 0).sum(0) * (self.trainer.world_size / count)
            loss = components.sum()
            self.log("train/flow_loss", loss, on_step=True, on_epoch=False, sync_dist=True, batch_size=1)
            return loss
        _, condition, target, info = self.prepare(audio)
        # Separate RNG from model initialisation; both geometries see matched draws.
        seed = self.hparams.seed + self.current_epoch * 10000019 + batch_idx * self.trainer.world_size + self.global_rank
        loss, components = self.flow_loss(condition, target, info[1], seed)
        self.log("train/flow_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=len(audio))
        for index, value in enumerate(components):
            self.log(f"train/coordinate_{index}_mse", value, on_epoch=True, sync_dist=True, batch_size=len(audio))
        return loss

    def validation_step(self, audio, batch_idx):
        if self.jamendo:
            self.jamendo_validation_step(audio, batch_idx)
            return
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

    def on_validation_epoch_start(self):
        if self.jamendo:
            self.validation_totals = torch.zeros(
                (1 + len(self.hparams.config["validation_nfe"]), 2), device=self.device, dtype=torch.float64
            )

    def jamendo_validation_step(self, batch, batch_idx):
        config = self.hparams.config
        audio, tracks = batch
        track = tracks[0]
        if not track or self.global_step % config["validate_every"]:
            return
        coarse = self.coarse_audio(audio)
        self.validation_totals[0, 0] += spectral_loss(audio, coarse).sum()
        self.validation_totals[0, 1] += 1
        for index, nfe in enumerate(config["validation_nfe"], start=1):
            for sample_seed in config["evaluation_seeds"]:
                seed = matched_seed("evaluation", track, sample_seed)
                output = self.enhance(coarse, nfe, seed, config["solver"])
                self.validation_totals[index, 0] += spectral_loss(audio, output).sum()
                self.validation_totals[index, 1] += si_sdr(audio, output).sum()
                if batch_idx == 0 and nfe == config["checkpoint_nfe"] and sample_seed == config["evaluation_seeds"][0]:
                    self.log_audio(audio, coarse, output, "val", nfe)

    def on_validation_epoch_end(self):
        config = self.hparams.config
        if not self.jamendo or self.global_step % config["validate_every"]:
            return
        totals = self.trainer.strategy.reduce(self.validation_totals, reduce_op="sum")
        self.log("val/ndac_spectral", totals[0, 0] / totals[0, 1])
        count = totals[0, 1] * len(config["evaluation_seeds"])
        for index, nfe in enumerate(config["validation_nfe"], start=1):
            spectral, sdr = totals[index] / count
            self.log(f"val/spectral_nfe{nfe}", spectral)
            self.log(f"val/si_sdr_nfe{nfe}", sdr)
            if nfe == config["checkpoint_nfe"]:
                self.log("val/spectral", spectral)

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
        if self.ema_updates is not None:
            self.log("ema/updates", self.ema_updates.float(), sync_dist=True, batch_size=1)
        if self.hparams.config.get("smoke"):
            print(f"Smoke rank {self.global_rank}: peak allocated {peak:.2f} GiB", flush=True)

    def on_test_epoch_end(self):
        peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
        self.log("test/gpu_peak_allocated_gb", peak, sync_dist=True, reduce_fx="max", batch_size=1)

    def configure_optimizers(self):
        if self.backbone_name == "flowdec":
            return torch.optim.Adam(self.network.parameters(), lr=1e-4)
        if self.jamendo:
            optimizer = torch.optim.AdamW(self.network.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.config["max_epochs"], eta_min=1e-6
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        return torch.optim.AdamW(self.network.parameters(), lr=self.hparams.config["lr"], weight_decay=0)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema_network is not None:
            # Update after the optimizer and before checkpoint callbacks run.
            decay = self.hparams.config.get("ema_decay", 0.999)
            with torch.no_grad():
                for averaged, current in zip(self.ema_network.parameters(), self.network.parameters()):
                    averaged.lerp_(current.detach(), 1 - decay)
                for averaged, current in zip(self.ema_network.buffers(), self.network.buffers()):
                    averaged.copy_(current)
                self.ema_updates.add_(1)

    def on_fit_start(self):
        self.started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(self.device)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["source_codec"] = self.codec.identity
        checkpoint["evaluation_weights"] = "ema" if self.ema_network is not None else "raw"
        checkpoint["fit_seconds"] = self.fit_seconds + (time.monotonic() - self.started if self.started else 0)

    def on_load_checkpoint(self, checkpoint):
        if checkpoint["source_codec"] != self.codec.identity:
            raise ValueError("NDAC or source feature/noise checkpoint changed")
        if checkpoint["hyper_parameters"].get("provenance") != self.hparams.provenance:
            raise ValueError("Training data or experiment settings changed while resuming")
        self.fit_seconds = checkpoint.get("fit_seconds", 0.0)
