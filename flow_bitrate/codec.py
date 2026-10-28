"""NDAC-75 + the released FlowDec-75m backbone. Everything here is specific to it."""

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/scratch/mg8253/flow-bitrate")
UPSTREAM = ROOT / "external/FlowDec"
SAMPLE_RATE = 48_000
BLOCK_SAMPLES = 96_000
CODEBOOKS = (4, 6, 8, 10)
RATES = (3.0, 4.5, 6.0, 7.5)
PROBE_TIMES = (0.2, 0.5, 0.8)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FlowCodec(nn.Module):
    def __init__(self, checkpoint_dir, flow_checkpoint=None):
        super().__init__()
        if not (UPSTREAM / "flowdec").is_dir():
            raise FileNotFoundError("Run bash scripts/setup.sh first")
        sys.path.insert(0, str(UPSTREAM))
        from dac import DAC
        from flowdec.backbones.ncsnpp import NCSNpp
        from flowdec.data.feature_extractors import AmplitudeCompressedComplexSTFT
        from omegaconf import OmegaConf

        checkpoint_dir = Path(checkpoint_dir)
        ndac_path = checkpoint_dir / "ndac/ndac-75/800k/dac/weights.pth"
        flow_path = Path(flow_checkpoint) if flow_checkpoint else (
            checkpoint_dir / "flowdec/flowdec_75m/step=800000.ckpt"
        )
        # These are trusted official/local checkpoints, not arbitrary uploaded files.
        ndac_state = torch.load(ndac_path, map_location="cpu", weights_only=False)
        import inspect
        kwargs = ndac_state["metadata"]["kwargs"]
        self.ndac = DAC(**{k: v for k, v in kwargs.items() if k in inspect.signature(DAC).parameters})
        self.ndac.load_state_dict(ndac_state["state_dict"], strict=True)
        config = yaml.safe_load((UPSTREAM / "config/model/backbone/ncsnpp_final_no_attn.yaml").read_text())
        config.pop("_target_")
        self.backbone = NCSNpp(**config)
        self.features = AmplitudeCompressedComplexSTFT(
            window_fn="hann", n_fft=1534, n_hops=4, sampling_rate=SAMPLE_RATE,
            alpha=0.3, beta=0.33,
        )
        # Importing OmegaConf above also registers it for legacy Lightning checkpoints.
        checkpoint = torch.load(flow_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("flow_state", checkpoint.get("_pl_ema_state_dict", checkpoint.get("state_dict")))
        if state is None:
            raise ValueError(f"No FlowDec state in {flow_path}")
        self.backbone.load_state_dict(
            {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}, strict=True
        )
        self.features.load_state_dict(
            {k.removeprefix("feature_extractor."): v for k, v in state.items() if k.startswith("feature_extractor.")},
            strict=True,
        )
        self.register_buffer("sigma", state["sigma_y"].float())
        if tuple(self.sigma.shape) not in {(768, 1), ()}:
            raise ValueError(f"Unexpected sigma shape: {self.sigma.shape}")
        self.identity = {"ndac": file_hash(ndac_path), "flow": file_hash(flow_path)}
        self.ndac.requires_grad_(False)
        self.features.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.ndac.eval()  # no RVQ dropout or changing codebooks during flow fine-tuning
        self.features.eval()
        return self

    @torch.no_grad()
    def encode(self, audio):
        # audio [B, 1, 96000], latent [B, 1024, 150], codes [B, 10, 150]
        audio = self.ndac.preprocess(audio, SAMPLE_RATE)
        latent = self.ndac.encoder(audio)
        _, codes, _, _, _ = self.ndac.quantizer(latent, n_quantizers=10)
        if latent.shape[1] != 1024 or codes.shape[1] != 10:
            raise ValueError("This experiment expects the NDAC-75 checkpoint")
        return latent, codes

    @torch.no_grad()
    def coarse(self, codes, codebooks):
        quantized, _, _ = self.ndac.quantizer.from_codes(codes[:, :codebooks])
        return self.ndac.decode(quantized)[..., :BLOCK_SAMPLES], quantized

    @torch.no_grad()
    def coarse_at_rates(self, codes, rate):
        # Different RVQ depth per batch item; the NDAC decoder still runs once.
        candidates = torch.stack([self.ndac.quantizer.from_codes(codes[:, :k])[0] for k in CODEBOOKS], 1)
        batch, _, channels, frames = candidates.shape
        quantized = candidates.gather(1, rate[:, None, None, None].expand(batch, 1, channels, frames)).squeeze(1)
        return self.ndac.decode(quantized)[..., :BLOCK_SAMPLES]

    def preprocess(self, coarse, original=None):
        # Complex STFTs [B, 1, 768, frames], padded to a multiple of 64 frames.
        scale = coarse.abs().amax(dim=(1, 2), keepdim=True)
        scale = torch.where(torch.isclose(scale, torch.zeros_like(scale)), 1.0, scale)
        condition = self.features(coarse.float() / scale)
        frames = condition.shape[-1]
        padding = (-frames) % 64
        condition = F.pad(condition, (0, padding))
        target = None
        if original is not None:
            target = F.pad(self.features(original.float() / scale), (0, padding))
        return condition, target, (scale, frames, coarse.shape[-1])

    def velocity(self, state, condition, time):
        if time.ndim == 0:
            time = time.expand(state.shape[0])
        return self.backbone(state, condition, time.float())

    def flow_loss(self, original, coarse, seed=None, time=None):
        condition, target, _ = self.preprocess(coarse, original)
        generator = None if seed is None else torch.Generator(device=condition.device).manual_seed(seed)
        noise = torch.randn(condition.shape, dtype=condition.dtype, device=condition.device, generator=generator)
        start = condition + self.sigma * noise
        if time is None:
            time = torch.rand(condition.shape[0], device=condition.device, generator=generator)
        else:
            time = torch.full((condition.shape[0],), time, device=condition.device)
        t = time[:, None, None, None]
        state = (1 - t) * start + t * target
        velocity = self.velocity(state, condition, time)
        expected = target - start  # shifted-prior JFM target, not original minus plain noise
        error = (velocity - expected).abs().square().flatten(1).mean(1)
        power = expected.abs().square().flatten(1).mean(1)
        speed = velocity.abs().square().flatten(1).mean(1)
        return error, power, speed

    @torch.no_grad()
    def probes(self, original, coarse, seed):
        condition, target, _ = self.preprocess(coarse, original)
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        noise = torch.randn(condition.shape, dtype=condition.dtype, device=condition.device, generator=generator)
        start = condition + self.sigma * noise
        expected = target - start
        power = expected.abs().square().flatten(1).mean(1).clamp_min(1e-8)
        measurements = []
        for time in PROBE_TIMES:
            state = (1 - time) * start + time * target
            velocity = self.velocity(state, condition, torch.tensor(time, device=state.device))
            error = (velocity - expected).abs().square().flatten(1).mean(1)
            speed = velocity.abs().square().flatten(1).mean(1)
            measurements.extend([error.log1p(), (error / power.clamp_min(1e-8)).log1p(), speed.log1p()])
            if time == 0.5:
                direction = torch.randn(state.shape, dtype=state.dtype, device=state.device, generator=generator)
                direction /= direction.abs().square().flatten(1).mean(1).sqrt()[:, None, None, None].clamp_min(1e-8)
                epsilon = 0.01 * state.abs().square().flatten(1).mean(1).sqrt().clamp_min(1e-3)
                step = epsilon[:, None, None, None] * direction
                t = torch.tensor(time, device=state.device)
                difference = (self.velocity(state + step, condition, t) - self.velocity(state - step, condition, t))
                sensitivity = (difference / (2 * epsilon[:, None, None, None])).abs().square().flatten(1).mean(1)
        # Last feature is a directional finite difference, not a Lipschitz bound.
        measurements.append(sensitivity.log1p())
        return torch.stack(measurements, dim=-1)  # [B, 10], five backbone calls per rate

    @torch.no_grad()
    def enhance(self, coarse, seed=0, steps=3):
        condition, _, (scale, frames, length) = self.preprocess(coarse)
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        noise = torch.randn(condition.shape, dtype=condition.dtype, device=condition.device, generator=generator)
        state = condition + self.sigma * noise
        # Same fixed-step midpoint rule as the paper, without torchdyn's import tree.
        dt = 1.0 / steps
        for index in range(steps):
            t = torch.tensor(index * dt, device=state.device)
            first = self.velocity(state, condition, t)
            state = state + dt * self.velocity(state + 0.5 * dt * first, condition, t + 0.5 * dt)
        waveform = self.features.invert(state[..., :frames], orig_length=length) * scale
        if not torch.isfinite(waveform).all():
            raise FloatingPointError("Non-finite FlowDec reconstruction")
        return waveform

    def export_flow(self):
        state = {"backbone." + k: v.detach().cpu() for k, v in self.backbone.state_dict().items()}
        state.update({"feature_extractor." + k: v.detach().cpu() for k, v in self.features.state_dict().items()})
        state["sigma_y"] = self.sigma.detach().cpu()
        return state
