"""Run this on a compute node before committing to cache preparation."""

import argparse

import torch

from flow_bitrate.codec import BLOCK_SAMPLES, CODEBOOKS, ROOT, SAMPLE_RATE, FlowCodec
from flow_bitrate.data import load_config, read_audio, read_manifest
from flow_bitrate.metrics import si_sdr, spectral_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/diagnostic.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    row = read_manifest(config["manifest"])[0]
    start = round(float(row.get("start_seconds") or 0) * SAMPLE_RATE)
    original = read_audio(row["path"])[None, ..., start:start + BLOCK_SAMPLES].to(args.device)
    if original.shape[-1] != BLOCK_SAMPLES:
        raise ValueError("The first manifest recording must be at least two seconds")
    codec = FlowCodec(config["checkpoints"], config.get("flow_checkpoint")).to(args.device)
    print(f"Preflight: encoder strides={codec.ndac.encoder_rates}, decoder strides={codec.ndac.decoder_rates}",
          flush=True)
    with torch.no_grad():
        _, codes = codec.encode(original)
        assert codes.shape == (1, 10, 150)
        for rate, depth in enumerate(CODEBOOKS):
            coarse, _ = codec.coarse(codes, depth)
            assert coarse.shape == original.shape, (original.shape, coarse.shape)
            coarse_loss = spectral_loss(original, coarse)
            assert torch.isfinite(coarse_loss).all() and torch.isfinite(si_sdr(original, coarse)).all()
            mixed = codec.coarse_at_rates(codes, torch.tensor([rate], device=args.device))
            torch.testing.assert_close(mixed, coarse)
            if not args.inference_only:
                probes = codec.probes(original, coarse, seed=1)
                assert torch.isfinite(probes).all()
            decoded = codec.enhance(coarse, seed=1, steps=config.get("steps", 3))
            assert decoded.shape == original.shape, (original.shape, decoded.shape)
            loss = spectral_loss(original, decoded)
            assert torch.isfinite(loss).all() and torch.isfinite(si_sdr(original, decoded)).all()
            print(f"Preflight {depth} codebooks: shape={tuple(decoded.shape)}, "
                  f"NDAC loss={coarse_loss.item():.4f}, flow loss={loss.item():.4f}", flush=True)
    if args.inference_only:
        print("Released checkpoint inference preflight passed.", flush=True)
        return
    codec.train()
    loss = codec.flow_loss(original, coarse, seed=1)[0].mean()
    loss.backward()
    trainable = [p for p in codec.backbone.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    assert all(p.grad is None for p in codec.ndac.parameters())
    print("Released checkpoint inference and flow backward finished.")


if __name__ == "__main__":
    main()
