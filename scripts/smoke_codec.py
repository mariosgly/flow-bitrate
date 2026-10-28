"""Run this on a compute node before committing to cache preparation."""

import argparse

import torch

from flow_bitrate.codec import BLOCK_SAMPLES, CODEBOOKS, ROOT, FlowCodec
from flow_bitrate.data import load_config, read_audio, read_manifest
from flow_bitrate.metrics import spectral_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/diagnostic.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    path = read_manifest(config["manifest"])[0]["path"]
    original = read_audio(path)[None, ..., :BLOCK_SAMPLES].to(args.device)
    if original.shape[-1] != BLOCK_SAMPLES:
        raise ValueError("The first manifest recording must be at least two seconds")
    codec = FlowCodec(config["checkpoints"], config.get("flow_checkpoint")).to(args.device)
    with torch.no_grad():
        _, codes = codec.encode(original)
        assert codes.shape == (1, 10, 150)
        for depth in CODEBOOKS:
            coarse, _ = codec.coarse(codes, depth)
            probes = codec.probes(original, coarse, seed=1)
            decoded = codec.enhance(coarse, seed=1, steps=config.get("steps", 3))
            assert decoded.shape == original.shape and torch.isfinite(probes).all()
            print(f"{depth} codebooks: loss={spectral_loss(original, decoded).item():.4f}", flush=True)
    codec.train()
    loss = codec.flow_loss(original, coarse, seed=1)[0].mean()
    loss.backward()
    trainable = [p for p in codec.backbone.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    assert all(p.grad is None for p in codec.ndac.parameters())
    print("Released checkpoint inference and flow backward finished.")


if __name__ == "__main__":
    main()
