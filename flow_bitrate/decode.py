"""Decode .fbr without access to the original audio, features, or allocator."""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from .allocation import unpack_codes
from .codec import BLOCK_SAMPLES, CODEBOOKS, ROOT, SAMPLE_RATE, SCRATCH, FlowCodec
from .data import load_config
from .prepare import block_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", default=ROOT / "configs/experiment.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    output = (Path(config["output"]) / args.output).resolve()
    if not output.is_relative_to(SCRATCH.resolve()):
        raise ValueError(f"Decoded audio must be written under {SCRATCH}")
    output.parent.mkdir(parents=True, exist_ok=True)
    codes, choices, seed = unpack_codes(args.input.read_bytes())
    codec = FlowCodec(config["checkpoints"], config.get("flow_checkpoint")).to(args.device).requires_grad_(False)
    audio = []
    with torch.inference_mode():
        for index, (code, choice) in enumerate(zip(codes, choices)):
            code = torch.from_numpy(code)[None].long().to(args.device)
            coarse = codec.coarse(code, CODEBOOKS[choice])[0]
            decoded = codec.enhance(coarse, block_seed("", index * BLOCK_SAMPLES, seed), config["steps"])
            audio.append(decoded[0, 0].cpu())
    sf.write(output, torch.cat(audio).numpy(), SAMPLE_RATE, subtype="FLOAT")


if __name__ == "__main__":
    main()
