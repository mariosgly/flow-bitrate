"""Small import fixes for the pinned upstream source. No weight changes."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/FlowDec"
REVISION = "26ec106281f061a720e0399b0edfd3c3c020d7ab"


def main():
    revision = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected FlowDec {REVISION}, got {revision}")

    # The package imports every optional evaluator before we can import its backbone.
    path = UPSTREAM / "flowdec/__init__.py"
    text = path.read_text()
    start = text.find("from . import backbones")
    if start >= 0:
        path.write_text(text[:start] + "# Submodules are imported explicitly by flow-bitrate.\n")

    # Upstream already has native Torch implementations. Use them on CUDA too;
    # this avoids nvcc/compiler requirements and preserves autograd.
    for filename, variable in [("upfirdn2d.py", "upfirdn2d_op"), ("fused_act.py", "fused")]:
        path = UPSTREAM / "flowdec/backbones/ncsnpp_utils/op" / filename
        text = path.read_text()
        start = text.find(f"{variable} = load(")
        if start >= 0:
            end = text.index("\n)\n", start) + len("\n)\n")
            text = text[:start] + f"{variable} = None\n" + text[end:]
        text = text.replace('if input.device.type == "cpu":', "if True:  # native Torch on CPU and CUDA")
        path.write_text(text)

    # view_as_complex does not accept bfloat16 after autocast.
    path = UPSTREAM / "flowdec/backbones/ncsnpp.py"
    text = path.read_text().replace("torch.view_as_complex(h)", "torch.view_as_complex(h.float())")
    path.write_text(text)
    print(f"FlowDec {revision}: native Torch ops, lazy imports, float32 complex output")


if __name__ == "__main__":
    main()
