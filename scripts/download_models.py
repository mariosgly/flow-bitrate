"""Fetch the official archive, or unpack one you already downloaded."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/facebookresearch/FlowDec/releases/download/checkpoints/checkpoints.zip"
FILES = ["ndac/ndac-75/800k/dac/weights.pth", "flowdec/flowdec_75m/step=800000.ckpt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = ROOT / "checkpoints"
    root.mkdir(exist_ok=True)
    archive = args.archive or root / "checkpoints.zip"
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        print("Downloading official checkpoints (large archive)...", flush=True)
        partial = archive.with_suffix(".zip.part")
        urllib.request.urlretrieve(URL, partial)
        partial.replace(archive)
    checksums = {}
    with zipfile.ZipFile(archive) as source:
        for name in FILES:
            matches = [item for item in source.namelist() if item == name or item.endswith("/" + name)]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name} in archive")
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            with source.open(matches[0]) as inp, partial.open("wb") as out:
                shutil.copyfileobj(inp, out)
            partial.replace(target)
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksums[name] = digest.hexdigest()
            print(target)
    (root / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


if __name__ == "__main__":
    main()
