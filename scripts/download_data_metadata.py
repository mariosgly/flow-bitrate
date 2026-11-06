"""Fetch Jamendo artist metadata into scratch. Run on a network-enabled login node."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/scratch/mg8253/flow-bitrate")
URL = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw.tsv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/postfilter_data.json")
    args = parser.parse_args()
    destination = Path(json.loads(args.config.read_text())["jamendo_metadata"]).resolve()
    if destination.exists():
        print(f"Already present: {destination}")
    else:
        if not destination.is_relative_to(SCRATCH.resolve()):
            raise ValueError(f"Download metadata under {SCRATCH}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tsv.part")
        with urlopen(URL, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        with temporary.open() as handle:
            if not handle.readline().startswith("TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH"):
                raise ValueError("Download is not Jamendo track/artist metadata")
        temporary.replace(destination)
        print(f"Saved {destination}")
