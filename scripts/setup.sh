#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/scratch.sh

# Run this inside the conda env you want to use on the cluster.
python -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Use Python 3.10–3.12"'
if [[ "${1:-}" == "--cuda" ]]; then
  python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
fi
python -m pip install -e '.[test]'
mkdir -p external
if [[ ! -d external/FlowDec/.git ]]; then
  git clone https://github.com/facebookresearch/FlowDec.git external/FlowDec
fi
git -C external/FlowDec checkout 26ec106281f061a720e0399b0edfd3c3c020d7ab
python scripts/patch_flowdec.py
