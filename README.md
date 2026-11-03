# Flow-guided bitrate allocation for audio codecs

Final project for **Learning to Match Distributions**.

Aramis Tanelus (at4219) and Marios Glytsos (mg8253).

## Installation

Assuming CUDA and access to the audio datasets are already set up:

```bash
conda create -n flow-bitrate python=3.10 pip git -y
conda activate flow-bitrate
bash scripts/setup.sh --cuda
python scripts/download_models.py
wandb login
```

## First study

Compare different bitrates and flow decoding steps using pretrained FlowDec, then check whether changing the bitrate across audio blocks can reduce reconstruction error under a fixed budget.

Set the audio paths and recording offsets in the manifest:

```bash
mkdir -p /scratch/mg8253/flow-bitrate/manifests
cp -n data/diagnostic.example.csv /scratch/mg8253/flow-bitrate/manifests/diagnostic.csv
```

Paths and experiment settings are in `configs/diagnostic.json`. Run on one GPU:

```bash
source scripts/scratch.sh
python -m flow_bitrate.sweep --config configs/diagnostic.json --device cuda
python -m flow_bitrate.oracle --config configs/diagnostic.json
```
