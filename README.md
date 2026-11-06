# Cylindrical flow matching for audio reconstruction

Final project for [Learning to Match Distributions](https://brightspace.nyu.edu/d2l/home/628003).

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

## Train the postfilters

Same pretrained NDAC, same CyFM U-Net, Cartesian or cylindrical flow matching. NDAC stays frozen; both postfilters train from scratch at 4.5 kbps.

Jamendo metadata is configured at your existing `inversion/datasets/jamendo/raw.tsv`. From the repository folder on HPC:

```bash
mkdir -p /scratch/mg8253/flow-bitrate/logs
```

Paths are in `configs/postfilter_data.json`. Prepare and check the data first:

```bash
sbatch slurm/postfilter_data.slurm
```

Rerun that job after a timeout; completed decode checks are cached. Once it prints `Ready`, check a tiny full training cycle first:

```bash
sbatch slurm/postfilter_smoke.slurm
# Optional: check the Cartesian arm too.
sbatch slurm/postfilter_smoke.slurm --geometry cartesian
```

Each smoke job requests two GPUs for ten minutes: two tiny epochs, validation, best/last checkpoints, checkpoint reload and testing. It keeps the full model and training batch size. Look for `SMOKE PASSED`, `smoke_success.json`, W&B metrics/audio and `smoke/passed = 1`.

Then submit full training:

```bash
sbatch slurm/postfilter_train.slurm --geometry cartesian --seed 0
sbatch slurm/postfilter_train.slurm --geometry cylindrical --seed 0
```

Each fresh submission creates a new run directory and W&B ID, even with the same `--name`. To continue a run, add `--resume /scratch/mg8253/flow-bitrate/runs/RUN_NAME/last.ckpt`. Slurm requeues resume automatically. Settings are in `configs/postfilter.json`. [Data and splits](docs/data.md).

Compare the saved models on held-out audio, including released FlowDec. Replace the example run names with the directories printed by your jobs:

```bash
sbatch slurm/postfilter_evaluate.slurm --released --checkpoints \
  /scratch/mg8253/flow-bitrate/runs/cartesian-k6-seed0-JOBID/best.ckpt \
  /scratch/mg8253/flow-bitrate/runs/cylindrical-k6-seed0-JOBID/best.ckpt
```

Plots, metrics and audio go to `/scratch/mg8253/flow-bitrate/runs/postfilter-comparison/`. [Experiment details](docs/postfilter.md).

## Original oracle study

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
