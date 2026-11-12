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

## Jamendo postfilter comparison

From the project root on HPC, with the pinned FlowDec dependency, frozen NDAC-75
weights and [audited postfilter manifest](docs/data.md) available:

```bash
bash scripts/submit_jamendo_runs.sh
```

The launcher submits five independent runs at a fixed nominal 4.5 kbps (six
codebooks). They train the original FlowDec NCSN++ with EMA and the CyFM U-Net in
Cartesian and cylindrical coordinates, each with and without EMA. All five
postfilters start from random weights. Only the shared frozen NDAC uses pretrained
weights. The noise curve comes from the upstream FlowDec-75m training config.

The trainer reads only Jamendo rows from the audited, artist-disjoint splits.
Each complete epoch uses one random two-second crop per training track. The track
order, crop offset, complex noise and flow time match across all arms. The last
distributed batch can contain empty slots to keep the ranks aligned. These slots
contribute no loss. All five use FlowDec's reduction over the padded STFT grid.

FlowDec uses Adam at a constant `1e-4`. CyFM uses AdamW at `2e-4` with weight
decay `1e-4` and epoch-wise cosine annealing to `1e-6`. EMA decay is `0.999`.
The [Jamendo config](configs/jamendo.json) sets 32 crops per GPU, 2 GPUs via the
launcher (64 crops per full optimizer update), bf16 mixed precision and a
40-epoch budget. Each job has an eight-hour allocation with a ten-minute stop
margin. To continue a run in another allocation,
submit the same training command with `--resume /absolute/path/to/last.ckpt`.

Validation runs every 1,000 optimizer updates on up to 32 fixed ten-second
excerpts. Each excerpt passes through NDAC and the postfilter as a single input.
`last.ckpt` records the latest scheduled validation and `best.ckpt` follows mean
spectral error at six network evaluations. Work after the latest validation
remains unsaved if the allocation ends between validations.

Pass the five resulting checkpoints to the shared evaluator:

```bash
sbatch slurm/postfilter_evaluate.slurm --name jamendo-comparison --checkpoints \
  /path/to/flowdec-ema/best.ckpt \
  /path/to/cyfm-cartesian-ema/best.ckpt \
  /path/to/cyfm-cartesian-noema/best.ckpt \
  /path/to/cyfm-cylindrical-ema/best.ckpt \
  /path/to/cyfm-cylindrical-noema/best.ckpt
```

The evaluator uses up to 64 fixed ten-second Jamendo test excerpts and the same
noise seeds, solver and network-evaluation budgets for every model. It reports
spectral error, SI-SDR, postfilter time, audio examples and a comparison plot.

## Train the mixed-dataset postfilters

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
# Check the EMA path before submitting the long runs.
sbatch slurm/postfilter_smoke.slurm --ema
```

Each smoke job requests two GPUs for ten minutes: two tiny epochs, validation, best/last checkpoints, checkpoint reload and testing. It keeps the full model and training batch size. Look for `SMOKE PASSED`, `smoke_success.json`, W&B metrics/audio and `smoke/passed = 1`.

Then submit all four runs: Cartesian/cylindrical, each with and without EMA. Each requests four GPUs for 24 hours:

```bash
bash scripts/submit_postfilter_runs.sh
```

Each run uses 64 crops per GPU (256 total), no accumulation, LR `5e-5`, and validation every 1,000 updates. Training stops by wall time, with no step/epoch cap and ten minutes reserved for saving. Startup counts against the allocation. All four simultaneous jobs use 16 GPUs total; Slurm queues them according to availability. EMA decay is `0.999`; validation/testing use averaged weights in the EMA runs. CFG is disabled in all runs.

Each fresh submission creates a new run directory and W&B ID, even with the same `--name`. To continue a run in another allocation, add `--resume /scratch/mg8253/flow-bitrate/runs/RUN_NAME/last.ckpt` to its original training command, keeping the same geometry and `--ema`/`--no-ema` flag. Jobs do not automatically requeue at the time limit. Settings are in `configs/postfilter.json`. [Data and splits](docs/data.md).

Compare the saved models on held-out audio, including released FlowDec. Replace the example run names with the directories printed by your jobs:

```bash
sbatch slurm/postfilter_evaluate.slurm --released --checkpoints \
  /scratch/mg8253/flow-bitrate/runs/cartesian-noema-k6-seed0-JOBID/best.ckpt \
  /scratch/mg8253/flow-bitrate/runs/cartesian-ema-k6-seed0-JOBID/best.ckpt \
  /scratch/mg8253/flow-bitrate/runs/cylindrical-noema-k6-seed0-JOBID/best.ckpt \
  /scratch/mg8253/flow-bitrate/runs/cylindrical-ema-k6-seed0-JOBID/best.ckpt
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
