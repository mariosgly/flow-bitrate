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
weights and [audited postfilter manifest](docs/data.md) available, smoke-test the
two EMA arms:

```bash
mkdir -p /scratch/mg8253/flow-bitrate/logs
sbatch --ntasks-per-node=2 --gres=gpu:2 --time=00:30:00 --job-name=jamendo-smoke-flowdec \
  slurm/postfilter_train.slurm --config configs/jamendo.json --smoke \
  --backbone flowdec --geometry cartesian --ema --batch-size 4 --accumulate 8
sbatch --ntasks-per-node=2 --gres=gpu:2 --time=00:30:00 --job-name=jamendo-smoke-cylindrical \
  slurm/postfilter_train.slurm --config configs/jamendo.json --smoke \
  --backbone cyfm --geometry cylindrical --ema
```

Each smoke job uses the configured batch size and accumulation for three
optimizer updates, then runs ten-second validation on two excerpts and checks
that `last.ckpt` was written. Look for `SMOKE PASSED` and per-rank peak
allocated GPU memory in the Slurm output. After both pass:

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
The [Jamendo config](configs/jamendo.json) defaults to 8 crops per GPU and 4
batches per optimizer update for CyFM. The launcher uses 4 crops per GPU and 8
batches per update for FlowDec to reduce its peak memory. Both use 2 GPUs and
64 crops per full update. Both the smoke commands and the five-run launcher
accept `--batch-size` and `--accumulate`. For a single batch per update, use
`--batch-size 8 --accumulate 1` for 16 crops per full update on two GPUs.
Use the same values for smoke testing and training. Training uses bf16 mixed
precision and a 40-epoch budget. Each
job has an eight-hour allocation with a ten-minute stop margin. To continue a
run in another allocation, submit the same training command with
`--resume /absolute/path/to/last.ckpt`.

Validation runs every 1,000 optimizer updates on up to 32 fixed ten-second
excerpts. Each excerpt passes through NDAC and the postfilter as a single input.
Each run keeps one rolling `last.ckpt`. It is replaced after scheduled validation
and at a graceful stop if training has reached a newer optimizer update.

For FAD, SI-SDR, fwSSNR and logSpecMSE on the held-out Jamendo split, use the
[resumable scorer](docs/jamendo-scoring.md). For a quick sweep over inference
steps with SI-SDR and the training spectral metric, use the original evaluator:

```bash
sbatch slurm/postfilter_evaluate.slurm --name jamendo-comparison --checkpoints \
  /path/to/flowdec-ema/last.ckpt \
  /path/to/cyfm-cartesian-ema/last.ckpt \
  /path/to/cyfm-cartesian-noema/last.ckpt \
  /path/to/cyfm-cylindrical-ema/last.ckpt \
  /path/to/cyfm-cylindrical-noema/last.ckpt
```

The evaluator uses up to 64 fixed ten-second Jamendo test excerpts and the same
noise seeds, solver and network-evaluation budgets for every model. It reports
spectral error, SI-SDR, postfilter time, audio examples and a comparison plot.

### Disk geometry and phase weighting

The [Jamendo disk experiment guide](docs/jamendo-geometry.md) gives the geometry,
validation diagnostic, training commands and blinded listening protocol. Start with
`bash scripts/submit_jamendo_geometry.sh audit`. The existing Cartesian and
cylindrical CyFM EMA checkpoints serve as the two controls.

### Jamendo comparison at FlowDec's parameter budget

Submit one job per arm from the project root. All five CyFM arms use the Jamendo FlowDec optimizer
(Adam, constant learning rate `1e-4`, default betas and epsilon, zero weight decay),
EMA at `0.999`, four crops per GPU, eight accumulation batches and two GPUs.
They follow the same crop, noise and flow-time draws for a given seed and stop at
20,000 optimizer updates across allocations. Each has its own run directory and W&B ID.

```bash
mkdir -p /scratch/mg8253/flow-bitrate/logs
sbatch --job-name=jamendo-large-disk      slurm/jamendo_matched.slurm disk
sbatch --job-name=jamendo-large-cyl       slurm/jamendo_matched.slurm cylindrical
sbatch --job-name=jamendo-large-warped    slurm/jamendo_matched.slurm warped-disk
sbatch --job-name=jamendo-large-disk-loss slurm/jamendo_matched.slurm disk-loss
sbatch --job-name=jamendo-large-weighted  slurm/jamendo_matched.slurm weighted
```

The CyFM spatial width is 160 and the time-conditioning width is 910. These
widths give 23,703,362 trainable parameters in Cartesian coordinates and
23,706,242 in cylindrical coordinates. The pinned NCSN++ has 23,703,640.
Counts cover the learned postfilter and exclude the frozen codec, EMA copy and
frozen Fourier features. The extra 2,880 cylindrical weights are in the input convolution.

The ordinary Euclidean disk uses the straight Cartesian chord. `disk` and
`cartesian` select exactly the same training recipe. If the matched large Cartesian
and cylindrical jobs are already running, submit only the last three commands.

| Candidate | Matched control | What changes |
| --- | --- | --- |
| `cylindrical` | `disk` | The network learns magnitude and shortest-arc phase velocities. |
| `warped-disk` | `disk` | The training path can cross zero for weak opposite-phase bins and bend around it for strong ones. Both runs use Cartesian inputs, velocities, MSE and sampler. |
| `disk-loss` | `cylindrical` | The cylindrical path stays fixed. Squared phase-velocity error is weighted by squared current magnitude. |
| `weighted` | `cylindrical` | The same phase error receives a bounded weight based on current magnitude and the codec's frequency-dependent noise scale. |

`weighted` reads the audit's `/scratch/mg8253/flow-bitrate/jamendo-geometry/calibration.json`.
If the audit was saved elsewhere, append `--phase-calibration /path/to/calibration.json`
to its command. The other arms can run independently of the audit.

Compare large Cartesian CyFM with Cartesian FlowDec to measure
the effect of backbone choice at matched size and optimizer. Use a saved FlowDec
checkpoint at the same update. If one is unavailable, submit
`sbatch --job-name=jamendo-ref-20k slurm/jamendo_matched.slurm flowdec`.
Prefix any CyFM arm with `small-` for a width-64 control with the same Adam recipe
and update budget, for example `small-warped-disk` or `small-disk-loss`.

For a three-update smoke run, add `--time=00:30:00` before the Slurm filename
and `--smoke` after the arm. To continue an eight-hour allocation, resubmit the
same arm and seed with `--resume /path/to/run/last.ckpt`. Keep both geometries
on the same GPU type and repeat the pair with `--seed 1` for confirmation.

```bash
sbatch slurm/postfilter_evaluate.slurm --name matched-20k-seed0 --cohort-seed 0 \
  --require-update 20000 --nfe 6 24 --checkpoints \
  /path/to/flowdec/last.ckpt /path/to/large-disk/last.ckpt /path/to/large-cyl/last.ckpt \
  /path/to/large-warped/last.ckpt /path/to/large-disk-loss/last.ckpt /path/to/large-weighted/last.ckpt
```

Pass distinct run directories' checkpoint files. The evaluator records parameter
counts, optimizer recipes, steps, fit time and postfilter time. The full-cohort
scoring planner also accepts `--require-update 20000`. For blinded listening,
pass one candidate and its matched control from the table together with `--listening-tracks 8`.

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
