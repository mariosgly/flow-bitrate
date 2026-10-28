# Flow bitrate

First ask whether [FlowDec](https://github.com/facebookresearch/FlowDec) needs more bits or more decoding computation. Run the frozen-model grid and oracle **before training anything**.

## Setup

Use the CUDA 12.6.3 SIF with `/scratch/mg8253/images/flow-matching-50GB.ext3`. After installing Miniforge and `/ext3/env.sh` as discussed, enter the overlay writable. Inside that container, from the repository folder:

```bash
source /ext3/env.sh
source scripts/scratch.sh
conda create --prefix /ext3/miniforge3/envs/flow-bitrate python=3.10 pip git -y
conda activate /ext3/miniforge3/envs/flow-bitrate
bash scripts/setup.sh --cuda
python scripts/download_models.py  # reads configs/diagnostic.json; optional --config / --archive
wandb login
exit
```

Exit the writable container before submitting jobs. Jobs mount the overlay read-only, source `/ext3/env.sh`, and activate that exact environment prefix. Copy source files to HPC, not the Mac `.venv`. The editable installation uses the HPC repository path; reinstall if you move it afterward.

Everything generated lives under `/scratch/mg8253/flow-bitrate`: pretrained downloads in `models/`, training checkpoints and reports in `runs/`, measurements/audio in `cache/`, W&B in `wandb/`, Slurm output in `logs/`, and test temporary files in `tmp/`. Configured artifact paths must stay under that project folder. The repository location is independent.

For the first run, follow [data/README.md](data/README.md) and copy the manifest template:

```bash
mkdir -p /scratch/mg8253/flow-bitrate/manifests
cp -n data/diagnostic.example.csv /scratch/mg8253/flow-bitrate/manifests/diagnostic.csv
```

Fill in its audio paths. Both configs already use the project scratch folder. Relative paths in those configs resolve there, not into the repository. For interactive commands, source `/ext3/env.sh` followed by `scripts/scratch.sh`, then activate the environment, just as the jobs do.

## First experiment — HPC

Submit from the repository folder under `/scratch`, `/projects`, or `/home`. Its absolute path is not hardcoded: jobs use `SLURM_SUBMIT_DIR`. Override `FLOW_IMAGE` / `FLOW_OVERLAY` if needed. Slurm still uses the `sbatch` command with `.slurm` files.

```bash
mkdir -p /scratch/mg8253/flow-bitrate/logs
grid_job=$(sbatch --parsable slurm/diagnostic.slurm)
sbatch --dependency=afterok:"$grid_job" slurm/oracle.slurm
```

The grid measures 4/6/8/10 codebooks at 2/6/12/24/48 evaluations, plus a 96-evaluation numerical reference and NDAC-only baseline. Same noise across budgets; two seeds for oracle selection and two separate seeds for evaluation. The sweep resumes completed recordings automatically. Pass `--config` to both jobs for another config; use a new cache if measurement settings change.

Results: `/scratch/mg8253/flow-bitrate/runs/initial-grid/summary.json` and `grid.csv`, `measurements.csv`, `effects.csv`, `oracle.csv`. W&B gets tables and rate/compute curves. Listening examples are in `/scratch/mg8253/flow-bitrate/cache/initial-grid/audio/`, one recording per domain. `WANDB_MODE=offline` is supported.

The oracle compares uniform rates, the best constant rate, and per-block allocation under a file budget including headers/flags/padding. Shuffling its assignments preserves the same rate histogram and tests whether placement matters. Positive gains mean lower spectral error; intervals resample independent groups. The oracle sees the original, so it is a diagnostic, not a deployable allocator.

A 96-evaluation solve is **not guaranteed converged**: inspect the 48-to-96 change. Waveform NMSE to that reference estimates numerical disagreement, not fidelity to the original. Spectral loss/SI-SDR do not prove perceptual quality. Listen to the examples, including joins between independent two-second blocks.

## Later — only if the diagnostic supports it

The allocator/fine-tuning pipeline is still available through `configs/experiment.json`, `slurm/prepare.slurm` and `slurm/train.slurm`. It uses Lightning + W&B, supports multi-GPU/multi-node jobs, and resumes with `--resume`. FlowDec-75m already supports all four rates; training across them alone is not a new contribution.

## Checks — run on HPC

```bash
sbatch slurm/single_gpu.slurm check  # unit tests + real weights forward/backward
```

No local execution or tests were run for this update. Full checkpoint/CUDA/Slurm checks are for HPC.

Setup pins FlowDec and patches eager imports, CUDA extension loading, and bf16 complex output. It uses native Torch ops. FlowDec's noncommercial license still applies.
