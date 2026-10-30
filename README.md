# Flow bitrate

First ask whether [FlowDec](https://github.com/facebookresearch/FlowDec) needs more bits or more decoding computation. Run the frozen-model grid and oracle **before training anything**.

## Setup

Use the CUDA 12.6.3 SIF with `/scratch/mg8253/images/flow-matching-50GB.ext3`. After installing Miniforge and `/ext3/env.sh` as discussed, enter the overlay writable. Inside that container, from the repository folder:

```bash
source /home/mg8253/.bashrc
source /ext3/env.sh
source scripts/scratch.sh
conda create --prefix /ext3/miniforge3/envs/flow-bitrate python=3.10 pip git -y
conda activate /ext3/miniforge3/envs/flow-bitrate
bash scripts/setup.sh --cuda
python scripts/download_models.py  # reads configs/diagnostic.json; optional --config / --archive
wandb login
exit
```

Exit the writable container before submitting jobs. Every job sources `/home/mg8253/.bashrc`, `/ext3/env.sh`, then `scripts/scratch.sh` **inside Singularity**, and activates that exact environment prefix. Nothing sources `.bashrc` on the host. Copy source files to HPC, not the Mac `.venv`. The editable installation uses the HPC repository path; reinstall if you move it afterward.

Setup installs the pinned dependencies (including tests), clones/patches FlowDec, and runs `pip check`. The download script fetches both required checkpoints. No job installs packages, downloads weights, or compiles custom CUDA extensions; first-use caches go to scratch. You still need the audio files and completed manifest. W&B needs network access for online logging; export `WANDB_MODE=offline` before submission otherwise.

Everything generated lives under `/scratch/mg8253/flow-bitrate`: pretrained downloads in `models/`, training checkpoints and reports in `runs/`, measurements/audio in `cache/`, W&B in `wandb/`, Slurm output in `logs/`, and test temporary files in `tmp/`. Configured artifact paths must stay under that project folder. The repository location is independent.

For the first run, follow [data/README.md](data/README.md) and copy the manifest template:

```bash
mkdir -p /scratch/mg8253/flow-bitrate/manifests
cp -n data/diagnostic.example.csv /scratch/mg8253/flow-bitrate/manifests/diagnostic.csv
```

Fill in its audio paths. Both configs already use the project scratch folder. Relative paths in those configs resolve there, not into the repository. For interactive commands inside Singularity, use the same sourcing/activation order as setup.

## First experiment — HPC

Submit from the repository folder under `/scratch`, `/projects`, or `/home`. Its absolute path is not hardcoded: jobs use `SLURM_SUBMIT_DIR`. Override `FLOW_IMAGE` / `FLOW_OVERLAY` if needed. Slurm still uses the `sbatch` command with `.slurm` files.

```bash
mkdir -p /scratch/mg8253/flow-bitrate/logs
sbatch slurm/initial.slurm
```

This requests **one GPU, four CPUs, 32 GB host RAM, and 24 hours**, with no array. It runs the full rate/computation grid, then oracle analysis in the same job. It asks whether extra bits or extra flow evaluations help, and whether placing bits differently helps under the same budget. Completed recordings are reused if you resubmit after a timeout. Optional: pass `--config configs/another.json`.

For **two-hour slices on one GPU**, use this instead, and rerun the same command until the experiment finishes:

```bash
sbatch slurm/initial_2h.slurm
```

Each initial job first checks one real recording with the released weights at all four rates, including both decoder paths, waveform lengths, spectral loss/SI-SDR, and flow reconstruction. It aborts before the grid if this fails. The two-hour job includes that preflight in its 1h55m sweep budget, leaving a small margin before Slurm's limit. A cleanly completed Slurm slice can still mean there is work left; its log says whether to resubmit. Once the grid is complete, the job runs the oracle and writes `runs/initial-grid/summary.json` under project scratch. If killed during analysis, rerun the same job; inference is already cached and analysis is repeated. No reinstall is needed after copying the updated Python/Slurm files.

Each completed reconstruction is checkpointed, including the active high-NFE reference, with flush/fsync and atomic replacement. Hard kills can lose the solve currently running (plus inexpensive setup), but not previously committed solves on a functioning filesystem. Partial `.part` writes are ignored. Listening audio is saved in per-block pieces and assembled at completion. Record locks prevent overlapping jobs from writing the same record. Do not change the manifest, weights, or measurement settings while resuming: their hashes are checked. Existing completed recording caches remain usable. Submit the next slice after the previous job has stopped; no manual checkpoint selection is needed. Reports are also written atomically; `summary.json` is published after all report CSVs.

For **40-minute jobs with no array concurrency cap**, use the array for the current 30-recording manifest:

```bash
sbatch slurm/diagnostic_40m.slurm
# After every recording finishes:
sbatch slurm/oracle.slurm
```

There are 30 tasks, each assigned one recording, with no script-imposed concurrency limit; actual concurrency depends on Slurm availability and account limits. Each requests one GPU, four CPUs, 32 GB RAM, and 40 minutes; the sweep stops near 35 minutes if unfinished. Existing completed recordings are skipped and partial recordings resume from the same cache. Stop/wait for the old job before switching. Rerun the array if needed; an unfinished task exits 75 rather than reporting success. For a subset retry, `sbatch --array=4,7,12 slurm/diagnostic_40m.slurm` keeps the original recording assignments. This changes scheduling, not the experiment. The older four-shard `diagnostic.slurm` is still available. No reinstall is needed.

The grid measures 4/6/8/10 codebooks at 2/6/12/24/48 evaluations, plus a 96-evaluation numerical reference and NDAC-only baseline. Same noise across budgets; two seeds for oracle selection and two separate seeds for evaluation. The sweep resumes completed recordings automatically. Pass `--config` to both jobs for another config; use a new cache if measurement settings change.

Results: `/scratch/mg8253/flow-bitrate/runs/initial-grid/summary.json` and `grid.csv`, `measurements.csv`, `effects.csv`, `oracle.csv`. W&B gets tables and rate/compute curves. Listening examples are in `/scratch/mg8253/flow-bitrate/cache/initial-grid/audio/`, one recording per domain. `WANDB_MODE=offline` is supported.

After the grid, `sbatch slurm/oracle.slurm` also writes six PNG figures in `runs/initial-grid/plots/`, logs them to W&B, and writes `report.html` with an interpretation for each plot. Matplotlib is already a dependency of the installed `descript-audiotools`; no new install is needed. Copy the result folder to your computer and open `report.html` to view it with the relative images.

For listening, run `sbatch slurm/listen.slurm`. It exports the first recording per domain at 4/10 codebooks, with NDAC and flow at NFE 6 (or the first configured budget) and the high-NFE reference. It reuses existing complete WAVs or block fragments. Missing waveforms are regenerated with the measured weights, source offset, and first evaluation seed; it does not repeat the measurement grid. Audio and a player page go to `runs/initial-grid/audio/index.html`, alongside `source.json` provenance. Waveforms use the same scale, not separate peak normalization. Rerunning skips valid files and resumes missing pieces; `--all` exports all recordings. Do not run exports and sweeps concurrently on the same recordings. Export does not change measurement tensors. If a completed cache has no WAVs, ordinary sweep resume skips it; the listening job repairs those missing assets.

Read the plots in order: (1) spectral error and SI-SDR versus flow evaluations, with dashed NDAC baselines; (2) paired spectral gains over NDAC with group-bootstrap intervals; (3) numerical agreement with the high-NFE reference, not fidelity; (4) allocation gains over the selected constant rate and shuffled placements, with intervals; (5) quality versus measured decoder time, requiring comparable GPUs; (6) per-block gains from extra bits versus extra evaluations. Positive gains are improvements. Allocation headroom does not show that a flow property can predict where bits should go. Listen before treating metric changes as perceptual improvements.

The oracle compares uniform rates, the best constant rate, and per-block allocation under a file budget including headers/flags/padding. Shuffling its assignments preserves the same rate histogram and tests whether placement matters. Positive gains mean lower spectral error; intervals resample independent groups. The oracle sees the original, so it is a diagnostic, not a deployable allocator.

A 96-evaluation solve is **not guaranteed converged**: inspect the 48-to-96 change. Waveform NMSE to that reference estimates numerical disagreement, not fidelity to the original. Spectral loss/SI-SDR do not prove perceptual quality. Listen to the examples, including joins between independent two-second blocks.

NDAC outputs are right-zero-padded or cropped to exactly 96,000 samples before flow reconstruction and scoring; odd decoder strides can otherwise leave a few samples missing. Differences of a full codec hop or more are rejected. This is our fixed-block boundary policy, recorded in the codec identity so caches from an older policy cannot be mixed. Metrics require equal shapes and do not silently crop the reference.

## Later — only if the diagnostic supports it

The allocator/fine-tuning pipeline is still available through `configs/experiment.json`, `slurm/prepare.slurm` and `slurm/train.slurm`. It uses Lightning + W&B, supports multi-GPU/multi-node jobs, and resumes with `--resume`. FlowDec-75m already supports all four rates; training across them alone is not a new contribution.

## Checks — run on HPC

```bash
sbatch slurm/single_gpu.slurm check  # unit tests + real weights forward/backward
```

No local execution or tests were run for this update. Full checkpoint/CUDA/Slurm checks are for HPC.

Setup pins FlowDec and patches eager imports, CUDA extension loading, and bf16 complex output. It uses native Torch ops. FlowDec's noncommercial license still applies.
