#!/bin/bash
# Source after /ext3/env.sh so its old cache paths don't take precedence.
export FLOW_SCRATCH=/scratch/mg8253/flow-bitrate
export PIP_CACHE_DIR="$FLOW_SCRATCH/cache/pip"
export CONDA_PKGS_DIRS="$FLOW_SCRATCH/cache/conda"
export XDG_CACHE_HOME="$FLOW_SCRATCH/cache"
export MPLCONFIGDIR="$FLOW_SCRATCH/cache/matplotlib"
export NUMBA_CACHE_DIR="$FLOW_SCRATCH/cache/numba"
export TORCH_HOME="$FLOW_SCRATCH/cache/torch"
export HF_HOME="$FLOW_SCRATCH/cache/huggingface"
export PYTHONPYCACHEPREFIX="$FLOW_SCRATCH/cache/pycache"
export WANDB_DIR="$FLOW_SCRATCH/wandb"
export WANDB_CACHE_DIR="$FLOW_SCRATCH/cache/wandb"
export WANDB_DATA_DIR="$FLOW_SCRATCH/cache/wandb-data"
export WANDB_ARTIFACT_DIR="$FLOW_SCRATCH/artifacts"
export TMPDIR="$FLOW_SCRATCH/tmp/${SLURM_JOB_ID:-interactive-$$}/${SLURM_PROCID:-0}"
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -o cache_dir=$FLOW_SCRATCH/cache/pytest --basetemp=$TMPDIR/pytest"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" \
  "$TORCH_HOME" "$HF_HOME" "$PYTHONPYCACHEPREFIX" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
  "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$TMPDIR" "$FLOW_SCRATCH/logs" "$FLOW_SCRATCH/manifests"
