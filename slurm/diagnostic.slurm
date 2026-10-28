#!/bin/bash
#SBATCH --job-name=flow-grid
#SBATCH --account=torch_pr_529_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/mg8253/logs/%x-%A_%a.out
#SBATCH --error=/scratch/mg8253/logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="/scratch/mg8253/flow-bitrate/matplotlib"
export WANDB_DIR="${WANDB_DIR:-/scratch/mg8253/flow-bitrate/wandb}"
mkdir -p "$MPLCONFIGDIR" "$WANDB_DIR"
FLOW_IMAGE="${FLOW_IMAGE:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}"
FLOW_OVERLAY="${FLOW_OVERLAY:-/scratch/mg8253/images/flow-matching-50GB.ext3}"
export FLOW_BITRATE_ENV="${FLOW_BITRATE_ENV:-/ext3/miniforge3/envs/flow-bitrate}"
srun --kill-on-bad-exit=1 singularity exec --nv --fakeroot --overlay "${FLOW_OVERLAY}:ro" \
  --bind /home:/home --bind /scratch:/scratch --bind /projects:/projects \
  "$FLOW_IMAGE" bash slurm/worker.sh diagnostic \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SLURM_ARRAY_TASK_COUNT" "$@"
