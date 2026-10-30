#!/bin/bash
#SBATCH --job-name=flow-grid-40m
#SBATCH --account=torch_pr_529_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --array=0-29
#SBATCH --requeue
#SBATCH --output=/scratch/mg8253/flow-bitrate/logs/%x-%A_%a.out
#SBATCH --error=/scratch/mg8253/flow-bitrate/logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
FLOW_IMAGE="${FLOW_IMAGE:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}"
FLOW_OVERLAY="${FLOW_OVERLAY:-/scratch/mg8253/images/flow-matching-50GB.ext3}"
export FLOW_BITRATE_ENV="${FLOW_BITRATE_ENV:-/ext3/miniforge3/envs/flow-bitrate}"

# One of the current 30 recordings per task; no array concurrency cap.
# Keep num-shards=30 even when retrying just a subset of array IDs.
srun --kill-on-bad-exit=1 singularity exec --nv --fakeroot --overlay "${FLOW_OVERLAY}:ro" \
  --bind /home:/home --bind /scratch:/scratch --bind /projects:/projects \
  "$FLOW_IMAGE" bash --noprofile --norc slurm/worker.sh diagnostic \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards 30 --max-seconds 2100 "$@"
