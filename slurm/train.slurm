#!/bin/bash
#SBATCH --job-name=flow-bitrate
#SBATCH --account=torch_pr_529_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@180
#SBATCH --requeue
#SBATCH --output=/scratch/mg8253/flow-bitrate/logs/%x-%j.out
#SBATCH --error=/scratch/mg8253/flow-bitrate/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
FLOW_IMAGE="${FLOW_IMAGE:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}"
FLOW_OVERLAY="${FLOW_OVERLAY:-/scratch/mg8253/images/flow-matching-50GB.ext3}"
export FLOW_BITRATE_ENV="${FLOW_BITRATE_ENV:-/ext3/miniforge3/envs/flow-bitrate}"

# Slurm starts each rank. Lightning detects SLURM_PROCID; no torchrun inside srun.
srun --kill-on-bad-exit=1 --gpu-bind=none \
  singularity exec --nv --fakeroot --overlay "${FLOW_OVERLAY}:ro" \
  --bind /home:/home --bind /scratch:/scratch --bind /projects:/projects \
  "$FLOW_IMAGE" bash slurm/worker.sh train "$@"
