#!/bin/bash
#SBATCH --job-name=flow-data
#SBATCH --account=torch_pr_529_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/mg8253/flow-bitrate/logs/%x-%j.out
#SBATCH --error=/scratch/mg8253/flow-bitrate/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
FLOW_IMAGE="${FLOW_IMAGE:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}"
FLOW_OVERLAY="${FLOW_OVERLAY:-/scratch/mg8253/images/flow-matching-50GB.ext3}"
export FLOW_BITRATE_ENV="${FLOW_BITRATE_ENV:-/ext3/miniforge3/envs/flow-bitrate}"
# CPU-only. Full decode checks are cached, so resubmit after a timeout.
srun --kill-on-bad-exit=1 singularity exec --fakeroot --overlay "${FLOW_OVERLAY}:ro" \
  --bind /home:/home --bind /scratch:/scratch --bind /projects:/projects \
  "$FLOW_IMAGE" bash --noprofile --norc slurm/worker.sh postfilter-data "$@"
