#!/bin/bash
#SBATCH --job-name=flow-labels
#SBATCH --account=torch_pr_529_general
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/mg8253/flow-bitrate/logs/%x-%A_%a.out
#SBATCH --error=/scratch/mg8253/flow-bitrate/logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LABEL_IMAGE="${FLOW_LABEL_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
LABEL_OVERLAY="${FLOW_LABEL_OVERLAY:-/scratch/mg8253/images/overlay-15GB-500K-aac.ext3}"
export FLOW_BITRATE_ENV="${FLOW_LABEL_ENV:-/ext3/miniforge3/envs/aac-dcase24}"
srun --kill-on-bad-exit=1 singularity exec --nv --fakeroot --overlay "${LABEL_OVERLAY}:ro" \
  --bind /home:/home --bind /scratch:/scratch --bind /projects:/projects \
  "$LABEL_IMAGE" bash --noprofile --norc slurm/worker.sh wavcaps-labels "$@"
