#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /scratch/mg8253/flow-bitrate/logs

# Four separate 24-hour, four-GPU jobs. Slurm decides when each can start.
for geometry in cartesian cylindrical; do
  sbatch --job-name="flow-${geometry}-noema" slurm/postfilter_train.slurm \
    --geometry "$geometry" --no-ema --seed 0 "$@"
  sbatch --job-name="flow-${geometry}-ema" slurm/postfilter_train.slurm \
    --geometry "$geometry" --ema --seed 0 "$@"
done
