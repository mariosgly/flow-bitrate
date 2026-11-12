#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /scratch/mg8253/flow-bitrate/logs

resources=(--ntasks-per-node=2 --gres=gpu:2 --time=08:00:00)

sbatch "${resources[@]}" --job-name=jamendo-flowdec-ema slurm/postfilter_train.slurm \
  --config configs/jamendo.json --seed 0 --backbone flowdec --geometry cartesian --ema

for geometry in cartesian cylindrical; do
  sbatch "${resources[@]}" --job-name="jamendo-cyfm-${geometry}-ema" slurm/postfilter_train.slurm \
    --config configs/jamendo.json --seed 0 --backbone cyfm --geometry "$geometry" --ema
  sbatch "${resources[@]}" --job-name="jamendo-cyfm-${geometry}-noema" slurm/postfilter_train.slurm \
    --config configs/jamendo.json --seed 0 --backbone cyfm --geometry "$geometry" --no-ema
done
