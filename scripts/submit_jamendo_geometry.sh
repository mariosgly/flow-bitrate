#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /scratch/mg8253/flow-bitrate/logs

arm="${1:?Choose audit, weighted, warped, disk-loss, disk, or cylinder}"
shift
case "$arm" in
  audit)
    exec sbatch --job-name=jamendo-phase-audit slurm/single_gpu.slurm jamendo-geometry "$@" ;;
  weighted)  settings=(--geometry cylindrical --phase-loss state-weighted) ;;
  warped)    settings=(--geometry warped-disk) ;;
  disk-loss) settings=(--geometry cylindrical --phase-loss disk) ;;
  disk)      settings=(--geometry disk) ;;
  cylinder)  settings=(--geometry cylindrical) ;;
  *) echo "Choose audit, weighted, warped, disk-loss, disk, or cylinder" >&2; exit 2 ;;
esac
walltime=08:00:00
for arg in "$@"; do
  if [[ "$arg" == --smoke ]]; then walltime=00:30:00; fi
done
exec sbatch --ntasks-per-node=2 --gres=gpu:2 --time="$walltime" --job-name="jamendo-$arm" \
  slurm/postfilter_train.slurm --config configs/jamendo.json --backbone cyfm --ema "${settings[@]}" "$@"
