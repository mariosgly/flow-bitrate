#!/bin/bash
#SBATCH --job-name=jamendo-matched
#SBATCH --account=torch_pr_529_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/mg8253/flow-bitrate/logs/%x-%j.out
#SBATCH --error=/scratch/mg8253/flow-bitrate/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
arm="${1:?Choose disk, cylindrical, warped-disk, disk-loss, weighted, or flowdec. Prefix a CyFM arm with small- for width 64.}"
shift
backbone=cyfm
match=capacity
if [[ "$arm" == flowdec ]]; then
  backbone=flowdec
  match=optimizer
  arm=cartesian
elif [[ "$arm" == small-* ]]; then
  match=optimizer
  arm="${arm#small-}"
fi
case "$arm" in
  cartesian|disk) settings=(--geometry cartesian) ;;
  cylindrical|cylinder) settings=(--geometry cylindrical) ;;
  warped-disk|warped) settings=(--geometry warped-disk) ;;
  disk-loss) settings=(--geometry cylindrical --phase-loss disk) ;;
  weighted|state-weighted) settings=(--geometry cylindrical --phase-loss state-weighted) ;;
  *) echo "Unknown matched Jamendo arm: $arm" >&2; exit 2 ;;
esac
exec bash slurm/postfilter_train.slurm "$@" --config configs/jamendo.json --backbone "$backbone" \
  "${settings[@]}" --flowdec-match "$match" --ema
