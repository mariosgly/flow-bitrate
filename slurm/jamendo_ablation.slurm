#!/bin/bash
#SBATCH --job-name=jamendo-ablation
#SBATCH --account=torch_pr_529_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/mg8253/flow-bitrate/logs/%x-%A_%a.out
#SBATCH --error=/scratch/mg8253/flow-bitrate/logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
campaign="${1:?Usage: sbatch --array=1-24 slurm/jamendo_ablation.slurm CAMPAIGN [training options]}"
shift
if [[ ! "$campaign" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Use letters, numbers, dashes, dots or underscores for the campaign name." >&2
  exit 2
fi
arm="${SLURM_ARRAY_TASK_ID:?Specify an ablation arm with sbatch --array}"
locks=/scratch/mg8253/flow-bitrate/runs/.ablation-locks
mkdir -p "$locks"
exec 9>"$locks/$campaign-$arm.lock"
if ! flock -n 9; then
  echo "Arm $arm of $campaign is already running." >&2
  exit 1
fi
bash slurm/postfilter_train.slurm --config configs/jamendo.json --campaign "$campaign" --ablation "$arm" "$@"
