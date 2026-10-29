#!/bin/bash
set -eo pipefail
# This worker is launched inside Singularity, never on the host.
source /home/mg8253/.bashrc
source /ext3/env.sh
source scripts/scratch.sh
conda activate "${FLOW_BITRATE_ENV:-/ext3/miniforge3/envs/flow-bitrate}"
export PYTHONUNBUFFERED=1
set -u
task="$1"
shift
if [[ "$task" == train ]]; then
  tasks_per_node="${SLURM_NTASKS_PER_NODE%%(*}"
  exec python -m flow_bitrate.train --devices "$tasks_per_node" --nodes "$SLURM_NNODES" \
    --accelerator gpu "$@"
elif [[ "$task" == prepare ]]; then
  exec python -m flow_bitrate.prepare --device cuda "$@"
elif [[ "$task" == evaluate ]]; then
  exec python -m flow_bitrate.evaluate --device cuda "$@"
elif [[ "$task" == diagnostic ]]; then
  exec python -m flow_bitrate.sweep --device cuda "$@"
elif [[ "$task" == initial ]]; then
  python scripts/smoke_codec.py --device cuda --inference-only "$@"
  # Separate processes release GPU memory before the CPU-only analysis.
  python -m flow_bitrate.sweep --device cuda --shard 0 --num-shards 1 "$@"
  exec python -m flow_bitrate.oracle "$@"
elif [[ "$task" == initial-resumable ]]; then
  slice_started=$SECONDS
  python scripts/smoke_codec.py --device cuda --inference-only "$@"
  remaining=$((6900 - SECONDS + slice_started))
  if (( remaining <= 0 )); then
    echo "Preflight used this slice's time budget; rerun the job to continue."
    exit 0
  fi
  status=0
  python -m flow_bitrate.sweep --device cuda --shard 0 --num-shards 1 --max-seconds "$remaining" "$@" || status=$?
  if [[ "$status" == 75 ]]; then
    echo "This two-hour slice is done; the experiment is not finished yet."
    echo "Rerun sbatch slurm/initial_2h.slurm with the same config to continue."
    exit 0
  elif [[ "$status" != 0 ]]; then
    exit "$status"
  fi
  exec python -m flow_bitrate.oracle "$@"
elif [[ "$task" == oracle ]]; then
  exec python -m flow_bitrate.oracle "$@"
elif [[ "$task" == check ]]; then
  python -m pytest -q
  exec python scripts/smoke_codec.py --device cuda "$@"
else
  echo "Unknown task: $task" >&2
  exit 1
fi
