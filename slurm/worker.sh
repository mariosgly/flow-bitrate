#!/bin/bash
set -eo pipefail
source /ext3/env.sh
conda activate "${FLOW_BITRATE_ENV:-/ext3/miniforge3/envs/flow-bitrate}"
set -u
export TMPDIR="/scratch/mg8253/flow-bitrate/tmp/${SLURM_JOB_ID}/${SLURM_PROCID:-0}"
mkdir -p "$TMPDIR"
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
elif [[ "$task" == oracle ]]; then
  exec python -m flow_bitrate.oracle "$@"
elif [[ "$task" == check ]]; then
  python -m pytest -q
  exec python scripts/smoke_codec.py --device cuda "$@"
else
  echo "Unknown task: $task" >&2
  exit 1
fi
