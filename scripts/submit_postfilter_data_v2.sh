#!/bin/bash
# Call after previous array tasks end; completed shards are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:?Use labels, audit or status}"
case "$stage" in
  labels) job=slurm/wavcaps_classify.slurm ;;
  audit) job=slurm/postfilter_v2_audit.slurm ;;
  status) job= ;;
  *) echo "Use labels, audit or status" >&2; exit 1 ;;
esac
shift
mkdir -p /scratch/mg8253/flow-bitrate/logs
remaining=$(python3 - "$stage" <<'PY'
import json
import sys
from pathlib import Path
config = json.loads(Path('configs/postfilter_data_v2.json').read_text())
root = Path(config['output'])
if not (root / 'inventory.json').exists():
    raise SystemExit('Run the inventory job first.')
if sys.argv[1] == 'status':
    for shard in range(config['num_shards']):
        parts = []
        for stage in ('labels', 'audit'):
            path = root / f'{stage}_{shard:02d}.progress.json'
            value = json.loads(path.read_text()) if path.exists() else None
            parts.append(f'{stage}: {value["completed"]}/{value["total"]}' if value else f'{stage}: not started')
        print(f'{shard:02d}  ' + '  '.join(parts))
    print('Manifest ready:', (root / 'postfilter.summary.json').exists())
    raise SystemExit(0)
pending = []
if sys.argv[1] == 'labels':
    incomplete = []
    for shard in range(config['num_shards']):
        path = root / f'audit_{shard:02d}.progress.json'
        if not path.exists() or not json.loads(path.read_text())['complete']:
            incomplete.append(str(shard))
    if incomplete:
        raise SystemExit('Finish CPU audit shards before labeling: ' + ','.join(incomplete))
for shard in range(config['num_shards']):
    path = root / f'{sys.argv[1]}_{shard:02d}.progress.json'
    if not path.exists() or not json.loads(path.read_text())['complete']:
        pending.append(str(shard))
print(','.join(pending))
PY
)
if [[ "$stage" == status ]]; then
  echo "$remaining"
  exit 0
fi
if [[ -z "$remaining" ]]; then
  echo "All $stage shards are complete."
else
  sbatch --array="$remaining" "$job" "$@"
fi
