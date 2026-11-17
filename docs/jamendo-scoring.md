# Score saved Jamendo checkpoints

Run from the `flow-bitrate` project root on HPC. Pass either the paths to
specific checkpoint files or the run directories containing `best.ckpt`. Use
directories for active runs. If a run has yet to produce a best checkpoint, the
planner lists it as pending and includes the ready runs. Each plan freezes a
copy of the selected checkpoint versions and records their training steps,
hashes, and the audited Jamendo test cohort.

## One-time setup

In a writable scoring environment with the project's CUDA PyTorch environment
already installed, add the CLAP and fwSSNR dependencies and prepare the model
assets once with network access:

```bash
source scripts/scratch.sh
python -m pip install -e '.[scoring]'
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from laion_clap import CLAP_Module

directory = Path(os.environ["FLOW_SCRATCH"]) / "models/clap-laion-audio"
checkpoint = hf_hub_download("lukewys/laion_clap", "630k-audioset-best.pt", local_dir=directory)
# Populate the shared Hugging Face cache required by the CLAP loader.
clap = CLAP_Module(enable_fusion=False, device="cpu", amodel="HTSAT-tiny")
clap.load_ckpt(checkpoint, verbose=False)
PY
```

If the scoring environment is separate from the training environment, point
Slurm to it with `export FLOW_BITRATE_ENV=/absolute/path/to/scoring-env` before
submission. Array jobs run offline and use the shared scratch cache.

## Plan, score and report

From the scoring environment, give the plan a name and pass whichever run
directories or checkpoints you want in the comparison:

```bash
mkdir -p /scratch/mg8253/flow-bitrate/logs
python -m flow_bitrate.score_jamendo plan --name jamendo-nfe6-v1 \
  --checkpoints /path/to/flowdec-ema-run /path/to/cyfm-cartesian-ema-run \
  /path/to/cyfm-cartesian-noema-run /path/to/cyfm-cylindrical-ema-run \
  /path/to/cyfm-cylindrical-noema-run

sbatch --array=0-7%4 slurm/jamendo_score.slurm --name jamendo-nfe6-v1 --shards 8

python -m flow_bitrate.score_jamendo status --name jamendo-nfe6-v1
python -m flow_bitrate.score_jamendo report --name jamendo-nfe6-v1
```

The default plan covers all eligible ten-second test excerpts, scores the
NDAC-75 baseline at 4.5 kbps, and scores each postfilter at six network
evaluations with midpoint and the two configured noise seeds. A separate plan
with `--limit 8` is a quick end-to-end check. Use `--nfe` at planning time
for another inference budget. Keep distinct plan names for distinct snapshots.

For an interrupted array, resubmit the same array command. Completed records
are skipped. To run only failed tasks, change the array indices and keep
`--shards 8`. You can also repartition the entire job with a different shard
count because record keys are independent of shard IDs. Status displays the
number of complete records for each checkpoint. A job ending at its time
limit can leave unfinished work. The report requires the entire planned
cohort and lists pending run directories separately.

When a running job has a new best checkpoint, make a new plan with the same
run directory. Its new checkpoint hash gets new results. The scorer
reuses matching records across plans, including clean reference embeddings,
NDAC scores and unchanged checkpoints. Each completed record is published
atomically, and file locks prevent overlapping jobs from scoring the same
clip and model at the same time. Only the in-flight result needs to be
recomputed after a kill. Checkpoint snapshots and records live under
`/scratch/mg8253/flow-bitrate/jamendo-scores/`. All commands accept
`--root /path/to/shared/storage` if you need a different location.

Reports are `summary.csv`, `summary.json` and `per_clip.csv` in the named plan
directory. The scorer uses the unfused LAION CLAP audio checkpoint with one
embedding per ten-second clip. It reports finite-sample FAD once per seed
against the same clean reference cohort and displays their mean times 100.
Paired metrics average seeds within each clip and then average over clips.
The report counts how many tracks contribute to each metric, including when
the upstream fwSSNR calculation returns a nonfinite value. SIGMOS is a
speech metric.
