# Expanded data, separate from the original manifests

Outputs: `/scratch/mg8253/flow-bitrate/manifests_v2/`. Original CSVs, SQLite audits and scan indexes are read-only inputs. Audio stays in place. No audio generation or new captions: Qwen classifies the existing WavCaps metadata.

## One-time setup

Copy the updated repository to `/home/mg8253/workspace/flow-bitrate`. Labeling uses the existing `aac-dcase24` environment. Its Torch 2.5.1, Transformers 4.57.6, Hugging Face Hub and safetensors cover these standalone scripts. Don't install the project's pinned extras into it: those target the separate training environment. Enter the read-only labeling container:

```bash
singularity exec --nv --fakeroot \
  --overlay /scratch/mg8253/images/overlay-15GB-500K-aac.ext3:ro \
  --bind /home:/home --bind /scratch:/scratch --bind /projects:/projects \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash --noprofile --norc
```

Inside the container:

```bash
source /home/mg8253/.bashrc
source /ext3/env.sh
cd /home/mg8253/workspace/flow-bitrate
source scripts/scratch.sh
conda activate /ext3/miniforge3/envs/aac-dcase24
python -c 'import torch, transformers; from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM; print("Torch:", torch.__version__, "Transformers:", transformers.__version__)'
python scripts/download_label_model.py
exit
```

Qwen2.5-7B-Instruct is downloaded to project scratch, with its resolved Hugging Face revision saved. Interrupted downloads resume at the same revision. Compute jobs use local weights and offline mode. No installs happen inside their read-only overlay. An invalid `-ransformers` distribution warning can be leftover installation metadata; the import check above must succeed. Extraction, inventory, audits, merge and training still use the original flow-bitrate image/environment. Labeling overrides are `FLOW_LABEL_IMAGE`, `FLOW_LABEL_OVERLAY` and `FLOW_LABEL_ENV`.

## Extract and inventory

On the host, from the repository root:

```bash
cd /home/mg8253/workspace/flow-bitrate
mkdir -p /scratch/mg8253/flow-bitrate/logs
EXTRACT_JOB=$(sbatch --parsable slurm/libritts_extract.slurm)
sbatch --dependency=afterok:"$EXTRACT_JOB" slurm/postfilter_v2_metadata.slurm inventory
```

The five CPU tasks check/extract the four dev/test archives and the restoration-failure list from `/scratch/mg8253/datasets/libritss/` into its `extracted/` folder. Rerun the extraction array after interruption: completed archives are skipped and the unfinished archive is re-extracted. No training archives are copied or downloaded. Extraction markers are required before inventory publication.

Inventory reads the old data config, scans the new dev/test folders and all four WavCaps audio sources, matches metadata IDs, and fixes the shard assignment. It reuses copies of old directory indexes. It must find all four LibriTTS-R held-out subsets and matched audio for every WavCaps source. Paths/settings are in `configs/postfilter_data_v2.json`.

Expected new metadata: `/scratch/mg8253/datasets/audioset/{ontology.json,class_labels_indices.csv,mid_to_display_name.tsv,audioset_train_strong.tsv,audioset_eval_strong.tsv}`. Existing segment CSVs are read from `/projects/work/marl/sonyc/AudioSet/csv`.

Watch `logs/flow-extract-ARRAY_TASK.out` and `logs/flow-v2-meta-JOB.out`. Wait for `Inventory ready` before the next commands. Inventory is frozen once published; changing inputs later requires another version directory. Inventory/merge are single CPU coordination jobs; text inference and decode checks are sharded.

## Submit the heavy work

```bash
bash scripts/submit_postfilter_data_v2.sh audit
```

Run audits first. Each array has ten two-hour jobs. Audit uses eight CPUs/task and no GPU. Once all audits are complete, submit `bash scripts/submit_postfilter_data_v2.sh labels`. Labels use one GPU/task, batch eight texts, bf16 where supported; OOM retries halve the batch size. No whole-AudioSet squashFS files are mounted or decoded.

Labeling requires its completed, matching audit shard. It skips failed/corrupt/silent recordings, train/validation files shorter than two seconds, and test files shorter than ten seconds. Exactly two/ten seconds is accepted. Annotation-based labels bypass Qwen too. Global duplicates are removed later during merge, so some duplicate copies can still receive labels. Cached labels remain reusable.

AudioSet labels are mapped through the ontology when the segment match is unambiguous; captions are the fallback. Strong event intervals and original event labels are preserved. Remaining WavCaps captions, tags and descriptions go through a fixed Qwen prompt with deterministic decoding. Speech/music/other_sound may co-occur; unclear or malformed outputs are marked for review. These are metadata-based content labels, not proven absence labels or exact event-duration measurements. The strong annotations do not describe sounds within musical segments exhaustively.

Labels commit after every batch; decode results commit per recording. A timeout can repeat unfinished work but committed results are reused. Shards have independent SQLite files, progress JSONs and exclusive writer locks. A graceful slice can finish with Slurm COMPLETED while still incomplete; trust the progress counts. The merger refuses missing or incomplete shards. Hard kills can still produce Slurm TIMEOUT/FAILED; resubmission recovers committed work.

After array jobs end, check and resubmit only unfinished shards:

```bash
bash scripts/submit_postfilter_data_v2.sh status
bash scripts/submit_postfilter_data_v2.sh audit
```

Repeat audits as needed. Once all audits are complete, submit labels with `bash scripts/submit_postfilter_data_v2.sh labels`; repeat that same command after labeling slices end to resume unfinished labels. Do not launch overlapping retries of the same shard; the lock rejects duplicate writers. An individual task can also be retried with `sbatch --array=3 slurm/wavcaps_classify.slurm` or `sbatch --array=3 slurm/postfilter_v2_audit.slurm`. Progress lives beside each shard; logs use `%A_%a` for array/task IDs.

Old unchanged successful CSV entries and old SQLite checks are reused by path/size/mtime. New dev/test and new WavCaps audio are decoded and checked for finite, non-silent mono samples, duration and partial seeks. Preferred FLAC copies are tried first, with alternatives on failure. Valid exactly-ten-second test clips are accepted. Shorter clips remain excluded from ten-second testing.

## Merge and inspect

Once every labels/audit count is complete:

```bash
sbatch slurm/postfilter_v2_metadata.slurm merge
```

This checks shard identities, removes duplicates globally with held-out copies winning, checks source groups and all actual CPU loader profiles, then publishes:

- `postfilter.csv`: source paths/splits, content labels, captions/metadata, event annotations and sampling buckets.
- `postfilter.summary.json`: readiness, hours/counts, exclusions and provenance.
- `postfilter.excluded.csv`: rejected recordings and reasons.
- `postfilter.review.csv`: examples across WavCaps sources/content categories plus uncertain cases to listen to.

The summary is the readiness marker. A failed/timed-out merge does not publish it; rerun the merge to finish. It may repeat coordination/loader checks, never all audio decodes. Review classifications before treating category statistics as reliable; model output is not ground truth. Full recording hours under mixed domains are exclusive; a clip-level label is not guaranteed to describe every two-second crop.

All three LibriTTS-R training subsets are used, with new official dev/test subsets; known restoration failures are excluded. Earlier diagnostic source groups stay reserved. Non-Libri old split assignments are retained. AudioSet eval videos are test-only; its remaining source groups use deterministic splits. FreeSound/SoundBible use authors, BBC uses recording IDs because session identity is unavailable. Exact PCM duplicate removal does not establish absence of near duplicates or cross-corpus identity overlap.

Training keeps three source sampling pools: speech, music and general audio. Within general audio, all WavCaps sources form one recording pool, so a small source is not oversampled as an equally weighted dataset. `domain` stores content; `sampling_domain` and `sampling_dataset` control loading. Validation still selects 32 fixed recordings per sampling pool, interleaving the actual datasets. Captions are not supplied to the flow network.

To use the new manifest explicitly:

```bash
sbatch slurm/postfilter_smoke.slurm --config configs/postfilter_v2.json --ema
# After SMOKE PASSED and reviewing the data:
bash scripts/submit_postfilter_runs.sh --config configs/postfilter_v2.json
```

The original training config still points to the original manifest. Start fresh runs for the changed dataset rather than resuming old checkpoints with different provenance. Prepared without local training, classification, audio decoding or tests.
