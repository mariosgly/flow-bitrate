# Postfilter data

FlowDec does not release its exact training, validation or test file lists. These are our splits for a matched Cartesian/cylindrical comparison, not a reproduction of its benchmark.

| Dataset | Use | Split unit |
|---|---|---|
| Original EARS, 48 kHz | train p001–p099; val p100–p101; test p102–p107 | speaker |
| LibriTTS-R | all available train subsets; honour dev/test subsets if present | speaker |
| MTG-Jamendo | train/val/test, approximately 80/10/10 of artists | artist ID from metadata |
| WavCaps-FreeSound | train/val/test, approximately 80/10/10 of uploaders | author from original metadata |
| MedleyDB | additional music test corpus, full mixes only | artist prefix |

If LibriTTS-R dev/test subsets are absent, speakers from its training directories are held out for validation/test. Existing diagnostic source groups are reserved for test. Your previous WavCaps `splits/` and `splits_wav/` files are ignored.

We select the FreeSound part of WavCaps, matching the public sound source used by FlowDec. AudioSet_SL, BBC Sound Effects and SoundBible are not included in this experiment. FreeSound contains varied audio; the label `sound` is not a verified environmental-only category.

Jamendo metadata is required because audio filenames do not identify artists. `python scripts/download_data_metadata.py` downloads the official `raw.tsv` into project scratch, using the Python standard library. It installs no packages. Alternatively, point `jamendo_metadata` in `configs/postfilter_data.json` at an existing copy.

MusicCaps is optional and disabled by default: the supplied listing establishes a directory, not usable audio. To add it, set `musiccaps` to a directory containing audio with YouTube IDs in the filenames. It is test-only. We are not using the paper's VCTK/MUSDB18/AudioSet benchmark or claiming its exact 3,000-sample protocol.

## Preparation

Run `sbatch slurm/postfilter_data.slurm` from the repository root. This is an eight-CPU, four-hour job, with no GPU and no downloads. It uses the same read-only container and environment as training. Rerun after a timeout; SQLite preserves completed checks in `manifests/postfilter.audit.sqlite`. A batch of at most 64 checks may need repeating after a kill. `--overwrite` rebuilds an already-published manifest using cached checks; do not rebuild while training is active.

The job discovers WAV, FLAC, MP3 and OGG files; joins Jamendo artist and FreeSound author metadata; and keeps one usable recording per dataset ID. WavCaps FLAC is preferred to its converted WAV copy. It decodes the files fully in bounded chunks, checks finite/non-silent mono samples and partial seeking, and removes identical decoded-PCM duplicates, preferring held-out copies. Unreadable alternatives can fall back to a valid copy.

It then exercises the actual multi-worker CPU dataloader for every split/source/sample-rate/channel/format profile. Training refuses an absent/stale audit, and rejects changes to audited file sizes or modification times. This checks data access and preprocessing; it does not replace the first GPU training check.

Outputs under `/scratch/mg8253/flow-bitrate/manifests/`:

- `postfilter.csv`: selected paths, source groups, splits, audio metadata and PCM fingerprints.
- `postfilter.summary.json`: readiness, recording/group counts, hours, library versions, metadata hashes and exclusion counts.
- `postfilter.excluded.csv`: missing metadata/audio, corrupt files, duplicates, silence and short recordings.
- `postfilter.audit.sqlite`: resumable per-file decode checks. Input audio stays in its shared directory.

## Loading and evaluation

Training chooses a domain uniformly, then a dataset within that domain, then a recording and random two-second crop. Both speech sources are therefore represented. Validation uses fixed two-second crops; test uses fixed ten-second crops. Test files shorter than 10.02 seconds are excluded, rather than repeated. The datasets are mono-mixed and resampled to 48 kHz on loading. Upsampling 24/32 kHz audio does not restore missing high frequencies; original EARS supplies full-band speech, and reports retain original sampling rates.

Source groups and datasets are interleaved when selecting validation/test subsets. `test_per_domain` limits evaluation size, not the manifest's training pool. Ten-second tests run as five independent two-second reconstructions, with waveform metrics calculated after concatenation. Results also include per-dataset summaries so Jamendo and MedleyDB can be inspected separately.

The audit removes exact decoded duplicates, not every resampled/edited copy or near-duplicate. Cross-corpus artist/person identities and the frozen NDAC's original pretraining overlap are not fully known. Both trained postfilters use the same frozen NDAC and the same manifest.

Sources: [FlowDec data preparation](https://github.com/facebookresearch/FlowDec#data-preparation), [EARS benchmark speaker split](https://github.com/sp-uhh/ears_benchmark/blob/main/generate_ears_wham.py), [MTG-Jamendo metadata](https://github.com/MTG/mtg-jamendo-dataset), [WavCaps](https://huggingface.co/datasets/cvssp/WavCaps).
