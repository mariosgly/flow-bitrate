# Audio for the first run

Start with about **30 recordings: 10 speech, 10 music, 10 everyday/environmental sounds**. This is a pilot, not a final benchmark. Use different speakers, songs and recording sources where possible.

For the shared HPC collections, run this inside Singularity with the project environment active (no GPU needed):

```bash
python scripts/make_manifest.py
```

Defaults search the shared LibriTTS-R `train-clean-100`, `/projects/work/marl/sonyc/marl/private_datasets/MedleyDB/Audio`, and `/projects/work/marl/datasets/sound_datasets/fsd50k`. Override paths with `--speech`, `--music`, or `--sounds`. It chooses ten different speakers, MedleyDB artists (full `*_MIX` files only), and FSD50K uploaders, with at least ten seconds plus a small rounding margin. Excerpts start near the recording's middle; unreadable, nonfinite, and effectively silent excerpts are skipped. Selection is seeded from the diagnostic config; no audio is copied and no packages are installed.

The script needs extracted WAV/FLAC files and FSD50K's extracted `dev_clips_info_FSD50K.json` and/or `eval_clips_info_FSD50K.json` somewhere under its sound root. It stops without writing a partial manifest if any domain has fewer than ten eligible sources. The output comes from `configs/diagnostic.json`; use `--overwrite` to replace an existing manifest. Use a fresh diagnostic cache if changing an already measured manifest. Automatic selection does not ensure event/genre balance or exclude speech/music events from FSD50K; inspect the selected examples before treating the pilot as a benchmark.

- Speech: original VCTK or EARS recordings, at least two seconds long. Short utterances still test rates/steps.
- Music: MUSDB18-**HQ** `mixture.wav` files. No stems needed. Pick a representative section with `start_seconds`.
- Sounds: original FSD50K WAVs, preferably at least ten seconds long. Include impacts, tonal sounds and textures.

Use recordings you consider suitable references. WAV/PCM does not by itself mean a recording is pristine. No codec outputs, transcripts, captions or detailed event labels are needed.

The reader averages stereo to mono and resamples to 48 kHz. The default takes up to **ten seconds** from each listed offset and drops the final partial two-second block. Allocation needs **five full blocks** by default: shorter excerpts enter the rate/compute grid, but are excluded from the oracle. Make sure you have enough long music/sound excerpts. Resampling does not restore missing bandwidth.

Copy `data/diagnostic.example.csv` to `/scratch/mg8253/flow-bitrate/manifests/diagnostic.csv` and fill it in:

```csv
path,split,group,domain,start_seconds
/scratch/mg8253/flow-bitrate/audio/speech/a.wav,val,speaker_01,speech,0
/scratch/mg8253/flow-bitrate/audio/music/song/mixture.wav,val,song_01,music,30
/scratch/mg8253/flow-bitrate/audio/sounds/b.wav,val,uploader_01,sound,0
```

Paths must exist **inside the Slurm container**. Group by speaker/song/uploader when examples share that source. Keep this exploratory set in `val`; no training split is needed yet. Reserve other recordings for later confirmation. Check pretrained-data overlap before claiming unseen-data results.

Also place the official weights at:

```text
/scratch/mg8253/flow-bitrate/models/ndac/ndac-75/800k/dac/weights.pth
/scratch/mg8253/flow-bitrate/models/flowdec/flowdec_75m/step=800000.ckpt
```

`scripts/download_models.py` reads the diagnostic config and downloads/extracts into its `checkpoints` directory, already set to the scratch `models/` folder. Download on HPC before submitting GPU jobs; compute nodes need no download access. Existing datasets may stay at other readable locations; all generated outputs stay in the project scratch folder.
