# Audio for the first run

Start with about **30 recordings: 10 speech, 10 music, 10 everyday/environmental sounds**. This is a pilot, not a final benchmark. Use different speakers, songs and recording sources where possible.

- Speech: original VCTK or EARS recordings, at least two seconds long. Short utterances still test rates/steps.
- Music: MUSDB18-**HQ** `mixture.wav` files. No stems needed. Pick a representative section with `start_seconds`.
- Sounds: original FSD50K WAVs, preferably at least ten seconds long. Include impacts, tonal sounds and textures.

Use recordings you consider suitable references. WAV/PCM does not by itself mean a recording is pristine. No codec outputs, transcripts, captions or detailed event labels are needed.

The reader averages stereo to mono and resamples to 48 kHz. The default takes up to **ten seconds** from each listed offset and drops the final partial two-second block. Allocation needs **five full blocks** by default: shorter excerpts enter the rate/compute grid, but are excluded from the oracle. Make sure you have enough long music/sound excerpts. Resampling does not restore missing bandwidth.

Copy `diagnostic.example.csv` to `diagnostic.csv`:

```csv
path,split,group,domain,start_seconds
/scratch/mg8253/audio/speech/a.wav,val,speaker_01,speech,0
/scratch/mg8253/audio/music/song/mixture.wav,val,song_01,music,30
/scratch/mg8253/audio/sounds/b.wav,val,uploader_01,sound,0
```

Paths must exist **inside the Slurm container**. Group by speaker/song/uploader when examples share that source. Keep this exploratory set in `val`; no training split is needed yet. Reserve other recordings for later confirmation. Check pretrained-data overlap before claiming unseen-data results.

Also place the official weights at:

```text
checkpoints/ndac/ndac-75/800k/dac/weights.pth
checkpoints/flowdec/flowdec_75m/step=800000.ckpt
```

`scripts/download_models.py` installs those from the official archive. Download on HPC before submitting GPU jobs; compute nodes need no download access. Point the diagnostic config at scratch locations for the manifest, checkpoints, cache and outputs.
