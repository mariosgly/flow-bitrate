# Cartesian versus cylindrical postfilter

Question: can cylindrical flow matching reach the same reconstruction quality with fewer network evaluations? Training efficiency is a second question: compare quality at matched optimizer updates and recorded fit time.

## What runs

- Frozen released NDAC-75, six codebooks: nominal 4.5 kbps.
- The U-Net and time embedding copied from the supplied CyFM repository, in `flow_bitrate/cyfm_unet.py`. Registry dependencies were removed; its layers were kept. See `licenses/` for attribution and source hashes.
- Both postfilters start from scratch. Each receives the current state, coarse NDAC spectrogram and time. Cartesian inputs have four channels; cylindrical inputs have six. Their remaining layers are identical.
- FlowDec's normalization, amplitude-compressed complex STFT, frequency-dependent noise and inverse STFT are reused through our existing wrapper. The released postfilter weights are not used to initialize the new U-Nets.
- Both geometries start from the same distribution: coarse complex spectrogram plus complex Gaussian noise. Cylindrical coordinates are taken after drawing that complex sample. The CyFM uniform prior and its optional amplitude/velocity ceilings are not used.
- Cartesian paths interpolate real/imaginary parts. Cylindrical paths interpolate magnitude and shortest-arc phase. Both regress squared velocity error, summed over the two coordinates, excluding padded STFT frames. Cylindrical phase has unit weight, with no target-amplitude weighting.
- No minibatch OT: each original audio crop stays paired with its own NDAC reconstruction.
- EMA is not implemented for these trainable postfilters. Conditioning is always present: no classifier-free dropout or classifier-free guidance.
- Sampling projects negative cylindrical amplitudes to zero and wraps phase. NFE=1 uses Euler; the default remaining budgets use midpoint, with two network calls per solver step.

The new training entry point is `flow_bitrate.train_postfilter`: Lightning + W&B, DDP under Slurm, mixed precision and checkpoints. Gradient accumulation is disabled. It uses the existing patched, pinned FlowDec dependency in `external/FlowDec`, not a second copy of the sibling checkout. No new packages are needed beyond the existing setup.

## Data

EARS and LibriTTS-R provide speech, MTG-Jamendo provides music, and WavCaps-FreeSound provides general sound. MedleyDB mixes are an additional music test corpus. Original dataset metadata defines speakers/artists/uploaders; your earlier WavCaps splits are ignored. Paths are in `configs/postfilter_data.json`. See [data preparation and splits](data.md).

Run the CPU data job before training: it decodes the selected audio, checks the actual dataloader and publishes an audit report required by training. Training uses random two-second crops, validation uses fixed two-second crops, and test uses ten-second excerpts reconstructed in five separate blocks. Domains and datasets within domains are sampled equally during training. This is our sampling policy, not FlowDec's original recipe.

The original diagnostic source groups are held out from training, but that study already informed the project; evaluation remains exploratory. Neither exact reproduction of the original paper's private data/file lists nor separation from frozen-checkpoint pretraining is established.

## Runs and results

Each Slurm job defaults to two GPUs. One Slurm task runs per GPU; Lightning handles distributed sampling. The worker sources `/home/mg8253/.bashrc`, `/ext3/env.sh` and the scratch settings inside the read-only container, then activates `/ext3/miniforge3/envs/flow-bitrate`. No packages or models are downloaded by a job.

The default effective batch is 128 crops: 64 per GPU, two GPUs, no gradient accumulation. Validation runs every 1,000 optimizer updates. Reconstruction metrics use the first two validation batches per rank, a small interleaved subset; flow loss uses the full selected validation set. Best-checkpoint selection uses spectral error at NFE=6 on that subset, not the incomparable flow losses of the two geometries. Increase `metric_batches` for broader reconstruction validation.

W&B receives velocity losses, validation spectral error/SI-SDR at several NFEs, elapsed fit time and audio. FLOAT WAV examples also live in each run's `audio/` folder. Waveforms share their original scale. Fit time includes validation and is not a pure training-kernel benchmark.

`last.ckpt` is saved every 1,000 optimizer updates, `best.ckpt` after improved validation, and milestone checkpoints every 10,000 updates. A new submission gets a unique job suffix and W&B ID, even when `--name` repeats. Local launches generate a shared random token for DDP children. Explicit `--resume /absolute/path/last.ckpt` reuses its run directory and W&B ID; Slurm requeues also resume automatically. A hard kill can lose work since the last readable checkpoint; random crop streams are not restored exactly. Do not resume one checkpoint in overlapping jobs. Keep the configuration, data and GPU count fixed while resuming.

## Two-GPU smoke job

After the data audit passes, submit `sbatch slurm/postfilter_smoke.slurm`. It defaults to cylindrical; append `--geometry cartesian` for the baseline. Each new submission is a fresh run.

The ten-minute job keeps width 64, two-second crops, batch size 64 per GPU, no gradient accumulation and bf16 mixed precision. It selects up to two source groups per dataset/split from the audited manifest, avoiding filesystem checks over the full corpus. A copy of those rows is saved as `smoke_manifest.csv`. Training samples 256 crops per epoch from this tiny source pool: two optimizer updates per epoch on two GPUs, for two epochs. The crop count follows the configured batch size so the training loader stays nonempty. Validation and test select two recordings per domain; test still reconstructs ten seconds in five blocks. Validation measures NFE 1/2 and test measures NFE 2.

Checks require two DDP ranks, finite nonzero gradients, changing weights, two train/validation epochs, best/last checkpoints and finite test metrics after Lightning reloads `best.ckpt`. W&B receives losses, gradients, peak allocated GPU memory, audio and test metrics. It must run online; missing credentials/network access or offline mode prevents a success report. `smoke_success.json` is written after testing and W&B finish, and the Slurm log prints `SMOKE PASSED` with the run URL. An OOM or timeout is a failure; the script does not lower the batch size to hide it. This is a short pipeline/memory check, not a reconstruction-quality result or proof of long-run stability.

Evaluation writes per-record measurements atomically and reuses completed records on rerun. It reports spectral error and SI-SDR against NFE, postfilter time, group means, dataset summaries, and one audio example per domain. Ten-second waveform metrics include the joins between two-second blocks. Postfilter timing includes feature transforms, excludes NDAC, and is not end-to-end streaming latency. Evaluation uses two fixed noise seeds; it performs no oracle allocation.

To compare training budgets, evaluate matching milestone files from the two run folders in a new `--name`, instead of comparing independently selected best checkpoints. Repeat with additional seeds before making a general claim. The released FlowDec reference has a different network, data and training budget; the newly trained Cartesian arm isolates the geometry change.

Success requires consistent held-out improvements at matched computation and listening checks. Unit-weight phase can waste capacity in quiet bins; magnitude/phase paths need not be STFT-consistent. Lower spectral error alone does not establish better sound, and neither fewer steps nor less training is guaranteed.

Implementation prepared without running local training, inference, tests or validation. Run the first short GPU job on HPC before committing a full training budget.
