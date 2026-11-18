# Flow matching on a disk for music reconstruction

Every run receives the same six NDAC codebooks at nominal 4.5 kbps. The decoder sees the coarse audio and starts from its compressed complex spectrum plus FlowDec's frequency-dependent Gaussian noise. The questions are whether magnitude should determine the importance of a phase error and whether it should determine the path the decoder learns. Music quality at a fixed budget and the number of calls needed to reach a given quality are separate outcomes.

## What the disk means

A complex coefficient has radius $A$ and angle $θ$. A turn of $θ$ radians sweeps a distance $A|θ|$ on a circle of radius $A$. All angles coincide at the center. The corresponding squared distance for a small movement is

$$ds^2=dA^2+A^2 dθ^2.$$

The shortest path between two values in this geometry is the straight chord. The project already learns that path in real and imaginary coordinates. `--geometry disk` selects this existing Cartesian implementation, including its checkpoint format. The radius can take every magnitude produced by the feature extractor.

The cylinder uses $ds^2=dA^2+dθ^2$. Between equal-magnitude endpoints it stays at that magnitude and rotates along the shortest arc. For endpoints separated by 120°, the disk chord retains half their magnitude at its midpoint. Here magnitude means $A=0.33|X|^{0.3}$, where $X$ is the linear STFT after the common coarse-waveform normalization. Inverse compression turns half the feature magnitude into about 10% of the original linear STFT magnitude. The cylinder retains 100% in this example. These intermediate states supply training examples for the velocity predictor.

| Launcher arm | Training path | Velocity loss on genuine frames | Comparison |
| --- | --- | --- | --- |
| `disk` | Straight Cartesian chord | Real and imaginary MSE | Existing Cartesian CyFM EMA control |
| `cylinder` | Linear magnitude and shortest phase arc | Radial MSE plus angular MSE | Existing cylindrical CyFM EMA control |
| `disk-loss` | Same cylindrical arc | Radial MSE plus $A_t^2$ times angular MSE | Does the ordinary disk's tangent-space error help on the same path? |
| `weighted` | Same cylindrical arc | Radial MSE plus a calibrated, bounded phase coefficient times angular MSE | Does prioritizing reliable phase help at matched average angular weight? |
| `warped` | Shortest path of the amplitude-dependent disk below | Same real and imaginary MSE as `disk` | Does changing only the training path help? |

`disk-loss` uses the exact Euclidean tangent norm on the compressed coefficient. Its angular scale can be above one and its overall mean can differ from the cylinder's. Also, the network still receives the cylinder's phase embedding near zero. `weighted` keeps a small positive floor and calibrates its mean. For both loss experiments, padded frames keep the original unit angular weight and the radial term keeps its original weight everywhere.

## How the new bridge works

For `warped`, the angular distance per radian grows like $A$ near silence and approaches a constant $τ_f$ at high magnitude.

$$ds^2=dA^2+\frac{τ_f^2 A^2}{τ_f^2+A^2}dθ^2.$$

The training code selects the rotation $ω$ minimizing $|e^{-iω}z_1-z_0|^2+τ_f^2ω^2$. In the rotating frame the path is a straight line. Translating it back gives

$$z_t=e^{itω}\big[(1-t)z_0+t e^{-iω}z_1\big],\qquad
\dot z_t=iωz_t+e^{itω}(e^{-iω}z_1-z_0).$$

For opposite phases with equal magnitude $A$, the cost of rotating relative to taking the chord changes by $τ_f^2ω^2-4A^2\sin^2(ω/2)$. Up to $A=τ_f$ the optimal path crosses the origin. Above that threshold the path bends around it. At an exact tie the implementation picks the positive rotation. The vectorized one-dimensional solve runs during training. The decoder predicts Cartesian velocity and uses the ordinary Cartesian sampler at inference. Quiet bins and loud bins receive the same number of network calls.

For `weighted`, the current cylindrical bridge magnitude supplies the weight

$$q=0.05+0.95\frac{A_t^2}{A_t^2+τ_f^2},\qquad w=q/\bar q.$$

The default $τ_f$ is FlowDec's noise scale for that frequency, floored at $10^{-6}$ in feature units. A positive weight determined by the observable state preserves the population squared-error velocity target. With finite network capacity it changes where the network spends modeling effort. Both the noise scale as a measure of phase reliability and the floor of 0.05 are experimental choices. The weighted checkpoint embeds its calibration. Evaluation and explicit resume can use it directly.

## 1. Measure cancellation and calibrate

Run from the project root on HPC using the same audited Jamendo manifest and frozen weights as the five original runs.

```bash
bash scripts/submit_jamendo_geometry.sh audit
```

This uses one GPU. It estimates `mean_q` once from 128 deterministically selected two-second **training** crops, using their training noise and flow times and counting only real frames. It separately audits up to 32 fixed ten-second **validation** tracks with both evaluation noise seeds. The default output directory is `/scratch/mg8253/flow-bitrate/jamendo-geometry`.

- `calibration.json` stores the mean, training cohort fingerprint, manifest hash, noise and codec identities, floor and transition scale.
- `summary.json` and `cancellation_by_track.csv` report the share of endpoint pairs with $D=|z_0+z_1|/(|z_0|+|z_1|)<0.5$, by bin count and clean linear-STFT energy. The denominator is the sum of endpoint magnitudes. Both-zero pairs and padded frames have separate exclusion counts. Energy reverses compression and gain normalization, then accounts for the two sides of the real-audio spectrum.
- The four disjoint coarse/clean magnitude groups use $τ_f$ as the threshold. In particular, `weak_coarse_strong_clean` captures detail that becomes strong in the clean music. Within each group `cancelled_energy_fraction` is the fraction of that group's energy on strongly cancelling paths. Divide its `cancelled_clean_energy` by the `all` group's `clean_energy` for its contribution to the entire cohort.
- `audio/` and `phase_substitution.csv` cover the first four examples at the first noise seed. Listen to clean, NDAC and two diagnostic resyntheses with clean magnitude and coarse phase substituted only in the named bins. Bins with undefined coarse phase keep clean phase. These clips use the clean target to diagnose phase sensitivity.

A high bin count alongside a small energy share suggests examining quiet tails. A substantial energy share in weak-coarse/strong-clean bins makes missing attacks and harmonics a priority for the listening comparison. The substituted-phase audio helps distinguish these cases. To try another transition scale, give the audit a distinct output, for example `audit --tau-multiplier 2 --output /scratch/mg8253/flow-bitrate/jamendo-geometry-tau2`. Pass its `calibration.json` to `weighted --phase-calibration ...` and use `warped --tau-multiplier 2` for the corresponding bridge.

## 2. Train the matched arms

Use **CyFM EMA** controls trained with the same seed, effective batch and update count. The launcher defaults to seed 0, two GPUs and the original Jamendo optimizer, crop and noise protocols. The FlowDec NCSN++ EMA run remains a separate reference for complete decoder recipes.

```bash
bash scripts/submit_jamendo_geometry.sh weighted --smoke
bash scripts/submit_jamendo_geometry.sh warped --smoke
```

After the audit has produced the calibration and the short runs finish, these commands submit full runs. Each run keeps one rolling `last.ckpt`, saved after validation every 1,000 optimizer updates and at a graceful stop if the step has advanced.

```bash
bash scripts/submit_jamendo_geometry.sh weighted
bash scripts/submit_jamendo_geometry.sh warped
bash scripts/submit_jamendo_geometry.sh disk-loss
```

If compute is limited, start with `weighted` against `cylinder`. Prioritize `warped` against `disk` when the audit shows material cancellation or the phase substitutions are audible. Use `disk-loss` against `cylinder` for the direct test of the ordinary disk's phase penalty. The warped and disk-loss arms can start independently of calibration.

Existing seed-0 control checkpoints can be scored immediately. Report the saved `global_step` for each model. To repeat a result, pass `--seed 1` to **both** the candidate and its matching control. Reuse the fixed seed-0 calibration and evaluation cohort.

Resume a job from its latest saved update with the original arm, seed and settings. For example, use `bash scripts/submit_jamendo_geometry.sh weighted --resume /path/to/run/last.ckpt`. The standard eight-hour allocation applies. Checkpoints appear under `/scratch/mg8253/flow-bitrate/runs/`.

## 3. Score quality and listen

Set the following shell variables to the checkpoints you want to compare. The examples use the existing controls' `best.ckpt` and the new runs' `last.ckpt`.

```bash
runs=/scratch/mg8253/flow-bitrate/runs
disk_ckpt="$runs/<cartesian-EMA-run>/best.ckpt"
cylinder_ckpt="$runs/<cylindrical-EMA-run>/best.ckpt"
weighted_ckpt="$runs/<state-weighted-EMA-run>/last.ckpt"
warped_ckpt="$runs/<warped-disk-EMA-run>/last.ckpt"
disk_loss_ckpt="$runs/<disk-loss-EMA-run>/last.ckpt"

sbatch slurm/postfilter_evaluate.slurm --name disk-sweep-seed0 --cohort-seed 0 \
  --checkpoints "$disk_ckpt" "$cylinder_ckpt" "$weighted_ckpt" "$warped_ckpt" "$disk_loss_ckpt"
```

The sweep uses 1, 2, 4, 6, 12, 24 and 48 network evaluations, with the same ten-second test excerpts, two noise seeds and NDAC baseline. One call uses Euler. Six calls use three midpoint steps. Read `quality_vs_nfe.png`, timing in `metrics.csv`, and `paired_summary.csv` and `paired_artists.csv` under the named run. Paired differences are candidate minus reference. Lower spectral error and higher SI-SDR favor the candidate. Seeds are averaged within track, tracks within artist, and artists equally in the summary. The 95% interval resamples artists. Check `summary.json` for checkpoint steps and hashes.

Use two checkpoints per listening comparison. This example produces a blinded packet for eight held-out tracks at two budgets.

```bash
sbatch slurm/postfilter_evaluate.slurm --name listen-warped-seed0 --cohort-seed 0 \
  --nfe 6 24 --per-domain 8 --listening-tracks 8 --checkpoints "$disk_ckpt" "$warped_ckpt"
sbatch slurm/postfilter_evaluate.slurm --name listen-weighted-seed0 --cohort-seed 0 \
  --nfe 6 24 --per-domain 8 --listening-tracks 8 --checkpoints "$cylinder_ckpt" "$weighted_ckpt"
```

Give listeners only the `listener/` folder in the printed packet directory. Each shuffled trial contains a clean reference, a named NDAC anchor and a randomized A/B pair from the same track, budget and noise draw. WAVs retain their original gain. Listen for sustained harmonics, cymbal tails and attacks. The packet parent holds the private answer key and checkpoint identities. Decode a completed ballot in the project environment with

```bash
python -m flow_bitrate.jamendo_comparison /path/to/listening-packet /path/to/completed-ballot.csv \
  --output /path/to/listener-01-artist-votes.csv
```

For whole-cohort CLAP FAD, logSpecMSE, fwSSNR and SI-SDR, use the [resumable scorer](jamendo-scoring.md) with separate immutable plans at `--nfe 6` and `--nfe 24` or `48`. Its report also writes artist-paired differences for the reference-based metrics. FAD is computed over the cohort. A reproducible listening gain at high NFE answers the quality question even when inference takes the same time.
