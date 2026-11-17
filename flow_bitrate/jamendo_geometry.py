"""Calibrate phase weights on training crops and measure chord cancellation on held-out validation music."""

import argparse
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

from .codec import ROOT, SCRATCH, FlowCodec, file_hash
from .data import load_config, matched_seed, read_manifest
from .metrics import si_sdr, spectral_loss
from .oracle import write_csv
from .phase_geometry import PHASE_FLOOR, matched_draw, noise_scale, phase_weight
from .score_jamendo import check_audio, digest, write_json
from .sweep import save_audio
from .train_postfilter import JamendoCrops


def spectra(codec, waveform, device, codebooks):
    audio = waveform.unsqueeze(0).to(device)
    samples = audio.shape[-1]
    _, codes = codec.encode(audio, samples)
    coarse, _ = codec.coarse(codes, codebooks, samples)
    condition, target, info = codec.preprocess(coarse, audio)
    return audio, coarse, condition, target, info


def calibrate(codec, train, count, device, tau, codebooks):
    indices = sorted(range(len(train)), key=lambda i: matched_seed("phase-calibration", train.rows[i]["source_id"]))
    selected = indices[:count]
    total, bins = 0.0, 0
    for index in selected:
        check_audio(train.rows[index])
        waveform, seed = train[index]
        _, _, condition, target, (_, frames, _) = spectra(codec, waveform, device, codebooks)
        start, time = matched_draw(condition, codec.sigma, [seed])
        a0, a1 = start[..., :frames].abs(), target[..., :frames].abs()
        weight = phase_weight(a0 + time[:, None, None, None] * (a1 - a0), tau)
        total += weight.sum(dtype=torch.float64).item()
        bins += weight.numel()
    return dict(mean_q=total / bins, bins=bins, tracks=len(selected), crop_seed=train.seed,
                tracks_sha256=digest([train.rows[i]["source_id"] for i in selected]))


def fractions(totals):
    bins, cancelled, energy, cancelled_energy = totals
    return dict(bins=int(bins), cancelled_bins=int(cancelled), clean_energy=energy,
                cancelled_clean_energy=cancelled_energy, cancelled_bin_fraction=cancelled / bins if bins else None,
                cancelled_energy_fraction=cancelled_energy / energy if energy else None)


def audit(codec, validation, seeds, device, tau, codebooks, audio_tracks, directory):
    records, listening = [], []
    totals = defaultdict(lambda: [0, 0, 0., 0.])
    padded = zero_pairs = 0
    for index, row in enumerate(validation.rows):
        check_audio(row)
        waveform, track = validation[index]
        audio, coarse, condition, target, (scale, frames, length) = spectra(codec, waveform, device, codebooks)
        clean = target[..., :frames]
        coarse_spectrum = condition[..., :frames]
        weak_coarse, strong_clean = coarse_spectrum.abs() < tau, clean.abs() >= tau
        groups = {"all": torch.ones_like(weak_coarse), "weak_coarse_strong_clean": weak_coarse & strong_clean,
                  "weak_coarse_weak_clean": weak_coarse & ~strong_clean,
                  "strong_coarse_strong_clean": ~weak_coarse & strong_clean,
                  "strong_coarse_weak_clean": ~weak_coarse & ~strong_clean}
        # Undo feature compression and per-excerpt gain. Count both sides of the real-audio FFT.
        energy = (codec.features.compress.invert(clean).abs().double() * scale.item()).square()
        energy[:, :, 1:-1] *= 2
        for sample_seed in seeds:
            start, _ = matched_draw(condition, codec.sigma, [matched_seed("evaluation", track, sample_seed)])
            padded += start[..., frames:].numel()
            start = start[..., :frames]
            denominator = start.abs() + clean.abs()
            eligible = denominator > 0
            zero_pairs += (~eligible).sum().item()
            cancelled = (start + clean).abs() < 0.5 * denominator
            for name, group in groups.items():
                include = group & eligible
                affected = include & cancelled
                counts = [include.sum().item(), affected.sum().item(),
                          (energy * include).sum().item(), (energy * affected).sum().item()]
                totals[name] = [x + y for x, y in zip(totals[name], counts)]
                records.append(dict(track=track, artist=row["group"], noise_seed=sample_seed,
                                    group=name, **fractions(counts)))
            if index < audio_tracks and sample_seed == seeds[0]:
                examples = directory / "audio" / f"{index:03d}"
                save_audio(examples / "clean.wav", audio.flatten().cpu())
                save_audio(examples / "ndac.wav", coarse.flatten().cpu())
                for name, mask in (("cancelled", cancelled), ("weak_coarse_strong_clean", groups["weak_coarse_strong_clean"])):
                    mask = mask & (coarse_spectrum.abs() > 0)
                    swapped = torch.where(mask, torch.polar(clean.abs(), coarse_spectrum.angle()), clean)
                    output = codec.features.invert(swapped, orig_length=length) * scale
                    save_audio(examples / f"{name}_coarse_phase.wav", output.flatten().cpu())
                    listening.append(dict(example=index, track=track, artist=row["group"], noise_seed=sample_seed,
                                          phase_substitution=name, replaced_bins=mask.sum().item(),
                                          spectral=spectral_loss(audio, output).item(), si_sdr=si_sdr(audio, output).item()))
    write_csv(directory / "cancellation_by_track.csv", records)
    if listening:
        write_csv(directory / "phase_substitution.csv", listening)
    return dict(validation_tracks=len(validation), noise_seeds=seeds, padded_bins_excluded=padded,
                zero_endpoint_pairs_excluded=zero_pairs, groups={k: fractions(v) for k, v in totals.items()})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/jamendo.json")
    parser.add_argument("--output", type=Path, default=SCRATCH / "jamendo-geometry")
    parser.add_argument("--calibration-tracks", type=int, default=128)
    parser.add_argument("--validation-tracks", type=int, default=32)
    parser.add_argument("--audio-tracks", type=int, default=4)
    parser.add_argument("--tau-multiplier", type=float, default=1.)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.calibration_tracks, args.validation_tracks) < 1 or args.audio_tracks < 0:
        parser.error("Use positive cohort sizes and a nonnegative audio count")
    if not math.isfinite(args.tau_multiplier) or args.tau_multiplier <= 0:
        parser.error("--tau-multiplier must be positive and finite")
    config = load_config(args.config)
    if config.get("dataset") != "jamendo" or config["codebooks"] != 6:
        parser.error("Use the six-codebook Jamendo config")
    manifest = Path(config["manifest"])
    manifest_sha = file_hash(manifest)
    if config.get("require_data_audit", True):
        report = json.loads(manifest.with_suffix(".summary.json").read_text())
        if (report["manifest_sha256"] != manifest_sha or report["audit"] != "full_decode_and_seek_v1"
                or not report.get("loader_check")):
            raise ValueError("The manifest must match its completed decode audit")
    rows = read_manifest(manifest, dataset="jamendo")
    train = JamendoCrops(rows, "train", seed=0)
    valid = JamendoCrops(rows, "val", seed=0, per_domain=args.validation_tracks, seconds=config["evaluation_seconds"])
    codec = FlowCodec(config["checkpoints"], load_backbone=False, configured_noise=True).eval().to(args.device)
    tau = noise_scale(codec.sigma, args.tau_multiplier)
    settings = dict(version=1, manifest_sha256=manifest_sha, codec=codec.identity, codebooks=config["codebooks"],
                    floor=PHASE_FLOOR, tau_multiplier=args.tau_multiplier)
    protocol = dict(**settings, calibration_tracks=args.calibration_tracks, validation_tracks=args.validation_tracks,
                    audio_tracks=args.audio_tracks, evaluation_seeds=config["evaluation_seeds"],
                    evaluation_seconds=config["evaluation_seconds"])
    output = args.output.resolve()
    if output.exists():
        if (output / "summary.json").is_file() and json.loads((output / "summary.json").read_text())["protocol"] == protocol:
            print(f"Completed geometry audit and calibration: {output}")
            return
        raise FileExistsError(f"Choose a new --output for this geometry audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".geometry-") as tmp:
        staging = Path(tmp) / "results"
        staging.mkdir()
        calibration = calibrate(codec, train, args.calibration_tracks, args.device, tau, config["codebooks"])
        write_json(staging / "calibration.json", {**settings, **calibration})
        result = audit(codec, valid, config["evaluation_seeds"], args.device, tau, config["codebooks"], args.audio_tracks, staging)
        write_json(staging / "summary.json", dict(protocol=protocol, **result))
        staging.rename(output)
    print(f"Mean training phase weight {calibration['mean_q']:.6f}. Validation results: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
