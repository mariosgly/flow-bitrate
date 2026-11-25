"""Freeze, score and report saved Jamendo postfilter checkpoints."""

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
from itertools import groupby
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torchaudio.transforms import Spectrogram
from tqdm import tqdm

from .codec import ROOT, SAMPLE_RATE, SCRATCH, FlowCodec, file_hash
from .data import load_config, matched_seed, read_manifest
from .jamendo_comparison import artist_reports
from .postfilter import Postfilter
from .train_postfilter import JamendoCrops


DEFAULT_ROOT = SCRATCH / "jamendo-scores"
DEFAULT_CLAP = SCRATCH / "models" / "clap-laion-audio" / "630k-audioset-best.pt"
PROTOCOL_VERSION = 1
METRICS = ("si_sdr", "fwssnr", "logspec_mse")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


@contextmanager
def atomic_output(path, mode="w", immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode, dir=path.parent, prefix=".pending-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            if immutable:
                os.link(temporary, path)
            else:
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, data, immutable=False):
    with atomic_output(path, immutable=immutable) as handle:
        json.dump(data, handle, allow_nan=False)


@contextmanager
def record_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            # The kernel releases the lock when a worker exits, even on SIGKILL.
            yield True


def snapshot(source, directory):
    """Copy a stable version of a live checkpoint, then publish it by hash."""
    directory.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        hasher = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".pending-", delete=False) as output:
            temporary = Path(output.name)
            try:
                with source.open("rb") as handle:
                    before = os.fstat(handle.fileno())
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(chunk)
                        output.write(chunk)
                    after = os.fstat(handle.fileno())
                stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                if stamp != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    continue
                if not before.st_size or output.tell() != before.st_size:
                    continue
                output.flush()
                os.fsync(output.fileno())
                output.close()
                sha = hasher.hexdigest()
                target = directory / f"{sha}{source.suffix}"
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass
                return sha, target
            finally:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not snapshot a stable file: {source}")


def check_audio(row):
    stat = Path(row["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (int(row["file_size"]), int(row["mtime_ns"])):
        raise ValueError(f"Audio changed after the manifest audit: {row['path']}")


def make_plan(args, root, plan_path):
    config = load_config(args.config)
    manifest = Path(config["manifest"])
    manifest_sha = file_hash(manifest)
    audit = read_json(manifest.with_suffix(".summary.json"))
    if audit["manifest_sha256"] != manifest_sha or audit["audit"] != "full_decode_and_seek_v1":
        raise ValueError("Evaluation needs the audited Jamendo manifest used for training")
    rows = read_manifest(manifest, dataset="jamendo")
    if file_hash(manifest) != manifest_sha:
        raise ValueError("Manifest changed while reading it")
    split_rows = [row for row in rows if row["split"] == args.split]
    dataset = JamendoCrops(split_rows, args.split, seed=0, seconds=10, per_domain=args.limit or len(split_rows))
    if len(dataset) < 2:
        raise ValueError("FAD needs at least two eligible ten-second excerpts")
    if len({row["source_id"] for row in dataset.rows}) != len(dataset):
        raise ValueError("The cohort needs unique track IDs")
    for row in dataset.rows:
        check_audio(row)

    models, pending = {}, []
    codec_identity = None
    for path in args.checkpoints:
        path = path.expanduser().resolve()
        source = path
        if path.is_dir():
            best = path / "best.ckpt"
            source = best if best.is_file() else path / "last.ckpt"
            if not source.is_file():
                if args.require_update is not None or args.require_crops is not None:
                    raise FileNotFoundError(f"No checkpoint at the requested training budget: {path}")
                pending.append(str(source))
                print(f"Awaiting first checkpoint in {path}", flush=True)
                continue
        sha, frozen = snapshot(source, root / "snapshots")
        if sha in models:
            continue
        saved = torch.load(frozen, map_location="cpu", weights_only=False)
        if args.require_update is not None and saved["global_step"] != args.require_update:
            raise ValueError(f"{source} is at optimizer update {saved['global_step']}; requested {args.require_update}")
        hparams = saved["hyper_parameters"]
        settings = hparams["config"]
        effective_batch = settings["batch_size"] * settings["accumulate"] * hparams["provenance"]["world_size"]
        crop_slots = saved["global_step"] * effective_batch
        if args.require_crops is not None and crop_slots != args.require_crops:
            raise ValueError(f"{source} is at {crop_slots:,} scheduled crops; requested {args.require_crops:,}")
        if (settings.get("dataset") != "jamendo" or settings["codebooks"] != 6
                or settings["evaluation_seconds"] != 10 or settings.get("jamendo_smoke_steps")
                or hparams["provenance"]["manifest_sha256"] != manifest_sha):
            raise ValueError(f"Expected a full fixed-rate Jamendo checkpoint on this manifest: {source}")
        identity = saved["source_codec"]
        if codec_identity is not None and identity != codec_identity:
            raise ValueError("Checkpoints must use the same frozen codec and FlowDec noise curve")
        codec_identity = identity
        backbone = settings.get("backbone", "cyfm")
        architecture = settings.get("architecture", "unet" if backbone == "cyfm" else "ncsnpp")
        cyfm_width = settings["width"] if backbone == "cyfm" and architecture not in {
            "band-tcn", "band-attention", "ncsnpp"
        } else None
        models[sha] = dict(source=str(source), snapshot=frozen.name, sha256=sha,
                           method=f"{source.parent.name}/{source.stem}", step=saved["global_step"],
                           backbone=backbone, architecture=architecture, geometry=hparams["geometry"],
                           band_width=settings.get("band_width", 256) if architecture.startswith("band-") else None,
                           highband_loss_weight=settings.get("highband_loss_weight", 0.0),
                           ablation_arm=settings.get("ablation_arm"), start_noise=settings.get("start_noise", "original"),
                           schedule=settings.get("schedule"), effective_batch=effective_batch, crop_slots=crop_slots,
                           cyfm_width=cyfm_width,
                           cyfm_time_width=settings.get("time_embedding_dim", 4 * cyfm_width) if cyfm_width else None,
                           optimizer_recipe=settings.get("optimizer_recipe", backbone),
                           trainable_parameters=saved.get("trainable_parameters"), fit_seconds=saved.get("fit_seconds"),
                           training_gpus=hparams["provenance"]["world_size"],
                           weights=saved["evaluation_weights"], training_seed=hparams["seed"],
                           phase_loss=settings.get("phase_loss", "unit"), tau_multiplier=settings.get("tau_multiplier"),
                           phase_weight_mean=settings.get("phase_calibration", {}).get("mean_q"))
        del saved
    if not models:
        raise ValueError("No best checkpoints available yet")
    ndac = Path(config["checkpoints"]) / "ndac/ndac-75/800k/dac/weights.pth"
    if file_hash(ndac) != codec_identity["ndac"]:
        raise ValueError("The configured NDAC weights differ from the training checkpoints")
    clap_sha, clap = snapshot(args.clap_weights.resolve(), root / "snapshots")
    seeds = config["evaluation_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("The evaluation config needs distinct integer noise seeds")
    protocol = dict(version=PROTOCOL_VERSION, seconds=10, sample_rate=SAMPLE_RATE,
                    codebooks=6, embedding="LAION CLAP HTSAT-tiny unfused, one embedding per clip",
                    clap_sha256=clap_sha)
    plan = dict(protocol=protocol, manifest_sha256=manifest_sha, codec=codec_identity,
                codec_dir=config["checkpoints"], clap=clap.name, nfe=args.nfe, solver="midpoint",
                split=args.split, seeds=seeds, tracks=dataset.rows, models=list(models.values()), pending=pending)
    if args.blend_weights is not None:
        plan["blend_weights"] = args.blend_weights
    write_json(plan_path, plan, immutable=True)
    print(f"Plan {plan_path}: {len(models)} checkpoint versions, {len(dataset)} {args.split} clips, {len(pending)} pending")


def derive_plan(args, root, plan_path):
    source_path = root / "plans" / args.from_plan / "plan.json"
    plan = read_json(source_path)
    if plan["protocol"]["version"] != PROTOCOL_VERSION:
        raise ValueError("Source plan uses a different scoring protocol")
    models = [model for model in plan["models"] if model.get("ablation_arm") in args.arms]
    if sorted(model["ablation_arm"] for model in models) != sorted(args.arms):
        raise ValueError("The source plan must contain exactly one checkpoint for each requested arm")
    plan.update(models=models, pending=[], blend_weights=args.blend_weights, source_plan=args.from_plan)
    write_json(plan_path, plan, immutable=True)
    print(f"Plan {plan_path}: reusing {len(models)} checkpoints and {len(plan['tracks'])} clips "
          f"from {args.from_plan} at NFE {plan['nfe']}")


class Records:
    def __init__(self, root, plan):
        self.root = root / "records" / digest(plan["protocol"])
        self.ndac_key = "ndac-" + plan["codec"]["ndac"]
        self.nfe = plan["nfe"]
        self.solver = plan["solver"]

    def path(self, row, model=None, seed=None, weight=1.0):
        track = digest(row)
        if model is None:
            name = self.ndac_key if seed is None else "clean"
        else:
            name = f"{model['sha256']}-{self.solver}-nfe{self.nfe}-seed{seed}"
            if weight != 1.0:
                name += f"-blend{weight}"
        return self.root / track[:2] / track / f"{name}.json"


class Metrics:
    def __init__(self, clap_path, device):
        from laion_clap import CLAP_Module
        from pysepm.qualityMeasures import fwSNRseg

        self.fwssnr = fwSNRseg
        self.clap = CLAP_Module(device=str(device), enable_fusion=False, amodel="HTSAT-tiny")
        self.clap.load_ckpt(str(clap_path), verbose=False)
        self.clap.eval()
        self.spectrum = Spectrogram(n_fft=1536, hop_length=384, window_fn=torch.signal.windows.hann, power=2)

    def embed(self, audio):
        if audio.shape != (1, 1, 10 * SAMPLE_RATE) or not torch.isfinite(audio).all():
            raise ValueError("CLAP input must be one finite ten-second mono waveform")
        vector = np.asarray(self.clap.get_audio_embedding_from_data(audio.reshape(1, -1).cpu().numpy()))[0]
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError("CLAP returned an invalid embedding")
        return vector.tolist()

    def paired(self, original, output):
        x, y = original.reshape(-1).cpu(), output.reshape(-1).cpu()
        # FlowDec's SI-SDR projects onto the original waveform without mean subtraction.
        x64, y64 = x.double(), y.double()
        target = x64 * (x64.dot(y64) / x64.square().sum().clamp_min(1e-12))
        sdr = 10 * torch.log10(target.square().sum().clamp_min(1e-12) /
                               (y64 - target).square().sum().clamp_min(1e-12))
        with np.errstate(divide="ignore", invalid="ignore"):
            fwssnr = float(self.fwssnr(x.numpy(), y.numpy(), fs=SAMPLE_RATE, frameLen=0.03, overlap=0.75))
        log_x = 10 * self.spectrum(x).clamp_min(1e-8).log10()
        log_y = 10 * self.spectrum(y).clamp_min(1e-8).log10()
        return dict(si_sdr=sdr.item(), fwssnr=fwssnr if math.isfinite(fwssnr) else None,
                    logspec_mse=(log_x - log_y).square().mean().item(), embedding=self.embed(output))


@torch.inference_mode()
def score(args, root, plan, records):
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else math.inf
    if os.environ.get("SLURM_JOB_END_TIME"):
        remaining = float(os.environ["SLURM_JOB_END_TIME"]) - time.time() - 90
        deadline = min(deadline, time.monotonic() + remaining)
    device = torch.device(args.device)
    dataset = JamendoCrops(plan["tracks"], plan.get("split", "test"), seed=0, seconds=10,
                          per_domain=len(plan["tracks"]))
    assigned = range(args.shard, len(dataset), args.shards)
    metrics = None
    written = 0
    weights = plan.get("blend_weights", [1.0])
    total = len(assigned) * (2 + len(plan["models"]) * len(plan["seeds"]) * len(weights))
    with tqdm(total=total, desc=f"Shard {args.shard}/{args.shards} NFE {plan['nfe']}",
              unit="record", mininterval=15) as progress:
        for method in [None, *plan["models"]]:
            model = None
            for index in assigned:
                if time.monotonic() >= deadline:
                    print(f"Time slice ended after saving {written} records. Resubmit this shard to continue.", flush=True)
                    return
                row = dataset.rows[index]
                requests = [(seed, weight) for seed in plan["seeds"] for weight in weights] if method else [
                    ("clean", 1.0), (None, 1.0)
                ]
                paths = {request: records.path(row, method, *request) for request in requests}
                missing = [request for request, path in paths.items() if not path.is_file()]
                progress.update(len(paths) - len(missing))
                if not missing:
                    continue
                lock_path = records.path(row, method, "lock" if method else "clean")
                with record_lock(lock_path) as acquired:
                    if not acquired:
                        continue
                    still_missing = [request for request in missing if not paths[request].is_file()]
                    progress.update(len(missing) - len(still_missing))
                    missing = still_missing
                    if not missing:
                        continue
                    check_audio(row)
                    audio, track = dataset[index]
                    original = audio[None]
                    if metrics is None:
                        metrics = Metrics(root / "snapshots" / plan["clap"], device)
                    clean = ("clean", 1.0)
                    if clean in missing:
                        write_json(paths[clean], {"embedding": metrics.embed(original)})
                        missing.remove(clean)
                        written += 1
                        progress.update(1)
                    if not missing:
                        continue
                    if model is None:
                        if method:
                            model = Postfilter.load_from_checkpoint(
                                root / "snapshots" / method["snapshot"], map_location="cpu"
                            ).eval().to(device)
                            codec = model.codec
                        else:
                            codec = FlowCodec(plan["codec_dir"], load_backbone=False, configured_noise=True).to(device)
                            model = codec
                        if codec.identity != plan["codec"]:
                            raise ValueError("The current NDAC or FlowDec noise assets differ from the plan")
                    if method:
                        coarse = model.coarse_audio(original.to(device))
                    else:
                        samples = original.shape[-1]
                        _, codes = model.encode(original.to(device), samples)
                        coarse, _ = model.coarse(codes, plan["protocol"]["codebooks"], samples)
                    for seed, variants in groupby(missing, key=lambda request: request[0]):
                        if time.monotonic() >= deadline:
                            print(f"Time slice ended after saving {written} records. Resubmit this shard to continue.", flush=True)
                            return
                        output = model.enhance(coarse, plan["nfe"], matched_seed("evaluation", track, seed),
                                               plan["solver"]) if method else coarse
                        for request in variants:
                            if time.monotonic() >= deadline:
                                print(f"Time slice ended after saving {written} records. Resubmit this shard to continue.",
                                      flush=True)
                                return
                            weight = request[1]
                            waveform = output if weight == 1.0 else torch.lerp(coarse, output, weight)
                            write_json(paths[request], metrics.paired(original, waveform))
                            written += 1
                            progress.update(1)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"Shard {args.shard}/{args.shards}: saved {written} new records", flush=True)


def report(args, plan, records, directory):
    tracks, seeds = plan["tracks"], plan["seeds"]
    blending = "blend_weights" in plan
    weights = plan.get("blend_weights", [1.0])
    methods = [(None, 0.0), *((model, weight) for model in plan["models"] for weight in weights)]
    incomplete = False
    groups = [("Clean references", [(row, records.path(row, seed="clean")) for row in tracks])]
    for model, weight in methods:
        label = f"{model['method']}@{model['sha256'][:12]}" if model else "NDAC-75"
        if blending and model:
            label += f" blend={weight}"
        groups.append((label, [(row, records.path(row, model, seed, weight)) for row in tracks
                              for seed in (seeds if model else [None])]))
    for label, jobs in groups:
        missing = [row["source_id"] for row, path in jobs if not path.is_file()]
        print(f"{label}: {len(jobs) - len(missing)}/{len(jobs)} records"
              + (f"; first missing tracks: {', '.join(dict.fromkeys(missing[:8]))}" if missing else ""))
        incomplete |= bool(missing)
    if args.action == "status":
        return
    if incomplete:
        raise SystemExit("Report requires the complete cohort. Resubmit the score array for this plan.")
    if all((directory / name).is_file() for name in
           ("summary.csv", "per_clip.csv", "summary.json", "paired_artists.csv", "paired_summary.csv")):
        print(f"Reports already exist in {directory}")
        return

    clean = np.array([read_json(path)["embedding"] for _, path in groups[0][1]], dtype=np.float64)
    covariance = np.cov(clean, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clean_root = (eigenvectors * np.sqrt(eigenvalues.clip(min=0))) @ eigenvectors.T
    clean_mean = clean.mean(axis=0)

    def fad(reconstructed):
        covariance_recon = np.cov(reconstructed, rowvar=False)
        product = clean_root @ covariance_recon @ clean_root
        cross_trace = np.sqrt(np.linalg.eigvalsh((product + product.T) * 0.5).clip(min=0)).sum()
        difference = clean_mean - reconstructed.mean(axis=0)
        return float(max(0, difference @ difference + np.trace(covariance + covariance_recon) - 2 * cross_trace))

    summary, details = [], []
    for (model, weight), (label, jobs) in zip(methods, groups[1:]):
        method_seeds = seeds if model else [None]
        by_seed = {seed: [] for seed in method_seeds}
        paired = {metric: [] for metric in METRICS}
        nfe = plan["nfe"] if model else 0
        for track_index, row in enumerate(tracks):
            results = []
            for seed_index, seed in enumerate(method_seeds):
                result = read_json(jobs[track_index * len(method_seeds) + seed_index][1])
                by_seed[seed].append(result["embedding"])
                results.append(result)
                detail = dict(method=label, nfe=nfe, seed=seed, track=row["source_id"],
                              artist=row["group"], **{metric: result[metric] for metric in METRICS})
                if blending:
                    detail["blend_weight"] = weight
                details.append(detail)
            for metric in METRICS:
                values = [result[metric] for result in results if result[metric] is not None]
                if values:
                    paired[metric].append(float(np.mean(values)))
        fad_by_seed = {str(seed): fad(np.asarray(vectors, dtype=np.float64)) for seed, vectors in by_seed.items()}
        row = dict(method=label, bitrate_kbps=4.5, nfe=nfe, tracks=len(tracks),
                   fad_x100=100 * float(np.mean(list(fad_by_seed.values()))))
        if blending:
            row["blend_weight"] = weight
        row.update({metric: float(np.mean(values)) if values else None for metric, values in paired.items()})
        row.update({f"{metric}_tracks": len(values) for metric, values in paired.items()})
        summary.append({**row, "fad_by_seed": fad_by_seed, "checkpoint": model})
    for name, rows in (("summary.csv", [{k: v for k, v in row.items() if k not in {"fad_by_seed", "checkpoint"}}
                                       for row in summary]), ("per_clip.csv", details)):
        with atomic_output(directory / name, mode="w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    artist_reports(directory, details, METRICS)
    write_json(directory / "summary.json", dict(protocol=plan["protocol"], manifest_sha256=plan["manifest_sha256"],
                                                split=plan.get("split", "test"), results=summary, pending=plan["pending"]))
    print(f"Reports saved in {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "score", "status", "report"))
    parser.add_argument("--name", required=True, help="Name of an immutable evaluation plan")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Shared cache and report directory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/jamendo.json")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoints", type=Path, nargs="+", help="Checkpoint files or run directories containing best.ckpt or last.ckpt")
    source.add_argument("--from-plan", help="Reuse the frozen checkpoint versions, cohort, seeds and NFE of this plan")
    parser.add_argument("--arms", type=int, nargs="+", help="Select ablation arms from --from-plan")
    parser.add_argument("--blend-weights", type=float, nargs="+", help="Score NDAC + weight * (enhanced - NDAC)")
    parser.add_argument("--clap-weights", type=Path, default=DEFAULT_CLAP)
    parser.add_argument("--nfe", type=int, help="Inference budget, 1 for Euler or an even midpoint budget (default 6)")
    parser.add_argument("--split", choices=("val", "test"), help="Cohort to use when planning (default test)")
    parser.add_argument("--limit", type=int, help="Use a fixed subset per domain when planning")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--require-update", type=int, help="Require this optimizer update for all checkpoints")
    budget.add_argument("--require-crops", type=int, help="Require this many scheduled crop slots for all checkpoints")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=int, default=0, help="Optional time limit per score invocation")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {"", ".", ".."}:
        parser.error("--name must be a folder name")
    root = args.root.resolve()
    directory = root / "plans" / args.name
    plan_path = directory / "plan.json"
    if args.action == "plan":
        if plan_path.exists():
            parser.error(f"Plan already exists. Reuse it for scoring or use a new name: {plan_path}")
        if args.blend_weights is not None and (
                len(set(args.blend_weights)) != len(args.blend_weights)
                or any(not 0 < weight <= 1 for weight in args.blend_weights)):
            parser.error("Use distinct --blend-weights in (0, 1]; NDAC is already included at weight 0")
        if args.from_plan is not None:
            if Path(args.from_plan).name != args.from_plan or args.from_plan in {"", ".", ".."}:
                parser.error("--from-plan must be a folder name")
            if not args.arms or len(set(args.arms)) != len(args.arms) or args.blend_weights is None:
                parser.error("Use distinct --arms and set --blend-weights with --from-plan")
            if any(value is not None for value in (args.nfe, args.split, args.limit, args.require_update, args.require_crops)):
                parser.error("--from-plan inherits the checkpoint budgets, cohort and NFE of the source plan")
            derive_plan(args, root, plan_path)
            return
        if args.arms is not None:
            parser.error("--arms requires --from-plan")
        args.nfe = 6 if args.nfe is None else args.nfe
        args.split = args.split or "test"
        if not args.checkpoints or args.nfe < 1 or (args.nfe != 1 and args.nfe % 2):
            parser.error("Pass checkpoints and either NFE 1 or a positive even NFE")
        if args.limit is not None and args.limit < 2:
            parser.error("--limit must be at least two")
        make_plan(args, root, plan_path)
    else:
        if any(value is not None for value in (args.require_update, args.require_crops, args.from_plan,
                                                args.arms, args.blend_weights)):
            parser.error("Checkpoint budget guards and blend settings apply when creating a plan")
        plan = read_json(plan_path)
        if plan["protocol"]["version"] != PROTOCOL_VERSION:
            parser.error("This plan uses a different scoring protocol")
        records = Records(root, plan)
        if args.action == "score":
            if not 0 <= args.shard < args.shards or args.max_seconds < 0:
                parser.error("Need 0 <= --shard < --shards and a nonnegative --max-seconds")
            score(args, root, plan, records)
        else:
            report(args, plan, records, directory)


if __name__ == "__main__":
    main()
