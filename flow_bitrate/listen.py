"""Export listening comparisons without repeating the measurement grid."""

import argparse
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

from .codec import BLOCK_SAMPLES, CODEBOOKS, ROOT, SAMPLE_RATE, FlowCodec, file_hash
from .data import load_config, read_audio
from .prepare import block_seed
from .sweep import commit_file, diagnostic_rows, save_audio, sweep_settings


def valid_wav(path, frames):
    try:
        info = sf.info(path)
        return info.frames == frames and info.samplerate == SAMPLE_RATE and info.channels == 1
    except (OSError, RuntimeError):
        return False


def copy_wav(source, destination):
    temporary = destination.with_suffix(".wav.part")
    shutil.copyfile(source, temporary)
    commit_file(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs/diagnostic.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--all", action="store_true", help="Export every recording instead of one per domain")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = sweep_settings(config)
    rows = diagnostic_rows(config)
    cache = Path(config["cache"])
    output = Path(config["output"]) / config["name"] / "audio"
    output.mkdir(parents=True, exist_ok=True)
    nfes = list(dict.fromkeys([6 if 6 in config["nfe"] else config["nfe"][0], config["reference_nfe"]]))
    depths = [CODEBOOKS[0], CODEBOOKS[-1]]
    counts, sections, codec = defaultdict(int), [], None
    with torch.inference_mode():
        for index, row in enumerate(rows):
            if not args.all and counts[row["domain"]] >= 1:
                continue
            counts[row["domain"]] += 1
            record_path = cache / f"record_{index:05d}.pt"
            if not record_path.exists():
                raise FileNotFoundError(f"{record_path.name} is missing; finish that recording's sweep first")
            record = torch.load(record_path, map_location="cpu", weights_only=False)
            if (record["record"] != row or record["metadata"]["settings"] != settings
                    or record.get("status") != "complete"):
                raise ValueError(f"Incomplete or different measurement settings in {record_path}")
            if record["source"]["sha256"] != file_hash(row["path"]):
                raise ValueError(f"Source audio changed since measurement: {row['path']}")
            blocks = record["blocks"]
            length = blocks * BLOCK_SAMPLES
            cache_audio = cache / "audio" / f"record_{index:05d}"
            destination = output / f"record_{index:05d}"
            destination.mkdir(parents=True, exist_ok=True)
            provenance = {"record": row, "source": record["source"], "metadata": record["metadata"],
                          "seed": config["evaluation_seeds"][0], "codebooks": depths, "nfe": nfes}
            stamp = destination / "source.json"
            if stamp.exists() and json.loads(stamp.read_text()) != provenance:
                raise ValueError(f"Different listening provenance in {stamp}; use a fresh output directory")
            start = round(row["start_seconds"] * SAMPLE_RATE)
            audio = None
            if not valid_wav(cache_audio / "original.wav", length):
                audio = read_audio(row["path"])[..., start:start + length]
                if audio.shape[-1] != length:
                    raise ValueError(f"Source is shorter than the cached blocks: {row['path']}")
                save_audio(cache_audio / "original.wav", audio[0])
            if not valid_wav(destination / "original.wav", length):
                copy_wav(cache_audio / "original.wav", destination / "original.wav")
            for depth in depths:
                variants = {f"ndac_k{depth}": 0, **{f"flow_k{depth}_nfe{nfe}": nfe for nfe in nfes}}
                for name in variants:
                    if not valid_wav(destination / f"{name}.wav", length) and valid_wav(cache_audio / f"{name}.wav", length):
                        copy_wav(cache_audio / f"{name}.wav", destination / f"{name}.wav")
                missing = [name for name in variants if not valid_wav(destination / f"{name}.wav", length)]
                for block in range(blocks):
                    needed = [name for name in missing if not valid_wav(
                        cache_audio / "blocks" / f"{name}_block{block:03d}.wav", BLOCK_SAMPLES)]
                    if not needed:
                        continue
                    if codec is None:
                        print("Missing listening waveforms: loading weights to regenerate only those examples.", flush=True)
                        codec = FlowCodec(config["checkpoints"], config.get("flow_checkpoint")).to(args.device).requires_grad_(False)
                    if codec.identity != record["metadata"]["codec"]:
                        raise ValueError("Current weights/waveform policy differ from the measurement cache")
                    coarse_path = cache_audio / "blocks" / f"ndac_k{depth}_block{block:03d}.wav"
                    if valid_wav(coarse_path, BLOCK_SAMPLES):
                        coarse = torch.from_numpy(sf.read(coarse_path, dtype="float32")[0])[None, None].to(args.device)
                    else:
                        if audio is None:
                            audio = read_audio(row["path"])[..., start:start + length]
                        original = audio[..., block * BLOCK_SAMPLES:(block + 1) * BLOCK_SAMPLES][None].to(args.device)
                        _, codes = codec.encode(original)
                        coarse = codec.coarse(codes, depth)[0]
                        save_audio(coarse_path, coarse[0, 0].cpu())
                    file_seed = block_seed(row["path"], 0, config["evaluation_seeds"][0])
                    seed = block_seed("", start + block * BLOCK_SAMPLES, file_seed)
                    for name in needed:
                        nfe = variants[name]
                        decoded = codec.enhance(coarse, seed, nfe // 2) if nfe else coarse
                        save_audio(cache_audio / "blocks" / f"{name}_block{block:03d}.wav", decoded[0, 0].cpu())
                    print(f"Listening record {index}: k={depth}, block {block + 1}/{blocks} saved", flush=True)
                for name in missing:
                    chunks = [torch.from_numpy(sf.read(cache_audio / "blocks" / f"{name}_block{block:03d}.wav",
                                                       dtype="float32")[0]) for block in range(blocks)]
                    save_audio(cache_audio / f"{name}.wav", torch.cat(chunks))
                    copy_wav(cache_audio / f"{name}.wav", destination / f"{name}.wav")
            temporary = destination / "source.json.part"
            temporary.write_text(json.dumps(provenance, indent=2) + "\n")
            commit_file(temporary, stamp)
            relative = destination.name
            title = html.escape(f"{row['domain']} — {row['group']} — {Path(row['path']).name}")
            cells = []
            for depth in depths:
                names = [f"ndac_k{depth}"] + [f"flow_k{depth}_nfe{nfe}" for nfe in nfes]
                cells.append(f'<tr><th>{depth} codebooks</th>' + ''.join(
                    f'<td>{name}<br><audio controls preload="none" src="{relative}/{name}.wav"></audio></td>'
                    for name in names) + '</tr>')
            sections.append(f'<section><h2>{title}</h2><p>Original:</p>'
                            f'<audio controls preload="none" src="{relative}/original.wav"></audio>'
                            '<table>' + ''.join(cells) + '</table></section>')
            print(f"Listening comparison ready: {destination}", flush=True)
    page = ('<!doctype html><meta charset="utf-8"><title>Listening comparisons</title>'
            '<style>body{font:16px system-ui;margin:30px;line-height:1.5}td,th{padding:12px;text-align:left}'
            'section{margin:40px 0}audio{width:280px}</style><h1>Listening comparisons</h1>'
            '<p>Same source excerpt, low/high bitrates, and one fixed evaluation noise seed. '
            'Audio is not independently peak-normalized. Listen for lost or invented detail, transients, '
            'speech changes and joins every two seconds. These examples do not replace a listening study.</p>' + ''.join(sections))
    temporary = output / "index.html.part"
    temporary.write_text(page)
    commit_file(temporary, output / "index.html")
    print(f"Open {output / 'index.html'}. Measurement caches were not changed.")


if __name__ == "__main__":
    main()
