"""Paired artist reports and blinded A/B listening packets for the Jamendo experiments."""

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from .data import matched_seed
from .oracle import interval, write_csv


def artist_reports(directory, records, metrics):
    methods = defaultdict(lambda: defaultdict(dict))
    for row in records:
        methods[row["method"], row["nfe"]][row["artist"], row["track"]][row["seed"]] = row
    details, summary = [], []
    for budget in sorted({nfe for _, nfe in methods if nfe}):
        available = [arm for arm in methods if arm[1] in (0, budget)]
        for reference, candidate in combinations(available, 2):
            by_artist = defaultdict(list)
            for track, left in methods[reference].items():
                right = methods[candidate].get(track)
                if not right:
                    continue
                seeds = left.keys() & right.keys() if reference[1] and candidate[1] else (right if candidate[1] else left)
                for metric in metrics:
                    differences = []
                    for seed in seeds:
                        a = left[seed if reference[1] else None][metric]
                        b = right[seed if candidate[1] else None][metric]
                        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                            differences.append(b - a)
                    if differences:
                        by_artist[metric, track[0]].append(np.mean(differences))
            pair = dict(reference=reference[0], candidate=candidate[0], nfe=budget)
            for metric in metrics:
                artists = [dict(**pair, metric=metric, artist=artist, tracks=len(values),
                                difference=float(np.mean(values)))
                           for (name, artist), values in by_artist.items() if name == metric]
                details.extend(artists)
                means = np.array([row["difference"] for row in artists])
                bounds = interval(means, 0) or [None, None]
                summary.append(dict(**pair, metric=metric, artists=len(artists), tracks=sum(r["tracks"] for r in artists),
                                    difference=means.mean() if len(means) else None, ci_low=bounds[0], ci_high=bounds[1]))
    write_csv(directory / "paired_artists.csv", details)
    write_csv(directory / "paired_summary.csv", summary)


def blind_packet(output, rows, sources, models, budgets, noise_seed):
    labels = [model["name"] for model in models]
    identity = [(row["source_id"], row["group"]) for row in rows]
    seed = matched_seed("blind-packet-v1", identity, [m["sha256"] for m in models], budgets, noise_seed)
    packet = output / f"listening-{seed:016x}"
    if packet.is_dir():
        return packet
    rng = random.Random(seed)
    trials = [(index, nfe) for index in range(len(rows)) for nfe in budgets]
    rng.shuffle(trials)
    key, ballot = [], []
    with tempfile.TemporaryDirectory(dir=output, prefix=".listening-") as tmp:
        staging = Path(tmp) / "packet"
        public = staging / "listener"
        public.mkdir(parents=True)
        for number, (index, nfe) in enumerate(trials, start=1):
            trial = f"trial-{number:03d}"
            folder = public / trial
            folder.mkdir()
            control = sources[index, labels[0]]
            for name in ("reference", "ndac"):
                shutil.copyfile(control / f"{name}.wav", folder / f"{name}.wav")
            order = rng.sample(labels, 2)
            for letter, label in zip(("A", "B"), order):
                shutil.copyfile(sources[index, label] / f"nfe{nfe}.wav", folder / f"{letter}.wav")
            key.append(dict(trial=trial, track=rows[index]["source_id"], artist=rows[index]["group"],
                            nfe=nfe, noise_seed=noise_seed, A=order[0], B=order[1]))
            ballot.append(dict(trial=trial, choice="", notes=""))
        write_csv(public / "ballot.csv", ballot)
        write_csv(staging / "answer_key.csv", key)
        write_csv(staging / "checkpoints.csv", [dict(method=m["name"], sha256=m["sha256"],
                                                     checkpoint=m["checkpoint"]) for m in models])
        (public / "instructions.txt").write_text(
            "For each trial, listen to reference.wav and the coarse codec anchor ndac.wav. "
            "Which of A.wav and B.wav better reproduces the music in reference.wav? "
            "Use the same playback level. Enter A, B, or tie in ballot.csv and add any listening notes.\n")
        staging.rename(packet)
    return packet


def main():
    parser = argparse.ArgumentParser(description="Decode one completed Jamendo listening ballot and summarize it by artist")
    parser.add_argument("packet", type=Path)
    parser.add_argument("ballot", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="CSV for this listener's artist-level preferences")
    args = parser.parse_args()
    with (args.packet / "answer_key.csv").open(newline="") as handle:
        key = {row["trial"]: row for row in csv.DictReader(handle)}
    votes = defaultdict(list)
    seen = set()
    with args.ballot.open(newline="") as handle:
        for row in csv.DictReader(handle):
            trial = row["trial"]
            choice = row["choice"].strip().upper()
            if trial not in key or trial in seen or choice not in {"A", "B", "TIE"}:
                parser.error(f"Expected a unique trial and a choice of A, B or tie: {trial}")
            seen.add(trial)
            assignment = key[trial]
            a, b = sorted((assignment["A"], assignment["B"]))
            preferred = assignment[choice] if choice != "TIE" else None
            votes[assignment["artist"], int(assignment["nfe"]), a, b].append(preferred)
    if seen != key.keys():
        parser.error("The ballot must cover every trial in this packet")
    results = [dict(artist=artist, nfe=nfe, first=a, second=b, first_wins=v.count(a), second_wins=v.count(b),
                    ties=v.count(None), trials=len(v)) for (artist, nfe, a, b), v in sorted(votes.items())]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, results)
    print(f"Saved artist-level preferences to {args.output}")


if __name__ == "__main__":
    main()
