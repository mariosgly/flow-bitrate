"""Local figures for the frozen-codec diagnostic. No model inference here."""

import html
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .codec import CODEBOOKS
from .sweep import commit_file


def paired_gain(rows, seed):
    groups = defaultdict(list)
    for row, gain in rows:
        groups[(row["domain"], row["group"])].append(gain)
    values = np.array([np.mean(group) for group in groups.values()])
    if len(values) < 2:
        return values.mean(), np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return values.mean(), low, high


def plot_report(report, grid, measurements, output):
    config, summary = report["config"], report["summary"]
    domains = ["all"] + sorted({row["domain"] for row in grid})
    nfes = config["nfe"] + [config["reference_nfe"]]
    colors = plt.get_cmap("tab10").colors
    directory = output / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    figures = []

    def finish(fig, filename, title, explanation):
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, .04, 1, .94))
        temporary = directory / (filename + ".part")
        fig.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        path = directory / filename
        commit_file(temporary, path)
        figures.append((path, title, explanation))

    def compute_axis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(nfes, [str(nfe) for nfe in nfes])
        ax.set_xlabel("Flow network evaluations (NFE)")
        ax.grid(alpha=.2)

    fig, axes = plt.subplots(2, len(domains), figsize=(4.4 * len(domains), 7), squeeze=False)
    for column, domain in enumerate(domains):
        for row_index, (metric, label) in enumerate((("loss", "Spectral error (lower better)"),
                                                   ("si_sdr", "SI-SDR, dB (higher better)"))):
            ax = axes[row_index, column]
            for color, depth in zip(colors, CODEBOOKS):
                points = sorted((row for row in summary["grid"] if row["domain"] == domain
                                 and row["codebooks"] == depth), key=lambda row: row["nfe"])
                baseline = next(row[metric] for row in points if row["nfe"] == 0)
                flow = [row for row in points if row["nfe"] > 0]
                ax.plot([row["nfe"] for row in flow], [row[metric] for row in flow],
                        "o-", color=color, label=f"{depth} codebooks")
                ax.axhline(baseline, color=color, linestyle="--", alpha=.65)
            ax.set_title(domain)
            ax.set_ylabel(label)
            compute_axis(ax)
    axes[0, 0].legend(fontsize=8)
    finish(fig, "01_quality_vs_compute.png", "Do more flow evaluations improve reconstruction? Dashed = matching NDAC",
           "Compare each solid curve with its same-color dashed NDAC baseline. Falling spectral error is good; rising "
           "SI-SDR is good. A plateau means extra evaluations add little under that metric. Neither metric proves listening quality.")

    baseline = {(row["record_index"], row["codebooks"]): row["loss"] for row in grid if row["nfe"] == 0}
    fig, axes = plt.subplots(1, len(domains), figsize=(4.4 * len(domains), 4.2), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        for color, depth in zip(colors, CODEBOOKS):
            estimates = []
            for nfe in nfes:
                values = [(row, baseline[(row["record_index"], depth)] - row["loss"]) for row in grid
                          if row["nfe"] == nfe and row["codebooks"] == depth
                          and (domain == "all" or row["domain"] == domain)]
                estimates.append(paired_gain(values, config["seed"]))
            mean, low, high = np.array(estimates).T
            ax.plot(nfes, mean, "o-", color=color, label=f"{depth} codebooks")
            ax.fill_between(nfes, low, high, color=color, alpha=.12)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(domain)
        ax.set_ylabel("NDAC error minus flow error")
        compute_axis(ax)
    axes[0, 0].legend(fontsize=8)
    finish(fig, "02_flow_vs_ndac.png", "Does flow beat NDAC on spectral error? Positive is better",
           "Above zero: flow improves spectral error at the same bitrate. Below zero: NDAC scores better. Shading is a "
           "95% group-bootstrap interval over paired differences, not over individual blocks. Crossing zero leaves the direction uncertain.")

    fig, axes = plt.subplots(1, len(domains), figsize=(4.4 * len(domains), 4.2), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        for color, depth in zip(colors, CODEBOOKS):
            points = sorted((row for row in summary["grid"] if row["domain"] == domain
                             and row["codebooks"] == depth and 0 < row["nfe"] < config["reference_nfe"]),
                            key=lambda row: row["nfe"])
            ax.plot([row["nfe"] for row in points], [max(row["nmse_reference"], 1e-12) for row in points],
                    "o-", color=color, label=f"{depth} codebooks")
        ax.set_yscale("log")
        ax.set_title(domain)
        ax.set_ylabel("Waveform NMSE to high-NFE output")
        compute_axis(ax)
    axes[0, 0].legend(fontsize=8)
    finish(fig, "03_numerical_agreement.png", f"Agreement with the {config['reference_nfe']}-NFE reference (lower better)",
           "Small disagreement indicates closeness to this numerical reference, not closeness to the original audio. "
           "The reference is omitted because its self-error is zero by construction. Values below 1e-12 are plotted at 1e-12. "
           "This cannot prove convergence beyond the largest tested budget.")

    fig, axes = plt.subplots(2, len(domains), figsize=(4.4 * len(domains), 7), squeeze=False)
    levels = [0] + nfes
    for column, domain in enumerate(domains):
        for row_index, metric in enumerate(("best_uniform_gain", "placement_gain")):
            ax = axes[row_index, column]
            for color, budget in zip(colors, config["budgets_kbps"]):
                points = sorted((row for row in summary["oracle"] if row["domain"] == domain
                                 and row["budget_kbps"] == budget), key=lambda row: row["nfe"])
                if not points:
                    continue
                x = [levels.index(row["nfe"]) for row in points]
                y = [row[metric] for row in points]
                limits = [row.get(metric + "_95ci") or [np.nan, np.nan] for row in points]
                ax.plot(x, y, "o-", color=color, label=f"budget {budget:g} kbps")
                ax.fill_between(x, [pair[0] for pair in limits], [pair[1] for pair in limits], color=color, alpha=.15)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title(domain)
            ax.set_ylabel("Gain vs selected constant rate" if row_index == 0 else "Gain vs shuffled placements")
            ax.set_xticks(range(len(levels)), ["NDAC"] + [str(nfe) for nfe in nfes], rotation=45)
            ax.set_xlabel("Decoder / NFE")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    finish(fig, "04_allocation_headroom.png", "Can adaptive bit allocation help? Positive is better; bands = 95% intervals",
           "Top: allocation versus the constant rate chosen on selection seeds, under the same file budget. Bottom: "
           "allocation versus shuffled placements with the exact same rate histogram. Positive gains in both support "
           "allocation headroom under this metric. Negative results or intervals crossing zero weaken the case. "
           "The oracle sees the original; this does not demonstrate a deployable predictor or a useful flow-based signal.")

    fig, axes = plt.subplots(1, len(domains), figsize=(4.4 * len(domains), 4.2), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        for color, depth in zip(colors, CODEBOOKS):
            points = sorted((row for row in summary["grid"] if row["domain"] == domain
                             and row["codebooks"] == depth), key=lambda row: row["nfe"])
            flow = [row for row in points if row["nfe"] > 0]
            ax.plot([row["decoder_rtf"] for row in flow], [row["loss"] for row in flow],
                    "o-", color=color, label=f"{depth} codebooks")
            ndac = next(row for row in points if row["nfe"] == 0)
            ax.scatter([ndac["decoder_rtf"]], [ndac["loss"]], marker="x", color=color)
        ax.set_title(domain)
        ax.set_xlabel("Decoder seconds / audio second")
        ax.set_ylabel("Spectral error (lower better)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    finish(fig, "05_quality_vs_time.png", "Quality versus measured decoding time; crosses = NDAC",
           "Lower-left is preferable on these axes. RTF=1 means one second of decoding per second of audio, excluding "
           "encoding, file I/O and metrics. Mixed GPU models or different load across jobs can confound timings; "
           "existing caches do not certify identical hardware. Do not claim a speed advantage from an uncontrolled comparison.")

    first = 6 if 6 in nfes else nfes[0]
    next_nfe = next((nfe for nfe in nfes if nfe > first), None)
    if next_nfe is not None:
        lookup = {(row["record_index"], row["block"], row["codebooks"], row["nfe"]): row["evaluation_loss"]
                  for row in measurements}
        fig, ax = plt.subplots(figsize=(8, 6))
        for color, domain in zip(colors, domains[1:]):
            x, y = [], []
            for row in measurements:
                if row["domain"] != domain or row["codebooks"] != CODEBOOKS[0] or row["nfe"] != first:
                    continue
                prefix = (row["record_index"], row["block"])
                base = row["evaluation_loss"]
                x.append(base - lookup[prefix + (CODEBOOKS[1], first)])
                y.append(base - lookup[prefix + (CODEBOOKS[0], next_nfe)])
            ax.scatter(x, y, label=domain, color=color, alpha=.55, s=20)
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel(f"Error reduction: {CODEBOOKS[0]} to {CODEBOOKS[1]} codebooks, at NFE {first}")
        ax.set_ylabel(f"Error reduction: NFE {first} to {next_nfe}, at {CODEBOOKS[0]} codebooks")
        ax.legend()
        ax.grid(alpha=.2)
        finish(fig, "06_bits_vs_compute.png", "Do different blocks benefit from different resources?",
               "Right: extra bits helped. Up: extra computation helped. Lower-right suggests bits helped while computation "
               "did not; upper-left suggests the reverse. Both positive: both helped. Blocks share recordings and are not "
               "independent statistical replicates. The two upgrades have different costs, so this is not an equal-cost efficiency comparison.")

    sections = []
    for path, title, explanation in figures:
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(explanation)}</p>'
                        f'<img src="plots/{path.name}" alt="{html.escape(title)}"></section>')
    page = ('<!doctype html><meta charset="utf-8"><title>Flow bitrate diagnostic</title>'
            '<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;line-height:1.5}'
            'img{width:100%;height:auto}section{margin:36px 0}p{max-width:1000px}</style>'
            '<h1>Flow bitrate diagnostic</h1><p>These plots describe the pilot, not a perceptual-quality verdict. '
            'Compare domains separately. Bands resample source groups, not audio blocks.</p>'
            '<p><a href="audio/index.html">Open listening comparisons</a> '
            '(generate with <code>sbatch slurm/listen.slurm</code> if missing).</p>' + ''.join(sections))
    temporary = output / "report.html.part"
    temporary.write_text(page)
    commit_file(temporary, output / "report.html")
    return [path for path, _, _ in figures]
