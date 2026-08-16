"""Generate all paper figures (PDF, vector) from the real experiment logs."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BLUE = "#2a78d6"    # DAVI
ORANGE = "#eb6834"  # diffusion-style

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def probes(path):
    recs = [r for r in load_jsonl(path) if "probe" in r]
    return recs


def fig_training():
    """Greedy probe solve rate vs wall-clock training time, 3x3, both methods."""
    davi = probes(os.path.join(ROOT, "runs/3x3_v1/metrics.jsonl"))
    diff = probes(os.path.join(ROOT, "runs/3x3_diff/metrics.jsonl"))
    IPS_DAVI, IPS_DIFF = 5.78, 28.5     # measured steady-state iterations/sec
    depths = ["10", "14", "30"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for ax, d in zip(axes, depths):
        ax.plot([r["iter"] / IPS_DAVI / 3600 for r in davi],
                [r["probe"].get(d, 0) * 100 for r in davi],
                color=BLUE, lw=1.6, label="DAVI (value)")
        ax.plot([r["iter"] / IPS_DIFF / 3600 for r in diff],
                [r["probe"].get(d, 0) * 100 for r in diff],
                color=ORANGE, lw=1.6, label="Denoising (ours)")
        ax.set_title(f"scramble depth {d}")
        ax.set_xlabel("wall-clock training (h)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("greedy solve rate (%)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_training.pdf"), bbox_inches="tight")
    print("fig_training.pdf")


def fig_width_scan():
    scan = json.load(open(os.path.join(ROOT, "width_scan.json")))
    widths = [1, 8, 32, 128, 512, 2048]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.2))
    for key, col, label in (("3x3_davi", BLUE, "DAVI (value)"),
                            ("3x3_diffusion", ORANGE, "Denoising (ours)")):
        d = scan[key]
        axes[0].plot(widths, [d[str(w)]["rate"] * 100 for w in widths],
                     "o-", color=col, lw=1.6, ms=3.5, label=label)
        axes[1].plot(widths, [d[str(w)]["secs"] for w in widths],
                     "o-", color=col, lw=1.6, ms=3.5, label=label)
    axes[0].set_xscale("log", base=2); axes[0].set_xticks(widths, [str(w) for w in widths])
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(widths, [str(w) for w in widths])
    axes[1].set_yscale("log")
    axes[0].set_xlabel("beam width"); axes[0].set_ylabel("solve rate (%)")
    axes[1].set_xlabel("beam width"); axes[1].set_ylabel("wall-clock, 200 cubes (s)")
    axes[0].grid(alpha=0.25, lw=0.5); axes[1].grid(alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_width_scan.pdf"), bbox_inches="tight")
    print("fig_width_scan.pdf")


def fig_error_structure():
    """2x2 exact-oracle error structure: policy move quality and value MAE by true depth."""
    diff = json.load(open(os.path.join(ROOT, "runs/2x2_diff/eval_report.json")))
    davi = json.load(open(os.path.join(ROOT, "runs/2x2_v1/eval_report.json")))
    mq = diff["argmax_move_reduces_distance"]["by_depth"]
    mae = davi["value_mae_by_depth"]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.2))
    ds = sorted(int(k) for k in mq)
    axes[0].bar(ds, [mq[str(d)] * 100 for d in ds], color=ORANGE, width=0.7)
    axes[0].set_xlabel("true distance $d^*(s)$")
    axes[0].set_ylabel("argmax move reduces\ndistance (%)")
    axes[0].set_ylim(0, 105)
    ds2 = sorted(int(k) for k in mae)
    axes[1].bar(ds2, [mae[str(d)] for d in ds2], color=BLUE, width=0.7)
    axes[1].set_xlabel("true distance $d^*(s)$")
    axes[1].set_ylabel("DAVI value MAE\n(moves)")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_error_structure.pdf"), bbox_inches="tight")
    print("fig_error_structure.pdf")


if __name__ == "__main__":
    fig_training()
    fig_width_scan()
    fig_error_structure()
