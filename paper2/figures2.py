"""Figures for paper 2, generated from paper2_data/ and runs/ logs."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a958c"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def jload(p):
    return json.load(open(os.path.join(ROOT, "paper2_data", p)))


def probes(path):
    out = []
    with open(os.path.join(ROOT, path)) as f:
        for line in f:
            r = json.loads(line)
            if "probe" in r:
                out.append(r)
    return out


def fig_summary():
    """Headline per-domain comparison: one bar pair per domain."""
    slide24 = jload("domain1_slide24.json")
    maze = jload("domain2_maze.json")
    soko = jload("domain3_sokoban.json")
    pend = jload("domain4_pendulum.json")
    g = jload("domain5_2048.json")
    doms = [
        ("24-puzzle\n(scale)", slide24["denoise"]["total_rate"],
         slide24["davi"]["total_rate"]),
        ("mazes\n(conditional)", maze["denoise"]["solve_rate"],
         maze["davi"]["solve_rate"]),
        ("Sokoban\n(irreversible)", soko["denoise"]["depth_60"]["solve_rate"],
         soko["davi"]["depth_60"]["solve_rate"]),
        ("pendulum\n(continuous)", pend["denoise_categorical_21bins"]["swingup_rate"],
         pend["value_baseline_lookahead"]["swingup_rate"]),
        ("2048\n(stochastic)", g["denoiser"]["reach_256"],
         None),  # no DAVI variant; heuristic shown separately
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    xs = range(len(doms))
    w = 0.36
    for i, (name, dn, dv) in enumerate(doms):
        ax.bar(i - w / 2, dn * 100, w, color=ORANGE,
               label="denoiser" if i == 0 else None)
        if dv is not None:
            ax.bar(i + w / 2, dv * 100, w, color=BLUE,
                   label="value baseline" if i == 0 else None)
    # 2048 references
    ax.bar(len(doms) - 1 + w / 2, g["greedy_merge"]["reach_256"] * 100, w,
           color=GRAY, label="greedy-merge heuristic")
    ax.axhline(g["random"]["reach_256"] * 100, ls=":", lw=1, color=GRAY)
    ax.text(len(doms) - 1.38, g["random"]["reach_256"] * 100 + 2, "random",
            fontsize=7, color=GRAY)
    ax.set_xticks(list(xs), [d[0] for d in doms])
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 108)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(frameon=False, ncols=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_summary.pdf"), bbox_inches="tight")
    print("fig2_summary.pdf")


def fig_slide24_training():
    dn = probes("runs/slide5_diff/metrics.jsonl")
    dv = probes("runs/slide5_davi/metrics.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), sharey=True)
    for ax, recs, name, col in ((axes[0], dn, "denoiser", ORANGE),
                                (axes[1], dv, "value (DAVI)", BLUE)):
        for d, alpha in (("30", 0.35), ("60", 0.55), ("120", 0.75), ("200", 1.0)):
            ax.plot([r["iter"] / 1000 for r in recs],
                    [r["probe"].get(d, 0) * 100 for r in recs],
                    color=col, alpha=alpha, lw=1.5, label=f"depth {d}")
        ax.set_title(name)
        ax.set_xlabel("iterations (k)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylabel("greedy solve rate (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_slide24.pdf"), bbox_inches="tight")
    print("fig2_slide24.pdf")


def fig_sokoban_depth():
    soko = jload("domain3_sokoban.json")
    depths = [10, 25, 40, 60]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for tag, col, lab in (("denoise", ORANGE, "denoiser"),
                          ("davi", BLUE, "value (DAVI)")):
        ax.plot(depths, [soko[tag][f"depth_{d}"]["solve_rate"] * 100
                         for d in depths], "o-", color=col, lw=1.6, ms=3.5,
                label=lab)
    ax.set_xlabel("instance pull depth")
    ax.set_ylabel("solve rate (%)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_sokoban.pdf"), bbox_inches="tight")
    print("fig2_sokoban.pdf")


def fig_pendulum_bars():
    pend = jload("domain4_pendulum.json")
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    names = ["MSE\nregression", "categorical\n(21 bins)", "value\nlookahead"]
    vals = [pend["denoise_mse_regression"]["swingup_rate"],
            pend["denoise_categorical_21bins"]["swingup_rate"],
            pend["value_baseline_lookahead"]["swingup_rate"]]
    cols = ["#e2c4b4", ORANGE, BLUE]
    ax.bar(names, [v * 100 for v in vals], color=cols, width=0.6)
    ax.set_ylabel("swing-up success (%)")
    ax.set_ylim(0, 108)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 3, f"{v*100:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_pendulum.pdf"), bbox_inches="tight")
    print("fig2_pendulum.pdf")


if __name__ == "__main__":
    fig_summary()
    fig_slide24_training()
    fig_sokoban_depth()
    fig_pendulum_bars()
