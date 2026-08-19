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


def fig_domains_strip():
    """Teaser: one real rendered instance per domain."""
    import sys
    sys.path.insert(0, ROOT)
    import torch
    import numpy as np
    from domains.slide import SlideEnv
    from domains.maze import MazeEnv, observe
    from domains.sokoban import SokobanEnv
    from domains.game2048 import spawn, S as S2048
    from domains.hanoi import HanoiEnv

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fig, axes = plt.subplots(1, 7, figsize=(7.0, 1.35))

    def grid_ax(ax, title):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=7.2, pad=3)

    # 1: 24-puzzle
    env = SlideEnv(5, dev)
    st = env.scramble(1, 300, generator=torch.Generator(device=dev).manual_seed(1))
    b = st[0].reshape(5, 5).cpu().numpy()
    ax = axes[0]
    ax.imshow((b > 0).astype(float), cmap="Oranges", vmin=-0.6, vmax=1.6)
    for r in range(5):
        for c in range(5):
            if b[r, c]:
                ax.text(c, r, str(b[r, c]), ha="center", va="center", fontsize=5)
    grid_ax(ax, "24-puzzle\n(scale)")

    # 2: maze
    envm = MazeEnv(15, device=dev)
    g = torch.Generator(device=dev).manual_seed(3)
    walls, goal = envm.new_instances(1, generator=g)
    stm = envm.scramble(walls, goal, 30, generator=g)
    img = stm[0].reshape(15, 15).cpu().numpy().astype(float)
    ax = axes[1]
    ax.imshow(img, cmap="Greys", vmin=0, vmax=4)
    grid_ax(ax, "maze\n(conditional)")

    # 3: sokoban
    envs = SokobanEnv(8, 3, device=dev)
    w, go, bx, ag, _ = envs.instances_and_scramble(
        1, 30, generator=torch.Generator(device=dev).manual_seed(5))
    img = envs.render(w, go, bx, ag)[0].reshape(8, 8).cpu().numpy().astype(float)
    ax = axes[2]
    ax.imshow(img, cmap="YlOrBr", vmin=0, vmax=6)
    grid_ax(ax, "Sokoban\n(irreversible)")

    # 4: pendulum sketch
    ax = axes[3]
    th = 2.6
    ax.plot([0, np.sin(th)], [0, np.cos(th)], lw=2.4, color=BLUE)
    ax.plot([0], [0], "o", ms=4, color="k")
    ax.plot([np.sin(th)], [np.cos(th)], "o", ms=8, color=BLUE)
    ax.plot([0, 0], [0, 1.05], ls=":", lw=0.8, color=GRAY)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    grid_ax(ax, "pendulum\n(continuous)")

    # 5: 2048
    g = torch.Generator(device=dev).manual_seed(0)
    bb = torch.zeros(1, S2048, dtype=torch.int8, device=dev)
    for _ in range(9):
        bb = spawn(bb, g)
    b4 = bb[0].reshape(4, 4).cpu().numpy()
    ax = axes[4]
    ax.imshow((b4 > 0).astype(float), cmap="Purples", vmin=-0.6, vmax=1.8)
    for r in range(4):
        for c in range(4):
            if b4[r, c]:
                ax.text(c, r, str(2 ** b4[r, c]), ha="center", va="center",
                        fontsize=6)
    grid_ax(ax, "2048\n(stochastic)")

    # 6: hanoi
    envh = HanoiEnv(10, dev)
    sth = envh.scramble(1, 400,
                        generator=torch.Generator(device=dev).manual_seed(2))
    pegs = sth[0].cpu().numpy()
    ax = axes[5]
    heights = [0, 0, 0]
    for disk in range(9, -1, -1):
        p = int(pegs[disk])
        wdt = 0.25 + 0.55 * (disk + 1) / 10
        ax.barh(heights[p], wdt, left=p - wdt / 2, height=0.8,
                color=ORANGE, edgecolor="white", linewidth=0.3)
        heights[p] += 1
    for p in range(3):
        ax.plot([p, p], [0, 10.5], color=GRAY, lw=0.8, zorder=0)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 11)
    grid_ax(ax, "Hanoi\n(horizon 1023)")

    # 7: POMDP maze
    obs = observe(stm, 15, 3)[0].reshape(15, 15).cpu().numpy().astype(float)
    ax = axes[6]
    ax.imshow(obs, cmap="Greys", vmin=0, vmax=4)
    grid_ax(ax, "POMDP maze\n(partial obs)")

    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(HERE, "fig2_domains.pdf"), bbox_inches="tight")
    print("fig2_domains.pdf")
