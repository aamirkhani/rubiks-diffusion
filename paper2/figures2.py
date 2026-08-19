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
    fig, axes = plt.subplots(1, 10, figsize=(7.0, 1.05))

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


    # 8: lights out
    from domains.lightsout import LightsOutEnv
    envl = LightsOutEnv(dev)
    stl = envl.scramble(1, 12, generator=torch.Generator(device=dev).manual_seed(4))
    ax = axes[7]
    ax.imshow(stl[0].reshape(5, 5).cpu().numpy(), cmap="YlOrBr", vmin=0, vmax=1.4)
    grid_ax(ax, "Lights Out\n(commutative)")

    # 9: mountain car
    ax = axes[8]
    xsr = np.linspace(-1.2, 0.6, 80)
    ax.plot(xsr, np.sin(3 * xsr) * 0.45, color=GRAY, lw=1.0)
    ax.plot([-0.5], [np.sin(-1.5) * 0.45 + 0.05], "o", ms=5, color=BLUE)
    ax.set_xlim(-1.3, 0.7); ax.set_ylim(-0.7, 0.7)
    grid_ax(ax, "Mountain Car\n(non-monotone)")

    # 10: peg solitaire
    from domains.pegsolitaire import PegEnv
    envpg = PegEnv(dev)
    stp, _ = envpg.scramble(1, 18, return_actions=True,
                            generator=torch.Generator(device=dev).manual_seed(6))
    img = stp[0].reshape(7, 7).cpu().numpy().astype(float)
    img[img == 2] = np.nan
    cmp_ = plt.get_cmap("Oranges").copy(); cmp_.set_bad("#efece6")
    ax = axes[9]
    ax.imshow(img, cmap=cmp_, vmin=-0.4, vmax=1.4)
    grid_ax(ax, "Peg Solitaire\n(non-conserv.)")

    fig.tight_layout(w_pad=0.5)
    fig.savefig(os.path.join(HERE, "fig2_domains.pdf"), bbox_inches="tight")
    print("fig2_domains.pdf")


def fig_architecture2():
    """One recipe, ten domains: per-domain encodings -> shared residual MLP
    -> denoiser head (M_d logits) or value head (scalar). Real logits from
    the trained Sokoban denoiser as the worked example."""
    import sys
    sys.path.insert(0, ROOT)
    import torch
    import torch.nn.functional as Fn
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    INK, MUT = "#1a1a19", "#6b6a66"

    doms = [("Rubik 3x3", 54, 6, 12), ("24-puzzle", 25, 25, 4),
            ("maze", 225, 4, 4), ("Sokoban", 64, 7, 4),
            ("pendulum", "3 feats", "-", 21), ("2048", 16, 12, 4),
            ("Hanoi", 10, 3, 6), ("POMDP maze", 225, 5, 4),
            ("Lights Out", 25, 2, 25), ("Mountain Car", "3 feats", "-", 3),
            ("Peg Solitaire", 49, 3, 196)]

    fig = plt.figure(figsize=(7.0, 3.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # left: domain chips
    ax.text(0.13, 0.985, "one-hot state encodings", ha="center", fontsize=8,
            color=INK, weight="bold")
    for i, (name, S_, V_, M_) in enumerate(doms):
        y = 0.93 - i * 0.077
        ax.add_patch(FancyBboxPatch((0.015, y - 0.030), 0.225, 0.060,
                                    boxstyle="round,pad=0.008",
                                    fc="#f0efec", ec="#c9c5bd", lw=0.7))
        ax.text(0.03, y, name, fontsize=6.8, va="center", color=INK)
        ax.text(0.235, y, f"S={S_}, |V|={V_}", fontsize=5.8, va="center",
                ha="right", color=MUT)
        ax.annotate("", xy=(0.30, 0.5), xytext=(0.245, y),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#cfc9c0",
                                    connectionstyle="arc3,rad=0.12"))

    def box(x, y, w, h, t, sub=None, fc="#f0efec"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                    fc=fc, ec=INK, lw=1.0))
        ax.text(x + w / 2, y + h / 2 + (0.03 if sub else 0), t, ha="center",
                va="center", fontsize=8, color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.045, sub, ha="center",
                    va="center", fontsize=6.4, color=MUT)

    box(0.305, 0.30, 0.115, 0.40, "Linear $h_1$", "LayerNorm+ReLU")
    box(0.445, 0.30, 0.115, 0.40, "Linear $h_2$", "LayerNorm+ReLU")
    box(0.585, 0.30, 0.135, 0.40, "ResBlock $h_2$", "$\\times$2--4")
    ax.annotate("", xy=(0.700, 0.78), xytext=(0.598, 0.78),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=MUT,
                                connectionstyle="arc3,rad=-0.4"))
    ax.text(0.65, 0.85, "skip", fontsize=6, color=MUT, ha="center")
    for x1, x2 in ((0.421, 0.443), (0.561, 0.583)):
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK))
    ax.annotate("", xy=(0.76, 0.62), xytext=(0.722, 0.55),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color=ORANGE))
    ax.annotate("", xy=(0.76, 0.33), xytext=(0.722, 0.44),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color=BLUE))

    # denoiser head with real Sokoban logits
    try:
        from domains.eval_sokoban import load
        env, pnet, _ = load(os.path.join(ROOT, "runs/soko_diff/ckpt_latest.pt"))
        g = torch.Generator(device="cuda").manual_seed(1)
        w, go, b, a_, lab = env.instances_and_scramble(1, 12, generator=g)
        st = env.render(w, go, b, a_)
        with torch.no_grad():
            pr = Fn.softmax(pnet(st)[0].float(), 0).cpu().numpy()
        axb = fig.add_axes([0.79, 0.56, 0.185, 0.30])
        cols = [ORANGE if i == int(pr.argmax()) else "#e2c4b4"
                for i in range(4)]
        axb.bar(["U", "D", "L", "R"], pr, color=cols, width=0.6)
        axb.set_yticks([])
        for s_ in axb.spines.values():
            s_.set_visible(False)
        axb.set_title("denoiser head: $p_\\theta(a|s)$\n(real Sokoban pass)",
                      fontsize=7, color=INK)
        axb.tick_params(labelsize=7, colors=INK)
    except Exception as e:
        print("arch head err:", e)
    box(0.79, 0.20, 0.185, 0.16, "value head: $V_\\theta(s)$",
        "scalar cost-to-go", fc="#e3edf9")
    ax.text(0.5, 0.012, "identical backbone across all domains and both "
            "objectives — only input width, head, and loss change",
            ha="center", fontsize=7.4, color=MUT, style="italic")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig2_architecture.{ext}"),
                    bbox_inches="tight", dpi=170)
    print("fig2_architecture.pdf")
