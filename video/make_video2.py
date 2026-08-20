"""Render the paper-2 YouTube explainer (1080p30, ~4 min): "Beyond the Cube —
which RL problems can diffusion actually solve?"

Every state shown is a real environment state; every denoising run is the
actual trained denoiser's greedy rollout (the same trajectories as the paper's
galleries, unsnapped to full length). Frames -> ffmpeg (H.264) + ambient pad.

Usage:  python video/make_video2.py [--fps 30] [--out video/beyond_the_cube.mp4]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "paper2"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import make_video as MV1
from make_video import (Video, ease, draw_cube, spiral_pts, astro, make_pad,
                        BG, INK, MUT, ORANGE, BLUE, GREEN, env3)

RED = "#e4574f"
GOLD = "#e7b93c"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- harvest FULL trajectories from the paper's gallery machinery ---------
import fig_process_gallery as fpg
import fig_reverse_gallery as frg
fpg.snap = lambda seq: seq                      # unsnap: keep every state
frg.snap_idx = lambda L: list(range(L + 1))

plt.rcParams.update({"figure.dpi": 100})


def clampa(x):
    return float(np.clip(x, 0, 1))


def resample(seq, n):
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return [seq[i] for i in idx]


# --------------------------------------------------------------- domain meta
# (key, no, title, deleted assumption, noise-operator label, verdict,
#  result lines, verdict color)
DOMS = [
    ("slide", 1, "24-puzzle", "group structure",
     "noise = random slides", "WORKS",
     ["denoiser  100%", "DAVI  0.6%", "$7.7\\times10^{24}$ states,\nmatched compute"], GREEN),
    ("maze", 2, "procedural mazes", "fixed goal & environment",
     "noise = random walk in a fresh maze", "WORKS",
     ["denoiser  78%", "DAVI  77%", "conditioning survives:\nnew maze every episode"], GREEN),
    ("soko", 3, "Sokoban", "invertible actions",
     "noise = pulls (the inverse of pushes)", "FIXABLE",
     ["denoiser  66%", "DAVI  83%", "hybrid  93%", "reverse data never\nshows a deadlock"], GOLD),
    ("pend", 4, "pendulum swing-up", "discreteness",
     "noise = random torques in reverse time", "WORKS",
     ["MSE head  0%", "categorical head  100%", "the fix is the output\nhead, not the recipe"], GREEN),
    ("g2048", 5, "2048", "determinism",
     "noise = reverse play: un-merges, un-spawns", "FAILS",
     ["denoiser  $=$  random play", "greedy heuristic wins", "exogenous randomness\nbreaks the recipe"], RED),
    ("hanoi", 6, "Tower of Hanoi", "bounded horizon",
     "noise = random legal moves (optimum: 1023)", "FAILS",
     ["both methods  ~2%", "walks never reach\nthe far shells", "the schedule, not the\nlearner, is the ceiling"], RED),
    ("pomdp", 7, "POMDP maze", "full observability",
     "noise = random walk, seen through a 7$\\times$7 window", "WORKS",
     ["denoiser  84.6%", "DAVI  86.2%", "partial observability\nis no obstacle"], GREEN),
    ("lo", 8, "Lights Out", "ordered solutions",
     "noise = random presses (solutions commute)", "WORKS",
     ["denoiser  100%", "DAVI  33.9%", "label ambiguity is\naveraged over, not fatal"], GREEN),
    ("mc", 9, "Mountain Car", "monotone progress",
     "noise = reverse-time physics", "WORKS",
     ["denoiser  95.8%", "value baseline  41.8%", "the momentum detour is\nin the training data"], GREEN),
    ("pegs", 10, "Peg Solitaire", "conservation",
     "noise = un-jumps that ADD pegs", "WORKS",
     ["denoiser  99.7%", "DAVI  0.0%", "mass creation/destruction\nis fine"], GREEN),
]
META = {d[0]: d for d in DOMS}
VCOL = {"WORKS": GREEN, "FIXABLE": GOLD, "FAILS": RED}
VMARK = {"WORKS": "✓", "FIXABLE": "✓*", "FAILS": "✗"}


def vid_rslide(ax, b, n=5):
    b = np.asarray(b).reshape(n, n)
    ax.imshow((b > 0).astype(float), cmap="Oranges", vmin=-0.6, vmax=1.6)
    for r in range(n):
        for c in range(n):
            if b[r, c]:
                ax.text(c, r, str(b[r, c]), ha="center", va="center",
                        fontsize=17, color="#3b1804", weight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def vid_r2048(ax, b):
    b = np.asarray(b).reshape(4, 4)
    ax.imshow((b > 0).astype(float), cmap="Purples", vmin=-0.6, vmax=1.8)
    for r in range(4):
        for c in range(4):
            if b[r, c]:
                ax.text(c, r, str(2 ** int(b[r, c])), ha="center", va="center",
                        fontsize=21, color="#241145", weight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def collect():
    print("collecting forward (noising) sequences...", flush=True)
    fwd = fpg.fwd_sequences()
    fwd["slide"] = (fwd["slide"][0], vid_rslide, fwd["slide"][2])
    fwd["g2048"] = (fwd["g2048"][0], vid_r2048, fwd["g2048"][2])
    print("collecting reverse (trained denoiser) trajectories...", flush=True)
    rev = fpg.rev_sequences()
    # cube scramble for the recap scene
    g = torch.Generator(device=DEV).manual_seed(7)
    sc_states, sc_moves = [env3.solved.cpu().numpy()], []
    s = env3.solved_batch(1)
    prev = -1
    for _ in range(12):
        a = int(torch.randint(env3.M, (1,), device=DEV, generator=g))
        while a == (prev ^ 1):
            a = int(torch.randint(env3.M, (1,), device=DEV, generator=g))
        sc_moves.append(a)
        s = env3.step(s, torch.tensor([a], device=DEV))
        sc_states.append(s[0].cpu().numpy())
        prev = a
    return fwd, rev, sc_states, sc_moves


def fade(ax, i, n, fps):
    """Cinematic fade in/out overlay (0.35 s each side)."""
    f = int(0.35 * fps)
    a = 0.0
    if i < f:
        a = 1 - i / f
    elif i > n - f:
        a = 1 - (n - i) / f
    if a > 0.01:
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=BG, alpha=clampa(a),
                                   zorder=50))


def panel(V, rect):
    ax = V.fig.add_axes(rect)
    ax.set_facecolor(BG)
    return ax


def tbar(ax, x0, x1, y, t, forward=True):
    """Schedule bar with a moving marker; t in [0,1] is noising progress."""
    ax.add_patch(plt.Rectangle((x0, y), x1 - x0, 0.010, color="#2c2a27"))
    ax.add_patch(plt.Rectangle((x0, y), (x1 - x0) * t, 0.010,
                               color=ORANGE if forward else GREEN))
    ax.text(x0, y - 0.035, "$t=0$\ngoal", ha="center", fontsize=15, color=MUT)
    ax.text(x1, y - 0.035, "$t=1$\nnoise", ha="center", fontsize=15, color=MUT)
    xm = x0 + (x1 - x0) * t
    ax.plot([xm], [y + 0.005], marker="v", ms=11,
            color=ORANGE if forward else GREEN)


# ---------------------------------------------------------------- scenes
def scene_title(V, fwd, secs=6.0):
    n = int(secs * V.fps)
    keys = [d[0] for d in DOMS]
    for i in range(n):
        t = i / n
        ax = V.canvas()
        a = min(1, t / 0.2)
        ax.text(0.5, 0.86, "Can every RL problem be solved by", ha="center",
                fontsize=44, color=INK, alpha=a, weight="bold")
        ax.text(0.5, 0.76, "DIFFUSION?", ha="center", fontsize=64,
                color=ORANGE, alpha=a, weight="bold")
        # 10 domain thumbnails pop in one by one
        for j, k in enumerate(keys):
            t_on = 0.22 + 0.05 * j
            if t < t_on:
                continue
            aa = min(1, (t - t_on) / 0.08)
            axp = panel(V, [0.06 + 0.184 * (j % 5), 0.30 - 0.26 * (j // 5),
                            0.15, 0.24])
            frames, rend, _ = fwd[k]
            rend(axp, frames[0])
            axp.patch.set_alpha(0)
            for im in axp.get_images():
                im.set_alpha(aa)
            for ln in axp.lines:
                ln.set_alpha(aa)
            axp.set_title(META[k][2], color=INK, fontsize=13, alpha=aa, pad=3)
        if t > 0.78:
            ax.text(0.5, 0.035, "a ten-domain stress test — every claim replay-verified",
                    ha="center", fontsize=20, color=MUT,
                    alpha=clampa((t - 0.78) / 0.12))
        fade(ax, i, n, V.fps)
        V.frame()


def scene_recap(V, sc_states, sc_moves, secs=16.0):
    """Paper-1 recap: spiral + photo + cube, ONE shared schedule."""
    n = int(secs * V.fps)
    x0 = spiral_pts()
    eps = np.random.default_rng(0).standard_normal(x0.shape)
    img0 = astro()
    epsI = np.random.default_rng(1).standard_normal(img0.shape)
    n_mv = len(sc_moves)
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.94, "Last time: a Rubik's Cube is a diffusion problem",
                ha="center", fontsize=34, color=INK, weight="bold")
        if t < 0.5:
            ph, lab, lc = ease(t / 0.5), "forward: noising", MUT
        else:
            ph, lab, lc = ease(1 - (t - 0.5) / 0.5), "reverse: denoising = SOLVING", GREEN
        # spiral
        axs = panel(V, [0.035, 0.30, 0.28, 0.48])
        sig = 0.55 * ph ** 2
        pts = (1 - 0.55 * ph ** 2) * x0 + sig * eps
        axs.scatter(pts[:, 0], pts[:, 1], s=2.6, c=BLUE, alpha=0.75, linewidths=0)
        axs.set_xlim(-1.8, 1.8); axs.set_ylim(-1.8, 1.8)
        axs.set_aspect("equal"); axs.axis("off")
        axs.set_title("particles (1st diffusion paper)", color=MUT, fontsize=17)
        # image
        axi = panel(V, [0.365, 0.30, 0.28, 0.48])
        sI = 1.0 * ph ** 2
        axi.imshow(np.clip(np.sqrt(max(1 - sI ** 2, 1e-4)) * img0 + sI * epsI,
                           0, 1))
        axi.axis("off")
        axi.set_title("images (DDPM)", color=MUT, fontsize=17)
        # cube: forward = scramble anim; reverse = play it backwards
        axc = panel(V, [0.66, 0.22, 0.32, 0.62])
        prog = ph * n_mv
        j = min(n_mv - 1, int(prog))
        frac = ease(min(1.0, prog - j))
        if t < 0.5:
            draw_cube(axc, sc_states[j], move=sc_moves[j], frac=frac,
                      yaw=-0.62 + 0.1 * np.sin(t * 6))
        else:
            draw_cube(axc, sc_states[j + 1], move=sc_moves[j] ^ 1,
                      frac=1 - frac, yaw=-0.62 + 0.1 * np.sin(t * 6))
        axc.set_title("cubes (noise = random moves)", color=ORANGE, fontsize=17)
        ax.text(0.5, 0.185, lab, ha="center", fontsize=26, color=lc, weight="bold")
        tbar(ax, 0.25, 0.75, 0.10, ph, forward=t < 0.5)
        if t > 0.55:
            ax.text(0.5, 0.025,
                    "one network, one objective: predict the move that undoes the last scramble step",
                    ha="center", fontsize=19, color=MUT, alpha=clampa((t - 0.55) / 0.1))
        fade(ax, i, n, V.fps)
        V.frame()


def scene_ladder(V, fwd, secs=13.0):
    n = int(secs * V.fps)
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.93, "But the cube is special.", ha="center",
                fontsize=38, color=INK, weight="bold", alpha=clampa(t / 0.12))
        if t > 0.10:
            ax.text(0.5, 0.865,
                    "So we delete its special properties — one per domain — and see what breaks.",
                    ha="center", fontsize=24, color=ORANGE,
                    alpha=clampa((t - 0.10) / 0.1))
        for j, (k, no, title, cut, _, verdict, _, vc) in enumerate(DOMS):
            t_on = 0.18 + 0.062 * j
            if t < t_on:
                continue
            aa = min(1, (t - t_on) / 0.06)
            y = 0.775 - 0.072 * j
            axp = panel(V, [0.075, y - 0.028, 0.048, 0.062])
            frames, rend, _ = fwd[k]
            rend(axp, frames[0])
            ax.text(0.145, y, f"{no:02d}", fontsize=20, color=MUT, alpha=aa,
                    va="center")
            ax.text(0.19, y, title, fontsize=22, color=INK, alpha=aa,
                    va="center", weight="bold")
            ax.text(0.47, y, "deletes:", fontsize=17, color=MUT, alpha=aa,
                    va="center")
            ax.text(0.555, y, cut, fontsize=21, color=ORANGE, alpha=aa,
                    va="center")
            if t > t_on + 0.35:
                ax.text(0.92, y, VMARK[verdict], fontsize=26, color=vc,
                        va="center", ha="center",
                        alpha=clampa((t - t_on - 0.35) / 0.08))
        fade(ax, i, n, V.fps)
        V.frame()


def scene_recipe(V, secs=13.0):
    n = int(secs * V.fps)
    steps = [
        (0.05, "1", "start AT the goal", "no exploration, no reward shaping"),
        (0.26, "2", "noise it with random moves", "record which action was taken"),
        (0.50, "3", "train a denoiser  $p_\\theta(a\\,|\\,s)$",
         "predict the action that UNDOES the last step"),
        (0.75, "4", "act by greedy rollout", "reverse diffusion = solving"),
    ]
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.92, "The recipe never changes", ha="center",
                fontsize=38, color=INK, weight="bold")
        ax.text(0.5, 0.855, "same architecture, same objective, same compute budget as the value baseline",
                ha="center", fontsize=20, color=MUT)
        for x, num, head, sub in steps:
            t_on = 0.08 + 0.16 * steps.index((x, num, head, sub))
            if t < t_on:
                continue
            aa = min(1, (t - t_on) / 0.10)
            ax.add_patch(plt.Circle((x + 0.02, 0.62), 0.030, color=ORANGE,
                                    alpha=aa))
            ax.text(x + 0.02, 0.612, num, ha="center", va="center",
                    fontsize=26, color=BG, weight="bold", alpha=aa)
            ax.text(x + 0.02, 0.52, head, fontsize=23, color=INK, alpha=aa,
                    ha="left" if x < 0.7 else "center", weight="bold")
            ax.text(x + 0.02, 0.465, sub, fontsize=17, color=MUT, alpha=aa,
                    ha="left" if x < 0.7 else "center")
        # flowing dot along the pipeline
        if t > 0.65:
            tt = ((t - 0.65) / 0.35 * 2) % 1.0
            ax.plot([0.10 + 0.72 * tt], [0.62], "o", ms=13, color=GOLD)
        if t > 0.72:
            ax.text(0.5, 0.24, "the ONLY thing that changes per domain is what 'noise' means",
                    ha="center", fontsize=27, color=ORANGE, weight="bold",
                    alpha=clampa((t - 0.72) / 0.1))
            ax.text(0.5, 0.17, "slides · pulls · torques · presses · un-jumps · reverse play",
                    ha="center", fontsize=21, color=MUT,
                    alpha=clampa((t - 0.78) / 0.1))
        fade(ax, i, n, V.fps)
        V.frame()


def play_domain(V, key, fwd, rev, secs=13.5):
    k, no, title, cut, noiselab, verdict, results, vc = META[key]
    frames_f, rend, _ = fwd[key]
    have_rev = rev.get(key) is not None
    if have_rev:
        frames_r, ok, caption = rev[key]
    n = int(secs * V.fps)
    smooth = key in ("pend", "mc")
    NF = 110 if smooth else 40
    NR = 130 if smooth else 48
    seq_f = resample(frames_f, NF)
    seq_r = resample(frames_r, NR) if have_rev else None
    # timeline: 0-.34 noising | .34-.42 beat | .42-.80 denoising | .80-1 verdict
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.055, 0.925, f"{no:02d}/10", fontsize=30, color=MUT)
        ax.text(0.17, 0.925, title, fontsize=40, color=INK, weight="bold")
        ax.text(0.17, 0.865, "deletes: ", fontsize=24, color=MUT)
        ax.text(0.275, 0.865, cut, fontsize=24, color=ORANGE, weight="bold")
        axp = panel(V, [0.16, 0.14, 0.46, 0.64])
        if t < 0.34:                                   # ---- noising
            ph = ease(t / 0.34)
            j = min(NF - 1, int(ph * NF))
            rend(axp, seq_f[j])
            ax.text(0.39, 0.80, "forward: noising", ha="center", fontsize=25,
                    color=ORANGE, weight="bold")
            ax.text(0.39, 0.045, noiselab, ha="center", fontsize=21, color=MUT)
            tbar(ax, 0.20, 0.58, 0.105, j / (NF - 1), forward=True)
        elif t < 0.42 or not have_rev:                  # ---- beat
            rend(axp, seq_r[0] if have_rev else seq_f[-1])
            ax.text(0.39, 0.80, "a FRESH fully-noised instance", ha="center",
                    fontsize=25, color=INK, weight="bold")
            ax.text(0.39, 0.045, "can the trained denoiser bring it back?",
                    ha="center", fontsize=21, color=MUT)
            tbar(ax, 0.20, 0.58, 0.105, 1.0, forward=False)
        elif t < 0.80:                                  # ---- denoising
            ph = ease((t - 0.42) / 0.38)
            j = min(NR - 1, int(ph * NR))
            rend(axp, seq_r[j])
            ax.text(0.39, 0.80, "reverse: trained denoiser, greedy",
                    ha="center", fontsize=25, color=GREEN, weight="bold")
            ax.text(0.39, 0.045, caption, ha="center", fontsize=20, color=MUT)
            tbar(ax, 0.20, 0.58, 0.105, 1 - j / (NR - 1), forward=False)
        else:                                           # ---- hold last
            rend(axp, seq_r[-1])
            ax.text(0.39, 0.045, caption, ha="center", fontsize=20, color=MUT)
            tbar(ax, 0.20, 0.58, 0.105, 0.0, forward=False)
        # right column: results appear during denoising; stamp at the end
        for r_i, line in enumerate(results):
            t_on = 0.46 + 0.09 * r_i
            if t < t_on:
                continue
            aa = min(1, (t - t_on) / 0.08)
            ax.text(0.80, 0.62 - 0.105 * r_i, line, ha="center", fontsize=23,
                    color=INK if r_i < 2 else MUT, alpha=aa)
        if t > 0.82:
            pop = ease(min(1, (t - 0.82) / 0.10))
            fs = 58 - 12 * pop
            ax.text(0.80, 0.80, f"{VMARK[verdict]}  {verdict}", ha="center",
                    fontsize=fs, color=vc, weight="bold", rotation=-6,
                    alpha=clampa(pop * 1.4),
                    bbox=dict(boxstyle="round,pad=0.45", ec=vc, fc="none",
                              lw=3.5, alpha=clampa(pop * 1.4)))
        fade(ax, i, n, V.fps)
        V.frame()


def scene_dive_sokoban(V, secs=10.0):
    n = int(secs * V.fps)
    bars = [("denoiser", 66, ORANGE), ("DAVI", 83, BLUE), ("hybrid", 93, GREEN)]
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.91, "Failure mode 1: deadlock-blindness — and a free fix",
                ha="center", fontsize=34, color=INK, weight="bold")
        ax.text(0.5, 0.83, "noising by PULLS can never demonstrate a deadlock, so the denoiser walks into them",
                ha="center", fontsize=21, color=MUT)
        for b_i, (lab, v, c) in enumerate(bars):
            t_on = 0.15 + 0.18 * b_i
            if t < t_on:
                continue
            g = ease(min(1, (t - t_on) / 0.22))
            x = 0.22 + 0.24 * b_i
            ax.add_patch(plt.Rectangle((x, 0.22), 0.13, 0.48 * v / 100 * g,
                                       color=c))
            ax.text(x + 0.065, 0.22 + 0.48 * v / 100 * g + 0.025,
                    f"{v}%", ha="center", fontsize=30, color=c, weight="bold")
            ax.text(x + 0.065, 0.16, lab, ha="center", fontsize=23, color=INK)
        if t > 0.62:
            aa = min(1, (t - 0.62) / 0.1)
            ax.text(0.5, 0.06,
                    "hybrid = denoiser proposes, value function vetoes deadlocks — ZERO extra training",
                    ha="center", fontsize=23, color=GREEN, weight="bold", alpha=aa)
        fade(ax, i, n, V.fps)
        V.frame()


def scene_dive_pendulum(V, secs=10.0):
    n = int(secs * V.fps)
    xs = np.linspace(-2, 2, 300)
    bimodal = np.exp(-((xs - 1.15) ** 2) / 0.12) + np.exp(-((xs + 1.15) ** 2) / 0.12)
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.91, "Failure mode 2: continuous actions and mode collapse",
                ha="center", fontsize=34, color=INK, weight="bold")
        ax.text(0.5, 0.83, "near the bottom, +torque and $-$torque are BOTH correct — the posterior is bimodal",
                ha="center", fontsize=21, color=MUT)
        axp = panel(V, [0.16, 0.18, 0.68, 0.56])
        g = ease(min(1, t / 0.3))
        axp.fill_between(xs, bimodal * g, color=BLUE, alpha=0.45)
        axp.plot(xs, bimodal * g, color=BLUE, lw=2)
        axp.set_xlim(-2, 2); axp.set_ylim(0, 1.5)
        axp.axis("off")
        axp.text(0, -0.13, "action (torque)", ha="center", fontsize=18,
                 color=MUT)
        if t > 0.35:
            aa = min(1, (t - 0.35) / 0.1)
            axp.plot([0], [0.04], "o", ms=17, color=RED, alpha=aa)
            axp.annotate("MSE regresses the MEAN:\n0 torque — the one useless answer",
                         xy=(0, 0.08), xytext=(0, 0.85), ha="center",
                         fontsize=21, color=RED, alpha=aa,
                         arrowprops=dict(arrowstyle="->", color=RED, alpha=aa))
        if t > 0.60:
            aa = min(1, (t - 0.60) / 0.1)
            for xb in (1.15, -1.15):
                axp.plot([xb], [1.05], "v", ms=15, color=GREEN, alpha=aa)
            axp.text(1.15, 1.22, "a categorical head\npicks a MODE", ha="center",
                     fontsize=20, color=GREEN, alpha=aa)
        if t > 0.78:
            ax.text(0.5, 0.065, "swing-up:  MSE 0%   $\\rightarrow$   categorical 100%",
                    ha="center", fontsize=28, color=GREEN, weight="bold",
                    alpha=clampa((t - 0.78) / 0.1))
        fade(ax, i, n, V.fps)
        V.frame()


def scene_dive_hanoi(V, secs=11.0):
    n = int(secs * V.fps)
    # measured (paper2_data/domain6_hanoi_stratified.json), backtrack-allowed
    Ks = np.array([1200, 5000, 20000])
    reach = np.array([79, 137, 278])               # max true distance reached
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.91, "Failure mode 3: the schedule itself can be the ceiling",
                ha="center", fontsize=34, color=INK, weight="bold")
        ax.text(0.5, 0.83, "Hanoi's optimal solution is 1023 moves — but random walks diffuse, they don't travel",
                ha="center", fontsize=21, color=MUT)
        axp = panel(V, [0.14, 0.16, 0.72, 0.58])
        g = ease(min(1, max(0, (t - 0.1) / 0.45)))
        m = max(1, int(round(g * len(Ks))))
        axp.plot(np.sqrt(Ks[:m]), reach[:m], "o-", color=ORANGE, lw=3, ms=10)
        axp.set_xlim(0, 150); axp.set_ylim(0, 1100)
        axp.axhline(1023, color=RED, lw=2.5, ls="--")
        axp.text(8, 1040, "distance needed: 1023", fontsize=20, color=RED)
        axp.set_xlabel("$\\sqrt{\\mathrm{walk\\ length}\\ K}$", fontsize=19,
                       color=INK)
        axp.set_ylabel("farthest shell reached", fontsize=19, color=INK)
        axp.tick_params(colors=MUT, labelsize=15)
        for sp in axp.spines.values():
            sp.set_color("#3c3833")
        if t > 0.58:
            aa = min(1, (t - 0.58) / 0.1)
            axp.text(np.sqrt(20000) - 4, 330,
                     "20,000-step walks:\nstill only 278 of 1023", fontsize=20,
                     color=ORANGE, alpha=aa, ha="right")
        if t > 0.74:
            ax.text(0.5, 0.055,
                    "growth is $\\sqrt{K}$ (diffusive) — inside covered shells both methods are near-perfect, beyond: zero",
                    ha="center", fontsize=22, color=GOLD, weight="bold",
                    alpha=clampa((t - 0.74) / 0.1))
        fade(ax, i, n, V.fps)
        V.frame()


def scene_dive_2048(V, secs=9.0):
    n = int(secs * V.fps)
    bars = [("denoiser", 6.2, ORANGE), ("spawn-aware\ndenoiser", 6.2, GOLD),
            ("random", 7.1, MUT), ("greedy\nheuristic", 61.8, BLUE)]
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.91, "The one HARD boundary: exogenous randomness (2048)",
                ha="center", fontsize=34, color=INK, weight="bold")
        ax.text(0.5, 0.83, "the deterministic reverse process describes a game that is never played",
                ha="center", fontsize=21, color=MUT)
        for b_i, (lab, v, c) in enumerate(bars):
            t_on = 0.12 + 0.14 * b_i
            if t < t_on:
                continue
            g = ease(min(1, (t - t_on) / 0.2))
            x = 0.16 + 0.19 * b_i
            ax.add_patch(plt.Rectangle((x, 0.24), 0.11, 0.42 * v / 62 * g,
                                       color=c))
            ax.text(x + 0.055, 0.24 + 0.42 * v / 62 * g + 0.025, f"{v}%",
                    ha="center", fontsize=26, color=c, weight="bold")
            ax.text(x + 0.055, 0.145, lab, ha="center", fontsize=20, color=INK)
        ax.text(0.955, 0.20, "reach the\n256 tile", ha="center", fontsize=16,
                color=MUT)
        if t > 0.70:
            ax.text(0.5, 0.045,
                    "even matching the training states to real play doesn't help — the undo-labels answer the wrong question",
                    ha="center", fontsize=21, color=RED, weight="bold",
                    alpha=clampa((t - 0.70) / 0.1))
        fade(ax, i, n, V.fps)
        V.frame()


def scene_scoreboard(V, secs=13.0):
    n = int(secs * V.fps)
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.93, "The map", ha="center", fontsize=44, color=INK,
                weight="bold")
        for j, (k, no, title, cut, _, verdict, results, vc) in enumerate(DOMS):
            t_on = 0.08 + 0.055 * j
            if t < t_on:
                continue
            aa = min(1, (t - t_on) / 0.07)
            y = 0.82 - 0.068 * j
            ax.text(0.09, y, title, fontsize=22, color=INK, alpha=aa, va="center",
                    weight="bold")
            ax.text(0.36, y, cut, fontsize=19, color=MUT, alpha=aa, va="center")
            ax.text(0.68, y, results[0].replace("\n", " "), fontsize=19,
                    color=INK, alpha=aa, va="center", ha="center")
            ax.text(0.90, y, f"{VMARK[verdict]} {verdict}", fontsize=21,
                    color=vc, alpha=aa, va="center", ha="center", weight="bold")
        if t > 0.72:
            ax.text(0.5, 0.045,
                    "7 wins · 1 zero-training fix · 2 sharp, mechanistic failures",
                    ha="center", fontsize=26, color=ORANGE, weight="bold",
                    alpha=clampa((t - 0.72) / 0.1))
        fade(ax, i, n, V.fps)
        V.frame()


def scene_checklist(V, secs=12.0):
    n = int(secs * V.fps)
    items = [
        ("Can you run the dynamics backward from the goal, deterministically?",
         "if randomness is exogenous (2048): stop"),
        ("Do noising walks actually REACH the states you must solve?",
         "check shell coverage first (Hanoi)"),
        ("Are actions continuous?", "use a distributional head, never MSE"),
        ("Are some actions irreversible?", "add a value veto — the hybrid"),
    ]
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.91, "Before you try this at home: four questions",
                ha="center", fontsize=36, color=INK, weight="bold")
        for j, (q, a) in enumerate(items):
            t_on = 0.10 + 0.18 * j
            if t < t_on:
                continue
            aa = min(1, (t - t_on) / 0.1)
            y = 0.74 - 0.17 * j
            ax.text(0.08, y, f"{j+1}.", fontsize=28, color=ORANGE, alpha=aa,
                    weight="bold")
            ax.text(0.13, y, q, fontsize=25, color=INK, alpha=aa)
            ax.text(0.13, y - 0.06, a, fontsize=20, color=MUT, alpha=aa)
        fade(ax, i, n, V.fps)
        V.frame()


def scene_outro(V, secs=8.0):
    n = int(secs * V.fps)
    for i in range(n):
        t = i / n
        ax = V.canvas()
        axc = panel(V, [0.36, 0.48, 0.28, 0.46])
        draw_cube(axc, env3.solved.cpu().numpy(), yaw=-0.62 + t * 1.4)
        ax.text(0.5, 0.40, "Scramble Inversion Beyond Groups", ha="center",
                fontsize=34, color=INK, weight="bold")
        ax.text(0.5, 0.33, "a ten-domain map of where denoising diffusion solves sequential decision problems",
                ha="center", fontsize=20, color=MUT)
        ax.text(0.5, 0.24, "all environments, trainers, logs, and both papers:",
                ha="center", fontsize=19, color=MUT)
        ax.text(0.5, 0.17, "github.com/aamirkhani/rubiks-diffusion", ha="center",
                fontsize=32, color=ORANGE, weight="bold")
        ax.text(0.5, 0.07, "every solution in this video was replay-verified in the environment",
                ha="center", fontsize=17, color=MUT)
        fade(ax, i, n, V.fps)
        V.frame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "beyond_the_cube.mp4"))
    ap.add_argument("--frames",
                    default="/tmp/claude-1000/-home-akhani/26a5857e-5125-4da3-b878-d4cc581c6562/scratchpad/vidframes2")
    args = ap.parse_args()
    assert os.path.abspath(args.out) != os.path.abspath(
        os.path.join(HERE, "rubiks_diffusion.mp4")), "never overwrite video 1"

    fwd, rev, sc_states, sc_moves = collect()
    missing = [k for k in META if rev.get(k) is None]
    print(f"reverse trajectories missing for: {missing or 'none'}", flush=True)

    import shutil
    shutil.rmtree(args.frames, ignore_errors=True)
    V = Video(args.fps, args.frames)
    scene_title(V, fwd); print(f"title: {V.k}", flush=True)
    scene_recap(V, sc_states, sc_moves); print(f"recap: {V.k}", flush=True)
    scene_ladder(V, fwd); print(f"ladder: {V.k}", flush=True)
    scene_recipe(V); print(f"recipe: {V.k}", flush=True)
    for key in [d[0] for d in DOMS]:
        play_domain(V, key, fwd, rev)
        print(f"domain {key}: {V.k}", flush=True)
        if key == "soko":
            scene_dive_sokoban(V)
        elif key == "pend":
            scene_dive_pendulum(V)
        elif key == "g2048":
            scene_dive_2048(V)
        elif key == "hanoi":
            scene_dive_hanoi(V)
    scene_scoreboard(V); print(f"scoreboard: {V.k}", flush=True)
    scene_checklist(V)
    scene_outro(V)
    print(f"all scenes: {V.k} frames = {V.k/args.fps:.1f}s", flush=True)

    secs = V.k / args.fps
    pad = os.path.join(args.frames, "pad.wav")
    make_pad(pad, secs)
    os.system(
        f"ffmpeg -y -framerate {args.fps} -i {args.frames}/f_%05d.png -i {pad} "
        f"-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p "
        f"-c:a aac -b:a 96k -shortest {args.out} 2>&1 | tail -2")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
