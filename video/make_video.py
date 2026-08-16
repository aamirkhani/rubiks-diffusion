"""Render the YouTube explainer video (1080p30, ~75s).

Every cube state is a real engine state; the centerpiece solve is the actual
trained denoiser's greedy rollout (replay-verified), animated with smooth
90-degree layer turns. Frames -> ffmpeg (H.264) + a generated ambient pad.

Usage:  python video/make_video.py [--fps 30] [--out video/rubiks_diffusion.mp4]
"""
import argparse
import os
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "paper"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import torch
import torch.nn.functional as F

from cube_env import CubeEnv, _sticker_geometry, FACE_AXES
from model import PolicyNet
from solve import verify_solutions
from solve_policy import policy_greedy

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BG = "#0e0d0c"
INK = "#f5f4f0"
MUT = "#a09a90"
ORANGE = "#eb6834"
BLUE = "#5598e7"
GREEN = "#35d97e"
FACE_HEX = ["#f6f6f4", "#c41e3a", "#009b48", "#ffd500", "#ff5800", "#0046ad"]

env3 = CubeEnv(3, DEV)


# ----------------------------------------------------------------- cube draw
def _rodrigues(p, a, t):
    a = np.asarray(a, float)
    c, s = np.cos(t), np.sin(t)
    return (p * c + np.cross(a, p) * s + a * np.dot(p, a) * (1 - c))


def draw_cube(ax, state, n=3, yaw=-0.62, pitch=0.42, move=None, frac=0.0,
              scale=1.0, lw=1.6):
    """Solid-cubie isometric cube: dark cubie bodies + colored stickers, so a
    mid-turn layer looks like a physical cube. If move (env move index) is
    given, its layer is rotated by frac of a quarter turn."""
    from itertools import product
    pos, nrm = _sticker_geometry(n)
    pos = pos - n / 2.0
    axis, ang = None, 0.0
    if move is not None and frac > 0:
        name = env3.move_names[move]
        axis = np.asarray(FACE_AXES[name[0]], float)
        ang = (np.pi / 2) * frac * (1.0 if "'" in name else -1.0)

    def in_layer(center):
        return axis is not None and center @ axis > n / 2 - 1 + 1e-6

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    def view(p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        x1 = x * cy + z * sy
        z1 = -x * sy + z * cy
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp
        return np.stack([x1, y2, z2], -1)

    quads, cols, deps, lws = [], [], [], []

    def add_quad(q3, color, lwq, normal=None):
        # backface culling: a quad facing away from the camera on a convex
        # cubie is never visible
        if normal is not None and view(np.asarray(normal))[2] <= 0.02:
            return
        q = view(np.asarray(q3))
        quads.append(q[:, :2]); deps.append(q[:, 2].mean())
        cols.append(color); lws.append(lwq)

    # cubie bodies
    h = 0.5 * 0.97
    E = np.eye(3)
    for a, b, c in product(range(n), repeat=3):
        cen = np.array([a + 0.5, b + 0.5, c + 0.5]) - n / 2.0
        basis = E.copy()
        if in_layer(cen):
            cen = _rodrigues(cen, axis, ang)
            basis = np.stack([_rodrigues(E[k], axis, ang) for k in range(3)])
        for k in range(3):
            e, u, v = basis[k], basis[(k + 1) % 3], basis[(k + 2) % 3]
            for sgn in (1, -1):
                fc = cen + e * h * sgn
                add_quad([fc + (u + v) * h, fc + (u - v) * h,
                          fc - (u + v) * h, fc - (u - v) * h], "#191816", 0.5,
                         normal=e * sgn)

    # stickers (offset slightly outward so they sort in front of their body)
    half = 0.42
    for i in range(6 * n * n):
        p, m = pos[i].copy(), nrm[i].copy()
        own = p - 0.5 * nrm[i]
        if in_layer(own):
            p = _rodrigues(p, axis, ang)
            m = _rodrigues(m, axis, ang)
        up = np.array([0, 0, 1.0]) if abs(m[1]) > 0.9 else np.array([0, 1.0, 0])
        u = np.cross(m, up)
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            u = np.cross(m, np.array([1.0, 0, 0])); nu = np.linalg.norm(u)
        u /= nu
        v = np.cross(m, u)
        p = p + m * 0.03
        add_quad([p + (u + v) * half, p + (u - v) * half,
                  p - (u + v) * half, p - (u - v) * half],
                 FACE_HEX[int(state[i])], lw, normal=m)

    order = np.argsort(deps)
    pc = PolyCollection([quads[k] for k in order],
                        facecolors=[cols[k] for k in order],
                        edgecolors="#0b0a09",
                        linewidths=[lws[k] for k in order], joinstyle="round")
    ax.add_collection(pc)
    lim = n * 0.98 / scale
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off")


# ------------------------------------------------------------------- assets
def get_solve_trajectory():
    ck = torch.load(os.path.join(ROOT, "runs/3x3_diff/ckpt_latest.pt"),
                    map_location=DEV, weights_only=True)
    net = PolicyNet(env3.S, env3.M, ck["cfg"]["h1"], ck["cfg"]["h2"],
                    ck["cfg"]["blocks"]).to(DEV)
    net.load_state_dict(ck["net"]); net.eval()
    best = None
    for seed in range(400):
        g = torch.Generator(device=DEV).manual_seed(seed)
        st = env3.scramble(1, 100, generator=g)
        solved, lengths, actions = policy_greedy(env3, net, st, max_steps=60)
        if solved[0]:
            L = int(lengths[0])
            if best is None or abs(L - 22) < abs(best[1] - 22):
                best = (st, L, actions)
            if 18 <= L <= 26:
                break
    st, L, actions = best
    assert verify_solutions(env3, st, actions)[0]
    moves = [int(a) for a in actions[0, :L].tolist()]
    states, confs = [st[0].cpu().numpy()], []
    s = st.clone()
    for mv in moves:
        with torch.no_grad():
            p = F.softmax(net(s)[0].float(), 0)[mv].item()
        confs.append(p)
        s = env3.step(s, torch.tensor([mv], device=DEV))
        states.append(s[0].cpu().numpy())
    # also a real 12-move scramble sequence for the analogy scene
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
    # example logits for the model scene (state = 6-move scramble)
    g = torch.Generator(device=DEV).manual_seed(4)
    stx, acts = env3.scramble(1, 6, return_actions=True, generator=g)
    with torch.no_grad():
        probs = F.softmax(net(stx)[0].float(), 0).cpu().numpy()
    return dict(solve_states=states, solve_moves=moves, solve_confs=confs,
                scr_states=sc_states, scr_moves=sc_moves,
                model_state=stx[0].cpu().numpy(), model_probs=probs)


def spiral_pts(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, n)
    ang = 4 * np.pi * s + 0.4
    r = 0.15 + 0.75 * s
    pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
    return pts + rng.normal(0, 0.02, pts.shape)


def astro():
    from skimage.data import astronaut
    from skimage.transform import resize
    return resize(astronaut(), (240, 240), anti_aliasing=True).astype(np.float32)


def ease(x):
    return 3 * x * x - 2 * x ** 3


# ------------------------------------------------------------------- scenes
class Video:
    def __init__(self, fps, outdir):
        self.fps = fps
        self.outdir = outdir
        self.k = 0
        os.makedirs(outdir, exist_ok=True)
        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

    def frame(self):
        self.fig.savefig(os.path.join(self.outdir, f"f_{self.k:05d}.png"),
                         facecolor=BG)
        self.k += 1
        self.fig.clf()

    def canvas(self):
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_facecolor(BG)
        self.fig.patch.set_facecolor(BG)
        return ax


def scene_title(V, A, secs=4.5):
    n = int(secs * V.fps)
    for i in range(n):
        t = i / n
        ax = V.canvas()
        a = min(1, t / 0.25)
        ax.text(0.36, 0.60, "Solving a Rubik's Cube", ha="center", fontsize=54,
                color=INK, alpha=a, weight="bold")
        ax.text(0.36, 0.50, "like a Diffusion Model", ha="center", fontsize=54,
                color=ORANGE, alpha=a, weight="bold")
        ax.text(0.36, 0.38, "noise = random moves  ·  denoising = solving",
                ha="center", fontsize=22, color=MUT, alpha=a)
        axc = V.fig.add_axes([0.62, 0.18, 0.34, 0.64])
        axc.set_facecolor(BG)
        draw_cube(axc, env3.solved.cpu().numpy(), yaw=-0.62 + 0.35 * np.sin(t * 2.2),
                  pitch=0.42 + 0.1 * np.sin(t * 1.3))
        V.frame()


def scene_diffusion(V, A, secs=9):
    n = int(secs * V.fps)
    x0 = spiral_pts()
    eps = np.random.default_rng(0).standard_normal(x0.shape)
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.90, "Diffusion models destroy data with noise…",
                ha="center", fontsize=34, color=INK)
        if t > 0.45:
            ax.text(0.5, 0.83, "…then learn to reverse it, one step at a time.",
                    ha="center", fontsize=34, color=ORANGE,
                    alpha=min(1, (t - 0.45) / 0.15))
        # forward for first 55%, reverse replay after
        f = ease(min(1, t / 0.55)) if t < 0.55 else ease(max(0, 1 - (t - 0.55) / 0.4))
        ab = np.clip((1 - f) ** 2, 1e-4, 1 - 1e-4)
        pts = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps * 0.55
        axp = V.fig.add_axes([0.30, 0.10, 0.40, 0.66])
        axp.set_facecolor(BG)
        axp.scatter(pts[:, 0], pts[:, 1], s=3.2, c=BLUE, alpha=0.75, linewidths=0)
        axp.set_xlim(-1.8, 1.8); axp.set_ylim(-1.8, 1.8)
        axp.set_aspect("equal"); axp.axis("off")
        lab = "forward: adding noise" if t < 0.55 else "reverse: denoising"
        axp.set_title(lab, color=MUT if t < 0.55 else ORANGE, fontsize=20)
        V.frame()


def scene_analogy(V, A, secs=12):
    x0 = astro()
    epsI = np.random.default_rng(1).standard_normal(x0.shape)
    sc_states, sc_moves = A["scr_states"], A["scr_moves"]
    n = int(secs * V.fps)
    per = n / len(sc_moves)
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.92, "On a cube, the noise is random moves",
                ha="center", fontsize=36, color=INK, weight="bold")
        ax.text(0.27, 0.13, "images: Gaussian noise", ha="center", fontsize=22, color=MUT)
        ax.text(0.73, 0.13, "cubes: random turns — scramble depth = timestep $t$",
                ha="center", fontsize=22, color=ORANGE)
        ab = max(1e-4, (1 - ease(t)) ** 2)
        xt = np.clip(np.sqrt(ab) * x0 + np.sqrt(1 - ab) * epsI, 0, 1)
        axi = V.fig.add_axes([0.11, 0.20, 0.32, 0.60])
        axi.imshow(xt); axi.axis("off")
        j = min(len(sc_moves) - 1, int(i / per))
        frac = ease(min(1.0, (i - j * per) / (per * 0.7)))
        axc = V.fig.add_axes([0.55, 0.14, 0.38, 0.70])
        axc.set_facecolor(BG)
        draw_cube(axc, sc_states[j], move=sc_moves[j], frac=frac,
                  yaw=-0.62 + 0.10 * np.sin(t * 3))
        axc.set_title(f"t = {j+1} moves", color=INK, fontsize=22)
        V.frame()


def scene_model(V, A, secs=7):
    n = int(secs * V.fps)
    probs = A["model_probs"]
    names = env3.move_names
    top = int(np.argmax(probs))
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.91, "A network learns one thing:", ha="center",
                fontsize=34, color=INK)
        ax.text(0.5, 0.84, "predict the move that undoes the last scramble step",
                ha="center", fontsize=34, color=ORANGE, weight="bold")
        axc = V.fig.add_axes([0.06, 0.22, 0.26, 0.5])
        axc.set_facecolor(BG)
        draw_cube(axc, A["model_state"])
        ax.annotate("", xy=(0.44, 0.47), xytext=(0.345, 0.47),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color=INK))
        # net glyph
        cols = [(0.47, 6), (0.53, 8), (0.59, 8), (0.64, 4)]
        for (x1, _), (x2, _) in zip(cols[:-1], cols[1:]):
            for y1 in np.linspace(0.35, 0.60, 6):
                for y2 in np.linspace(0.35, 0.60, 6):
                    ax.plot([x1, x2], [y1, y2], lw=0.3, color="#3c3833", zorder=1)
        for x, nn in cols:
            ax.scatter([x] * nn, np.linspace(0.33, 0.62, nn), s=42,
                       c=ORANGE if x == cols[-1][0] else "#8a7a6d", zorder=2)
        ax.text(0.555, 0.27, "denoiser  $p_\\theta(a\\,|\\,s)$", ha="center",
                fontsize=20, color=MUT)
        ax.annotate("", xy=(0.72, 0.47), xytext=(0.66, 0.47),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color=INK))
        axb = V.fig.add_axes([0.73, 0.28, 0.24, 0.42])
        axb.set_facecolor(BG)
        grow = ease(min(1, max(0, (t - 0.2) / 0.4)))
        ccs = [ORANGE if k == top else "#54493f" for k in range(12)]
        axb.bar(range(12), probs * grow, color=ccs, width=0.72)
        axb.set_xticks(range(12), names, fontsize=13, color=INK)
        axb.set_yticks([]); axb.set_ylim(0, 1.02)
        for s_ in axb.spines.values():
            s_.set_visible(False)
        axb.tick_params(colors=INK)
        if t > 0.62:
            axb.set_title(f"→ undo with  {names[top]}", color=GREEN,
                          fontsize=24, weight="bold")
        V.frame()


def scene_solve(V, A, secs_per_move=0.62, hold=2.2):
    states, moves, confs = A["solve_states"], A["solve_moves"], A["solve_confs"]
    L = len(moves)
    ax = None
    # intro hold on the scrambled cube
    for i in range(int(1.6 * V.fps)):
        ax = V.canvas()
        ax.text(0.5, 0.93, "Now watch the reverse process solve a real scramble",
                ha="center", fontsize=36, color=INK, weight="bold")
        ax.text(0.5, 0.87, "trained model, greedy — no search, no human algorithm",
                ha="center", fontsize=22, color=MUT)
        axc = V.fig.add_axes([0.25, 0.06, 0.5, 0.76])
        axc.set_facecolor(BG)
        draw_cube(axc, states[0])
        ax.text(0.5, 0.075, "scrambled with 100 random moves", ha="center",
                fontsize=22, color=ORANGE)
        V.frame()
    pm = int(secs_per_move * V.fps)
    rot = int(pm * 0.66)
    for j, mv in enumerate(moves):
        for i in range(pm):
            frac = ease(min(1.0, i / rot))
            ax = V.canvas()
            ax.text(0.5, 0.945, "Reverse diffusion  =  solving", ha="center",
                    fontsize=34, color=INK, weight="bold")
            axc = V.fig.add_axes([0.22, 0.05, 0.56, 0.82])
            axc.set_facecolor(BG)
            yaw = -0.62 + 0.22 * np.sin((j * pm + i) / (V.fps * 9.0))
            draw_cube(axc, states[j], move=mv, frac=frac, yaw=yaw)
            ax.text(0.865, 0.62, env3.move_names[mv], ha="center", fontsize=64,
                    color=ORANGE, weight="bold")
            ax.text(0.865, 0.53, f"confidence {confs[j]*100:.0f}%", ha="center",
                    fontsize=20, color=MUT)
            ax.text(0.865, 0.44, f"move {j+1} / {L}", ha="center", fontsize=20,
                    color=INK)
            ax.add_patch(plt.Rectangle((0.80, 0.38), 0.13, 0.012, color="#3c3833"))
            ax.add_patch(plt.Rectangle((0.80, 0.38), 0.13 * (j + frac) / L, 0.012,
                                       color=ORANGE))
            ax.text(0.135, 0.53, "denoiser picks\none move at a time",
                    ha="center", fontsize=20, color=MUT)
            V.frame()
    for i in range(int(hold * V.fps)):
        t = i / (hold * V.fps)
        ax = V.canvas()
        axc = V.fig.add_axes([0.22, 0.05, 0.56, 0.82])
        axc.set_facecolor(BG)
        draw_cube(axc, states[-1], yaw=-0.62 + 0.5 * t)
        ax.text(0.5, 0.93, f"SOLVED — {L} moves", ha="center", fontsize=48,
                color=GREEN, weight="bold", alpha=min(1, t / 0.2))
        ax.text(0.5, 0.075, "every solution is verified by replaying it in the environment",
                ha="center", fontsize=20, color=MUT)
        V.frame()


def scene_results(V, A, secs=9):
    n = int(secs * V.fps)
    rows = [
        ("2×2×2:  100.000000% of all 3,674,160 states solved", ORANGE, 0.10),
        ("3×3×3:  1000 / 1000 fully scrambled cubes solved", ORANGE, 0.28),
        ("trained in 4.9 h on one laptop GPU", INK, 0.46),
        ("2.8× faster than the value-iteration baseline (DeepCubeA-style)", INK, 0.60),
        ("every solve replay-verified · exact-oracle validation", MUT, 0.74),
    ]
    for i in range(n):
        t = i / n
        ax = V.canvas()
        ax.text(0.5, 0.90, "Does it work?", ha="center", fontsize=44,
                color=INK, weight="bold")
        for txt, col, start in rows:
            if t > start:
                a = min(1, (t - start) / 0.10)
                y = 0.72 - 0.13 * rows.index((txt, col, start))
                ax.text(0.5, y, txt, ha="center", fontsize=30, color=col, alpha=a)
        V.frame()


def scene_outro(V, A, secs=6):
    n = int(secs * V.fps)
    for i in range(n):
        t = i / n
        ax = V.canvas()
        axc = V.fig.add_axes([0.36, 0.44, 0.28, 0.5])
        axc.set_facecolor(BG)
        draw_cube(axc, env3.solved.cpu().numpy(), yaw=-0.62 + t * 1.2)
        ax.text(0.5, 0.34, "Scramble Inversion as Discrete Denoising Diffusion",
                ha="center", fontsize=30, color=INK, weight="bold")
        ax.text(0.5, 0.26, "paper + code + trained models", ha="center",
                fontsize=22, color=MUT)
        ax.text(0.5, 0.18, "github.com/aamirkhani/rubiks-diffusion", ha="center",
                fontsize=30, color=ORANGE, weight="bold")
        V.frame()


# ------------------------------------------------------------------- audio
def make_pad(path, secs, sr=22050):
    """Soft generated ambient pad (two alternating chords), safely original."""
    t = np.arange(int(secs * sr)) / sr
    chords = [[130.81, 164.81, 196.0, 246.94],    # Cmaj7
              [110.0, 130.81, 164.81, 220.0]]     # Am
    seg = 8.0
    x = np.zeros_like(t)
    for k, f0 in enumerate(np.arange(0, secs, seg)):
        ch = chords[k % 2]
        idx = (t >= f0) & (t < f0 + seg)
        tt = t[idx] - f0
        env = np.minimum(tt / 2.5, 1) * np.minimum((seg - tt) / 2.5, 1)
        env = np.clip(env, 0, 1)
        for f in ch:
            x[idx] += env * (np.sin(2 * np.pi * f * tt) * 0.5 +
                             np.sin(2 * np.pi * f * 2 * tt) * 0.08)
    x *= 0.05 * np.minimum(t / 3, 1) * np.minimum((secs - t) / 3, 1).clip(0, 1)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "rubiks_diffusion.mp4"))
    ap.add_argument("--frames", default="/tmp/claude-1000/-home-akhani/26a5857e-5125-4da3-b878-d4cc581c6562/scratchpad/vidframes")
    args = ap.parse_args()

    print("collecting real trajectories...", flush=True)
    A = get_solve_trajectory()
    print(f"solve length: {len(A['solve_moves'])} moves", flush=True)

    import shutil
    shutil.rmtree(args.frames, ignore_errors=True)
    V = Video(args.fps, args.frames)
    for fn in (scene_title, scene_diffusion, scene_analogy, scene_model):
        fn(V, A)
        print(f"{fn.__name__}: {V.k} frames total", flush=True)
    scene_solve(V, A)
    print(f"scene_solve: {V.k} frames total", flush=True)
    scene_results(V, A)
    scene_outro(V, A)
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
