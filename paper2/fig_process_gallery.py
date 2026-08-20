"""Per-domain process galleries: for EVERY domain a pair of rows —
  top:    the forward/noising schedule (goal -> noise, t = 0 -> 1)
  bottom: the trained reverse/denoising trajectory (noise -> goal, t = 1 -> 0)
Rendered natively per domain. Reverse rows are skipped (with a note) until
their checkpoint exists; rerun after training to complete.
Outputs: fig2_gallery_A.pdf (domains 1-5), fig2_gallery_B.pdf (6-10) + PNGs.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import PolicyNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ORANGE, BLUE, GRAY = "#eb6834", "#2a78d6", "#9a958c"
GREEN, RED = "#1e9e57", "#c9403c"
NCOL = 8
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})


def snap(seq):
    idx = sorted(set(int(round(x)) for x in np.linspace(0, len(seq) - 1, NCOL)))
    return [seq[i] for i in idx]


# ------------------------------------------------------------- renderers
def rslide(ax, b, n=5):
    b = b.reshape(n, n)
    ax.imshow((b > 0).astype(float), cmap="Oranges", vmin=-0.6, vmax=1.6)
    for r in range(n):
        for c in range(n):
            if b[r, c]:
                ax.text(c, r, str(b[r, c]), ha="center", va="center",
                        fontsize=4.2)
    ax.set_xticks([]); ax.set_yticks([])


def rgrid(cmap, vmax, n):
    def f(ax, img):
        ax.imshow(img.reshape(n, n), cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
    return f


def rpend(ax, th):
    ax.plot([0, 0], [0, 1.05], ls=":", lw=0.6, color=GRAY)
    ax.plot([0, np.sin(th)], [0, np.cos(th)], lw=2.0, color=BLUE)
    ax.plot([np.sin(th)], [np.cos(th)], "o", ms=5, color=BLUE)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal"); ax.axis("off")


def r2048(ax, b):
    b = b.reshape(4, 4)
    ax.imshow((b > 0).astype(float), cmap="Purples", vmin=-0.6, vmax=1.8)
    for r in range(4):
        for c in range(4):
            if b[r, c]:
                ax.text(c, r, str(2 ** int(b[r, c])), ha="center", va="center",
                        fontsize=4.0)
    ax.set_xticks([]); ax.set_yticks([])


def rhanoi(ax, pegs, n=10):
    heights = [0, 0, 0]
    for disk in range(n - 1, -1, -1):
        p = int(pegs[disk])
        w = 0.25 + 0.55 * (disk + 1) / n
        ax.barh(heights[p], w, left=p - w / 2, height=0.85, color=ORANGE,
                edgecolor="white", linewidth=0.2)
        heights[p] += 1
    for p in range(3):
        ax.plot([p, p], [0, n + 0.5], color=GRAY, lw=0.6, zorder=0)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, n + 1)
    ax.axis("off")


def rmc(ax, x):
    xs = np.linspace(-1.2, 0.6, 80)
    ax.plot(xs, np.sin(3 * xs) * 0.45, color=GRAY, lw=1.0)
    ax.plot([x], [np.sin(3 * x) * 0.45 + 0.06], "o", ms=6, color=BLUE)
    ax.plot([0.5], [np.sin(1.5) * 0.45 + 0.08], marker="$⚑$", ms=7,
            color=GREEN)
    ax.set_xlim(-1.3, 0.7); ax.set_ylim(-0.65, 0.75)
    ax.axis("off")


def rlo(ax, b):
    ax.imshow(b.reshape(5, 5), cmap="YlOrBr", vmin=0, vmax=1.4)
    ax.set_xticks([]); ax.set_yticks([])


def rpeg(ax, b):
    img = b.reshape(7, 7).astype(float)
    img[img == 2] = np.nan
    cm = plt.get_cmap("Oranges").copy()
    cm.set_bad("#efece6")
    ax.imshow(img, cmap=cm, vmin=-0.4, vmax=1.4)
    ax.set_xticks([]); ax.set_yticks([])


# ------------------------------------------- forward (noising) sequences
def fwd_sequences():
    """Return dict name -> (list of frames, renderer, title)."""
    from domains.slide import SlideEnv
    from domains.maze import MazeEnv, observe
    from domains.sokoban import SokobanEnv
    from domains.pendulum import rev_step as prev, UMAX
    from domains.game2048 import noising_batch as g2048_noise
    from domains.hanoi import HanoiEnv
    from domains.lightsout import LightsOutEnv
    from domains.mountaincar import rev_step as mcrev, GOAL_X, VMAX
    from domains.pegsolitaire import PegEnv

    out = {}
    g = lambda s: torch.Generator(device=DEV).manual_seed(s)

    env = SlideEnv(5, DEV)
    frames, s = [env.solved.cpu().numpy()], env.solved_batch(1)
    gg = g(1)
    for t in range(120):
        legal = env.legal_mask(s)
        u = torch.rand(1, 4, device=DEV, generator=gg).clamp_min(1e-9)
        a = (-torch.log(-torch.log(u))).masked_fill(~legal, -1e9).argmax(1)
        s = env.step(s, a)
        frames.append(s[0].cpu().numpy())
    out["slide"] = (snap(frames), rslide, "24-puzzle")

    envm = MazeEnv(15, device=DEV)
    walls, goal = envm.new_instances(1, generator=g(3))
    frames = []
    for d in np.linspace(0, 60, NCOL).astype(int):
        st = envm.scramble(walls, goal, max(int(d), 0) or 1, generator=g(7))
        frames.append(st[0].cpu().numpy())
    out["maze"] = (frames, rgrid("Greys", 4, 15), "maze")

    envs = SokobanEnv(8, 3, device=DEV)
    frames = []
    for d in np.linspace(0, 50, NCOL).astype(int):
        w, go, b, a_, _ = envs.instances_and_scramble(1, max(int(d), 1),
                                                      generator=g(5))
        frames.append(envs.render(w, go, b, a_)[0].cpu().numpy())
    out["soko"] = (frames, rgrid("YlOrBr", 6, 8), "Sokoban (pulls)")

    th = torch.zeros(1, device=DEV); om = torch.zeros(1, device=DEV)
    frames = [0.0]
    gg = g(2)
    for t in range(200):
        u = (torch.rand(1, device=DEV, generator=gg) * 2 - 1) * UMAX
        th, om = prev(th, om, u)
        frames.append(float(th))
    out["pend"] = (snap(frames), rpend, "pendulum (reverse time)")

    boards, _ = g2048_noise(64, 40, device=DEV, generator=g(4))
    # show one board across increasing K instead: regenerate at increasing K
    frames = []
    for k in np.linspace(1, 40, NCOL).astype(int):
        b, _ = g2048_noise(1, int(k), device=DEV, generator=g(6))
        frames.append(b[0].cpu().numpy() if b.shape[0] else
                      np.zeros(16, dtype=np.int8))
    out["g2048"] = (frames, r2048, "2048 (reverse play)")

    envh = HanoiEnv(10, DEV)
    frames, s = [envh.solved.cpu().numpy()], envh.solved_batch(1)
    gg = g(2)
    for t in range(1000):
        legal = envh.legal_mask(s)
        u = torch.rand(1, 6, device=DEV, generator=gg).clamp_min(1e-9)
        a = (-torch.log(-torch.log(u))).masked_fill(~legal, -1e9).argmax(1)
        s = envh.step(s, a)
        frames.append(s[0].cpu().numpy())
    out["hanoi"] = (snap(frames), rhanoi, "Hanoi")

    frames = []
    st0 = envm.scramble(walls, goal, 45, generator=g(9))
    for k, d in enumerate(np.linspace(0, 60, NCOL).astype(int)):
        stx = envm.scramble(walls, goal, max(int(d), 0) or 1, generator=g(11))
        frames.append(observe(stx, 15, 3)[0].cpu().numpy())
    out["pomdp"] = (frames, rgrid("Greys", 4, 15), "POMDP maze (agent view)")

    envl = LightsOutEnv(DEV)
    frames = []
    for d in np.linspace(1, 25, NCOL).astype(int):
        st = envl.scramble(1, int(d), generator=g(8))
        frames.append(st[0].cpu().numpy())
    out["lo"] = (frames, rlo, "Lights Out")

    x = torch.tensor([0.55], device=DEV); v = torch.tensor([0.02], device=DEV)
    frames = [0.55]
    gg = g(3)
    for t in range(250):
        a = torch.randint(3, (1,), device=DEV, generator=gg)
        x2, v2 = mcrev(x, v, a)
        ok = (x2 > -1.19) & (x2 < 0.59) & (v2.abs() < VMAX)
        x = torch.where(ok, x2, x); v = torch.where(ok, v2, v)
        frames.append(float(x))
    out["mc"] = (snap(frames), rmc, "Mountain Car (reverse time)")

    envp = PegEnv(DEV)
    frames = []
    for d in np.linspace(0, 25, NCOL).astype(int):
        st, _ = envp.scramble(1, max(int(d), 0) or 1, return_actions=True,
                              generator=g(12))
        frames.append(st[0].cpu().numpy())
    frames[0] = envp.goal_state.cpu().numpy()
    out["pegs"] = (frames, rpeg, "Peg Solitaire (un-jumps)")
    return out


# ---------------------------------------------- reverse (denoise) loaders
def rev_sequences():
    """Trained-model trajectories; entries None until ckpts exist."""
    out = {k: None for k in ("slide", "maze", "soko", "pend", "g2048",
                             "hanoi", "pomdp", "lo", "mc", "pegs")}
    sys.path.insert(0, HERE)
    try:
        import fig_reverse_gallery as frg
        out["slide"] = frg.traj_slide()
        out["maze"] = frg.traj_maze()
        out["soko"] = frg.traj_sokoban()
        pd = frg.traj_pendulum()
        out["pend"] = pd
        out["g2048"] = frg.traj_2048()
        h = frg.traj_hanoi()
        if h:
            out["hanoi"] = h
        p = frg.traj_pomdp()
        if p:
            out["pomdp"] = p
    except Exception as e:
        print("rev core err:", e)
    # new domains (post-sweep ckpts)
    try:
        from domains.lightsout import LightsOutEnv
        from domains.train_misc import rollout, verify, get_env
        for key, dom, run, steps in (("lo", "lightsout", "runs/lo_diff_s0", 60),
                                     ("pegs", "pegs", "runs/peg_diff_s0", 60)):
            ck = os.path.join(ROOT, run, "ckpt_latest.pt")
            if not os.path.exists(ck):
                continue
            env = get_env(dom, DEV)
            c = torch.load(ck, map_location=DEV, weights_only=True)
            cfg = c["cfg"]
            net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"],
                            vocab=env.vocab).to(DEV)
            net.load_state_dict(c["net"]); net.eval()
            gg = torch.Generator(device=DEV).manual_seed(3)
            st = env.scramble(64, 20 if dom == "lightsout" else 22,
                              generator=gg)
            if isinstance(st, tuple):
                st = st[0]
            solved, actions = rollout(env, net, "denoise", st, steps,
                                      forbid_repeat=(dom == "lightsout"))
            i = int(solved.float().argmax())
            moves = [int(a) for a in actions[i].tolist() if a >= 0]
            frames, s = [st[i].cpu().numpy()], st[i:i+1].clone()
            for mv in moves:
                s = env.step(s, torch.tensor([mv], device=DEV))
                frames.append(s[0].cpu().numpy())
            out[key] = (snap(frames), bool(solved[i]),
                        f"{dom} — {len(moves)} moves, greedy")
    except Exception as e:
        print("rev misc err:", e)
    try:
        from domains.mountaincar import Net, rollout as mcroll, fwd_step
        ck = os.path.join(ROOT, "runs/mc_diff_s0/ckpt_latest.pt")
        if os.path.exists(ck):
            c = torch.load(ck, map_location=DEV, weights_only=True)
            net = Net(3).to(DEV)
            net.load_state_dict(c["net"]); net.eval()
            x = torch.tensor([-0.5], device=DEV)
            v = torch.zeros(1, device=DEV)
            traj = [float(x)]
            with torch.no_grad():
                for t in range(250):
                    a = net(x, v).argmax(-1)
                    x, v = fwd_step(x, v, a)
                    traj.append(float(x))
                    if float(x) >= 0.5:
                        break
            out["mc"] = (snap(traj), float(traj[-1]) >= 0.5,
                         f"mountain car — reached in {len(traj)-1} steps")
    except Exception as e:
        print("rev mc err:", e)
    return out


def reference_rows():
    """Familiar diffusion anchors: spiral particles + a real photo, noised on
    the same t grid — shown atop every gallery so each domain's strip reads
    as 'the same schedule, a different state space'."""
    rng = np.random.default_rng(3)
    sp = rng.uniform(0, 1, 1200)
    ang = 4 * np.pi * sp + 0.4
    r = 0.15 + 0.75 * sp
    x0 = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
    x0 += rng.normal(0, 0.02, x0.shape)
    eps = rng.standard_normal(x0.shape)
    spiral_frames, img_frames = [], []
    try:
        from skimage.data import astronaut
        from skimage.transform import resize
        img0 = resize(astronaut(), (160, 160), anti_aliasing=True).astype(np.float32)
    except Exception:
        img0 = np.ones((160, 160, 3), dtype=np.float32) * 0.5
    rngI = np.random.default_rng(0)
    epsI = rngI.standard_normal(img0.shape)
    for k in range(NCOL):
        t = k / (NCOL - 1)
        # quadratic noise growth: early frames stay recognizable, matching
        # the gradual corruption of the domain rows below
        sig = 0.55 * t ** 2
        shrink = 1.0 - 0.55 * t ** 2
        spiral_frames.append(shrink * x0 + sig * eps)
        sigI = 1.0 * t ** 2
        img_frames.append(np.clip(np.sqrt(max(1 - sigI ** 2, 1e-4)) * img0 +
                                  sigI * epsI, 0, 1))
    return spiral_frames, img_frames


def rspiral(ax, pts):
    ax.scatter(pts[:, 0], pts[:, 1], s=0.9, c=BLUE, alpha=0.7, linewidths=0)
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal"); ax.axis("off")


def rimg(ax, arr):
    ax.imshow(arr)
    ax.set_xticks([]); ax.set_yticks([])


ORDER_A = ["slide", "maze", "soko", "pend", "g2048"]
ORDER_B = ["hanoi", "pomdp", "lo", "mc", "pegs"]


def build(name, order, fwd, rev):
    rows = [("__ref_spiral", "ref", None), ("__ref_img", "ref", None)]
    for k in order:
        rows.append((k, "fwd", fwd[k]))
        rows.append((k, "rev", rev.get(k)))
    R = len(rows)
    spiral_frames, img_frames = reference_rows()
    fig, axes = plt.subplots(R, NCOL, figsize=(7.0, 0.92 * R))
    for r, (k, kind, payload) in enumerate(rows):
        if kind == "ref":
            frames = spiral_frames if k == "__ref_spiral" else img_frames
            rend0 = rspiral if k == "__ref_spiral" else rimg
            for c in range(NCOL):
                rend0(axes[r, c], frames[c])
                if r == 0:
                    axes[r, c].set_title(f"$t={c/(NCOL-1):.2f}$", fontsize=7)
            lab = ("particles (ref.)" if k == "__ref_spiral"
                   else "image (ref.)")
            axes[r, 0].text(-0.24, 0.5, lab, rotation=90, fontsize=5.8,
                            color="#6b6a66", transform=axes[r, 0].transAxes,
                            va="center", ha="center")
            continue
        rend = fwd[k][1]
        if kind == "fwd":
            frames, _, title = payload
            for c in range(NCOL):
                ax = axes[r, c]
                if c < len(frames):
                    rend(ax, frames[c])
                else:
                    ax.axis("off")
            axes[r, 0].text(-0.30, 0.5, title, rotation=90, fontsize=7,
                            transform=axes[r, 0].transAxes, va="center",
                            ha="center")
            axes[r, 0].text(-0.52, 0.5, "noising →", rotation=90, fontsize=6,
                            color=GRAY, transform=axes[r, 0].transAxes,
                            va="center", ha="center")
        else:
            if payload is None:
                for c in range(NCOL):
                    axes[r, c].axis("off")
                axes[r, NCOL // 2].text(0.5, 0.5, "(denoising row: training "
                                        "in progress)", fontsize=7,
                                        color=GRAY, ha="center",
                                        transform=axes[r, NCOL // 2].transAxes)
                continue
            frames, ok, label = payload
            for c in range(NCOL):
                ax = axes[r, c]
                if c < len(frames):
                    rend(ax, frames[c])
                else:
                    ax.axis("off")
            mark, col = ("✓", GREEN) if ok else ("✗", RED)
            axes[r, -1].text(1.15, 0.5, mark, fontsize=12, color=col,
                             weight="bold", va="center",
                             transform=axes[r, -1].transAxes)
            axes[r, 0].text(-0.52, 0.5, "← denoising", rotation=90, fontsize=6,
                            color=ORANGE, transform=axes[r, 0].transAxes,
                            va="center", ha="center")
            axes[r, 0].text(0.0, -0.22, label, fontsize=5.8, color="#6b6a66",
                            transform=axes[r, 0].transAxes, ha="left")
    fig.tight_layout(w_pad=0.25, h_pad=0.7)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig2_gallery_{name}.{ext}"),
                    bbox_inches="tight", dpi=170)
    print(f"fig2_gallery_{name}.pdf/.png")


def main():
    fwd = fwd_sequences()
    rev = rev_sequences()
    build("A", ORDER_A, fwd, rev)
    build("B", ORDER_B, fwd, rev)


if __name__ == "__main__":
    main()
