"""Visual figures for the paper: forward-process comparison, architecture with
real input/output, and an actual reverse-process (solve) trajectory.

Everything is generated from the real engine and the trained checkpoints —
no mock data.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch
import torch.nn.functional as F

from cube_env import CubeEnv
from model import PolicyNet, ValueNet
from cube_render import render_cube, cube_image

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#1a1a19", "#6b6a66"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "figure.dpi": 150,
})

env3 = CubeEnv(3, DEV)
g = torch.Generator(device=DEV).manual_seed(11)


def landscape_image(px=200):
    """Synthetic landscape (sky, sun, hills, tree) — an obviously non-cube
    'natural image' for the Gaussian-diffusion row."""
    fig, ax = plt.subplots(figsize=(2, 2), dpi=px // 2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    # sky gradient
    grad = np.linspace(0, 1, 128)[:, None] * np.ones((1, 128))
    ax.imshow(grad, extent=(0, 1, 0.35, 1), origin="lower", cmap="Blues_r",
              vmin=-0.4, vmax=1.6, aspect="auto", zorder=0)
    ax.add_patch(plt.Circle((0.76, 0.8), 0.10, color="#ffd75e", zorder=1))
    ax.fill_between([0, 0.25, 0.5, 0.75, 1], [0.35, 0.62, 0.42, 0.58, 0.38],
                    0.3, color="#7fa96b", zorder=2)
    ax.fill_between([0, 0.3, 0.6, 1], [0.42, 0.35, 0.5, 0.4], 0,
                    color="#4d7a45", zorder=3)
    ax.add_patch(plt.Rectangle((0.18, 0.30), 0.035, 0.14, color="#6b4a2b", zorder=4))
    ax.add_patch(plt.Circle((0.198, 0.50), 0.085, color="#2f5d2a", zorder=4))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    plt.close(fig)
    return buf


def scrambled_state(t, seed=11):
    gg = torch.Generator(device=DEV).manual_seed(seed)
    if t == 0:
        return env3.solved.cpu().numpy()
    return env3.scramble(1, t, generator=gg)[0].cpu().numpy()


# ---------------------------------------------------------------- forward
def fig_forward():
    """Three generations of the same forward process, at matched noise levels.
    Row 1: particle diffusion on the swiss-roll density (the setting of the
    original diffusion paper, Sohl-Dickstein et al. 2015) — our own rendering.
    Row 2: Gaussian image diffusion (DDPM linear beta schedule) of a rendered
    cube photo. Row 3: the group forward process — the cube after t random
    generator moves."""
    T = 1000
    betas = np.linspace(1e-4, 0.02, T)              # DDPM linear schedule
    abar = np.cumprod(1 - betas)
    img_ts = [0, 50, 150, 300, 600, 1000]
    cube_ts = [0, 1, 3, 6, 12, 30]

    # swiss roll "particles" (Sohl-Dickstein et al. 2015 demo distribution)
    rng = np.random.default_rng(3)
    n_pts = 1600
    th = rng.uniform(0.6, 3.0, n_pts) * 2 * np.pi * 0.55
    r = th / (2 * np.pi * 0.55 * 3.0)
    roll = np.stack([r * np.cos(th * 2.1), r * np.sin(th * 2.1)], 1)
    roll += rng.normal(0, 0.012, roll.shape)
    roll /= np.abs(roll).max() * 1.15

    # real photograph (public domain NASA portrait, the classic test image)
    from skimage.data import astronaut
    from skimage.transform import resize
    x0 = resize(astronaut(), (200, 200), anti_aliasing=True).astype(np.float32)
    rngI = np.random.default_rng(0)

    fig, axes = plt.subplots(3, len(img_ts), figsize=(7.0, 3.95))
    for k, t in enumerate(img_ts):
        a = 1.0 if t == 0 else abar[t - 1]
        # particles: q(x_t|x_0) applied to every point
        pts = np.sqrt(a) * roll + np.sqrt(1 - a) * rng.standard_normal(roll.shape) * 0.55
        ax = axes[0, k]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.7, c=BLUE, alpha=0.6, linewidths=0)
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"$t/T={t/T:.2f}$", fontsize=8, pad=2)
        # image
        ax = axes[1, k]
        xt = x0 if t == 0 else np.sqrt(a) * x0 + np.sqrt(1 - a) * rngI.standard_normal(x0.shape)
        ax.imshow(np.clip(xt, 0, 1))
        ax.axis("off")
    for k, t in enumerate(cube_ts):
        ax = axes[2, k]
        render_cube(ax, scrambled_state(t))
        ax.set_title(f"$t={t}$ moves", fontsize=8, pad=2)
    row_labels = [
        "particle diffusion\n(Sohl-Dickstein '15\nsetting)",
        "Gaussian diffusion\n(image, linear $\\beta_t$)",
        "group diffusion\n(scramble, linear $t$)",
    ]
    for r_, lab in enumerate(row_labels):
        axes[r_, 0].text(-0.30, 0.5, lab, transform=axes[r_, 0].transAxes,
                         rotation=90, va="center", ha="center", fontsize=7.6, color=INK)
    fig.tight_layout(w_pad=0.4, h_pad=0.7)
    fig.savefig(os.path.join(HERE, "fig_forward_process.pdf"), bbox_inches="tight")
    print("fig_forward_process.pdf")


# ------------------------------------------------------------ architecture
def _box(ax, x, y, w, h, label, sub=None, fc="#f0efec", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=INK, lw=lw))
    cy = y + h / 2 + (0.018 if sub else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center", fontsize=8, color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.028, sub, ha="center", va="center",
                fontsize=6.8, color=MUTED)


def _arrow(ax, x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.1, color=color))


def fig_architecture():
    """Real forward pass: scrambled state -> one-hot -> backbone -> both heads,
    with the trained denoiser's actual logits and the trained DAVI net's value."""
    ck = torch.load(os.path.join(ROOT, "runs/3x3_diff/ckpt_latest.pt"),
                    map_location=DEV, weights_only=True)
    pnet = PolicyNet(env3.S, env3.M, ck["cfg"]["h1"], ck["cfg"]["h2"],
                     ck["cfg"]["blocks"]).to(DEV)
    pnet.load_state_dict(ck["net"]); pnet.eval()
    ckv = torch.load(os.path.join(ROOT, "runs/3x3_v1/ckpt_latest.pt"),
                     map_location=DEV, weights_only=True)
    vnet = ValueNet(env3.S, ckv["cfg"]["h1"], ckv["cfg"]["h2"],
                    ckv["cfg"]["blocks"]).to(DEV)
    vnet.load_state_dict(ckv["net"]); vnet.eval()

    # a real scrambled state; last scramble move known -> true label
    gg = torch.Generator(device=DEV).manual_seed(4)
    st, acts = env3.scramble(1, 8, return_actions=True, generator=gg)
    true_inv = int(env3.inverse_action[acts[0, 7]].item())
    with torch.no_grad():
        logits = pnet(st)[0].float().cpu().numpy()
        probs = F.softmax(torch.tensor(logits), dim=0).numpy()
        value = float(vnet(st)[0].float().item())
    state_np = st[0].cpu().numpy()

    fig = plt.figure(figsize=(7.0, 2.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # input cube (real render)
    axc = fig.add_axes([0.015, 0.28, 0.145, 0.5]); render_cube(axc, state_np)
    ax.text(0.088, 0.84, "input state $s$", ha="center", fontsize=8, color=INK)
    ax.text(0.088, 0.20, "54 stickers\n(8-move scramble)", ha="center",
            fontsize=6.8, color=MUTED)

    # one-hot heatmap (real)
    axh = fig.add_axes([0.20, 0.30, 0.075, 0.46])
    onehot = np.eye(6)[state_np]                       # [54, 6]
    axh.imshow(onehot, aspect="auto", cmap="Greys", interpolation="nearest")
    axh.set_xticks([]); axh.set_yticks([])
    for s in axh.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.6)
    ax.text(0.2375, 0.84, "one-hot", ha="center", fontsize=8, color=INK)
    ax.text(0.2375, 0.22, "$54{\\times}6\\to324$", ha="center", fontsize=6.8, color=MUTED)

    # backbone
    _box(ax, 0.310, 0.36, 0.108, 0.34, "Linear 5000", "LN + ReLU")
    _box(ax, 0.443, 0.36, 0.108, 0.34, "Linear 1000", "LN + ReLU")
    _box(ax, 0.576, 0.36, 0.124, 0.34, "ResBlock 1000", "$\\times 4$ (LN, ReLU)")
    # residual skip arc
    ax.annotate("", xy=(0.688, 0.73), xytext=(0.585, 0.73),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=MUTED,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(0.6375, 0.815, "skip", fontsize=6.5, color=MUTED, ha="center")

    _arrow(ax, 0.163, 0.53, 0.197, 0.53)
    _arrow(ax, 0.278, 0.53, 0.312, 0.53)
    _arrow(ax, 0.422, 0.53, 0.442, 0.53)
    _arrow(ax, 0.552, 0.53, 0.572, 0.53)
    _arrow(ax, 0.702, 0.60, 0.735, 0.70, color=ORANGE)
    _arrow(ax, 0.702, 0.46, 0.735, 0.33, color=BLUE)

    # denoiser head: real logits bar chart
    axl = fig.add_axes([0.755, 0.56, 0.225, 0.36])
    names = env3.move_names
    cols = [ORANGE if i == int(np.argmax(logits)) else "#e2c4b4" for i in range(12)]
    axl.bar(range(12), probs, color=cols, width=0.75)
    axl.set_xticks(range(12), names, fontsize=5.6, rotation=0)
    axl.set_yticks([])
    axl.set_title(f"denoiser head: $p_\\theta(a\\,|\\,s)$   "
                  f"argmax $=$ {names[int(np.argmax(logits))]}"
                  f"{' = true inverse' if int(np.argmax(logits)) == true_inv else ''}",
                  fontsize=7.2, color=INK)
    for s in ("top", "right", "left"):
        axl.spines[s].set_visible(False)

    # DAVI head: real scalar
    _box(ax, 0.755, 0.12, 0.225, 0.20,
         f"DAVI head:  $V_\\theta(s) = {value:.2f}$",
         "single scalar (cost-to-go)", fc="#e3edf9")
    ax.text(0.5, 0.045, "backbone identical for both objectives — only the head and loss differ",
            ha="center", fontsize=7.2, color=MUTED, style="italic")
    fig.savefig(os.path.join(HERE, "fig_architecture.pdf"), bbox_inches="tight")
    print("fig_architecture.pdf")


# ------------------------------------------------- tiny real 2-D DDPM
def _spiral_points(n, rng):
    s = rng.uniform(0, 1, n)
    ang = 4 * np.pi * s + 0.4
    r = 0.15 + 0.75 * s
    pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
    pts += rng.normal(0, 0.02, pts.shape)
    return pts


def train_spiral_ddpm(T=200, iters=40000, batch=4096):
    """Train an actual epsilon-prediction DDPM on the spiral density; return
    reverse-trajectory snapshots. Schedule chosen so abar_T ~ 2e-3 (the forward
    endpoint really is ~pure noise, matching the N(0,1) sampling start)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    betas = torch.linspace(1e-4, 0.06, T, device=DEV)
    alphas = 1 - betas
    abar = torch.cumprod(alphas, 0)

    ws = torch.logspace(0, 3, 8, device=DEV)          # non-periodic on [0,1]

    def temb(tfrac):
        f = tfrac * ws
        return torch.cat([tfrac, torch.sin(f), torch.cos(f)], 1)

    net = torch.nn.Sequential(
        torch.nn.Linear(2 + 17, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, 2)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for it in range(iters):
        if it == int(iters * 0.8):
            for gp in opt.param_groups:
                gp["lr"] = 1e-4
        x0 = torch.tensor(_spiral_points(batch, rng), dtype=torch.float32, device=DEV)
        t = torch.randint(0, T, (batch,), device=DEV)
        a = abar[t].unsqueeze(1)
        eps = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        pred = net(torch.cat([xt, temb((t.float() / T).unsqueeze(1))], 1))
        loss = F.mse_loss(pred, eps)
        opt.zero_grad(); loss.backward(); opt.step()

    # ancestral sampling, recording snapshots
    with torch.no_grad():
        x = torch.randn(1600, 2, device=DEV)
        snaps = {T: x.cpu().numpy().copy()}
        for t in range(T - 1, -1, -1):
            tt = torch.full((x.shape[0], 1), t / T, device=DEV)
            eps = net(torch.cat([x, temb(tt)], 1))
            a, ab = alphas[t], abar[t]
            x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 0:
                x = x + betas[t].sqrt() * torch.randn_like(x)
            snaps[t] = x.cpu().numpy().copy()
    return T, snaps


# --------------------------------------------------------------- reverse
def fig_reverse():
    """An actual greedy solve by the trained denoiser: rendered trajectory with
    the chosen move and its probability at each step. Replay-verified."""
    from solve_policy import policy_greedy
    from solve import verify_solutions
    ck = torch.load(os.path.join(ROOT, "runs/3x3_diff/ckpt_latest.pt"),
                    map_location=DEV, weights_only=True)
    pnet = PolicyNet(env3.S, env3.M, ck["cfg"]["h1"], ck["cfg"]["h2"],
                     ck["cfg"]["blocks"]).to(DEV)
    pnet.load_state_dict(ck["net"]); pnet.eval()

    # find a scramble the greedy rollout solves in a moderate number of moves
    best = None
    for seed in range(400):
        gg = torch.Generator(device=DEV).manual_seed(seed)
        st_c = env3.scramble(1, 100, generator=gg)
        solved, lengths, actions = policy_greedy(env3, pnet, st_c, max_steps=60)
        if solved[0]:
            L_c = int(lengths[0])
            if best is None or abs(L_c - 21) < abs(best[1] - 21):
                best = (st_c, L_c, actions)
            if 16 <= L_c <= 26:
                break
    st, L, actions = best
    solved, _, _ = policy_greedy(env3, pnet, st, max_steps=60)
    assert solved[0] and verify_solutions(env3, st, actions)[0]
    moves = [int(a) for a in actions[0, :L].tolist()]

    # replay to collect states and probabilities
    states = [st[0].cpu().numpy()]
    confs = []
    s = st.clone()
    for mv in moves:
        with torch.no_grad():
            p = F.softmax(pnet(s)[0].float(), dim=0)[mv].item()
        confs.append(p)
        s = env3.step(s, torch.tensor([mv], device=DEV))
        states.append(s[0].cpu().numpy())

    show = sorted(set(int(round(x)) for x in np.linspace(0, L, 8)))
    ncol = len(show)

    # real spiral reconstruction by a tiny DDPM trained here (same lineage row)
    T2, snaps = train_spiral_ddpm()
    spiral_show = [int(round(f * T2)) for f in np.linspace(1.0, 0.0, ncol)]

    fig, axes = plt.subplots(2, ncol, figsize=(7.0, 2.9))
    for k, t2 in enumerate(spiral_show):
        ax = axes[0, k]
        pts = snaps[t2]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.7, c=BLUE, alpha=0.6, linewidths=0)
        ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("noise $x_T$" if k == 0 else
                     ("reconstructed" if t2 == 0 else f"$t/T={t2/T2:.2f}$"),
                     fontsize=7, pad=2)
    for k, idx in enumerate(show):
        ax = axes[1, k]
        render_cube(ax, states[idx])
        if idx == 0:
            title = "scrambled\n(100 moves)"
        elif idx == L:
            title = f"solved\n({L} moves)"
        else:
            title = f"step {idx}\n{env3.move_names[moves[idx]]} " \
                    f"({confs[idx]*100:.0f}%)"
        ax.set_title(title, fontsize=7, pad=2)
        if k < ncol - 1:
            ax.annotate("", xy=(1.30, 0.5), xytext=(1.04, 0.5),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=ORANGE))
    axes[0, 0].text(-0.34, 0.5, "spiral DDPM\n(reverse)", rotation=90,
                    transform=axes[0, 0].transAxes, va="center", ha="center",
                    fontsize=7.6, color=INK)
    axes[1, 0].text(-0.34, 0.5, "cube denoiser\n(reverse)", rotation=90,
                    transform=axes[1, 0].transAxes, va="center", ha="center",
                    fontsize=7.6, color=INK)
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    fig.savefig(os.path.join(HERE, "fig_reverse_process.pdf"), bbox_inches="tight")
    print(f"fig_reverse_process.pdf (solve length {L})")


# ---------------------------------------------------------------- teaser
def fig_teaser():
    """Page-1 strip: forward noising into a real reverse (solve) rollout."""
    from solve_policy import policy_greedy
    from solve import verify_solutions
    ck = torch.load(os.path.join(ROOT, "runs/3x3_diff/ckpt_latest.pt"),
                    map_location=DEV, weights_only=True)
    pnet = PolicyNet(env3.S, env3.M, ck["cfg"]["h1"], ck["cfg"]["h2"],
                     ck["cfg"]["blocks"]).to(DEV)
    pnet.load_state_dict(ck["net"]); pnet.eval()

    # forward: record a 30-move scramble trajectory; require greedy to solve it
    for seed in range(300):
        gg = torch.Generator(device=DEV).manual_seed(seed)
        s = env3.solved_batch(1)
        fwd_states = [s[0].cpu().numpy()]
        prev = -1
        for t in range(30):
            a = torch.randint(env3.M, (1,), device=DEV, generator=gg)
            while int(a) == (prev ^ 1):
                a = torch.randint(env3.M, (1,), device=DEV, generator=gg)
            s = env3.step(s, a); prev = int(a)
            fwd_states.append(s[0].cpu().numpy())
        solved, lengths, actions = policy_greedy(env3, pnet, s, max_steps=60)
        if solved[0] and int(lengths[0]) <= 30:
            break
    assert solved[0] and verify_solutions(env3, s, actions)[0]
    L = int(lengths[0])
    rev_states = [s[0].cpu().numpy()]
    ss = s.clone()
    for mv in actions[0, :L].tolist():
        ss = env3.step(ss, torch.tensor([int(mv)], device=DEV))
        rev_states.append(ss[0].cpu().numpy())

    fwd_idx = [0, 6, 14, 30]
    rev_idx = [int(round(x)) for x in np.linspace(0, L, 5)][1:]
    panels = [fwd_states[i] for i in fwd_idx] + [rev_states[i] for i in rev_idx]
    n_p = len(panels)

    fig, axes = plt.subplots(1, n_p, figsize=(7.0, 1.25))
    for k, stt in enumerate(panels):
        render_cube(axes[k], stt)
        if k < n_p - 1:
            col = MUTED if k < len(fwd_idx) - 1 else ORANGE
            axes[k].annotate("", xy=(1.32, 0.45), xytext=(1.02, 0.45),
                             xycoords="axes fraction",
                             arrowprops=dict(arrowstyle="-|>", lw=1.1, color=col))
    axes[0].set_title("solved $x_0$", fontsize=7.5, pad=2)
    axes[len(fwd_idx) - 1].set_title("fully noised", fontsize=7.5, pad=2)
    axes[-1].set_title(f"re-solved ({L} moves)", fontsize=7.5, pad=2)
    fig.text(0.245, 0.045, "forward process: noise = random moves",
             ha="center", fontsize=7.6, color=MUTED)
    fig.text(0.72, 0.045, "reverse process: learned denoiser $p_\\theta(a\\,|\\,s)$, no search",
             ha="center", fontsize=7.6, color=ORANGE)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(HERE, "fig_teaser.pdf"), bbox_inches="tight")
    print(f"fig_teaser.pdf (reverse length {L})")


if __name__ == "__main__":
    fig_forward()
    fig_architecture()
    fig_reverse()
    fig_teaser()
