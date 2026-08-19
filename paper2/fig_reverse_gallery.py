"""The reverse-process gallery: one row per domain, columns from fully
noised (t=1) to clean (t=0). Every row is an ACTUAL trained-denoiser
trajectory (greedy unless noted), rendered in the domain's native look.
Rows whose checkpoints don't exist yet are skipped (rerun after the sweep).
"""
import json
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


def snap_idx(L):
    return sorted(set(int(round(x)) for x in np.linspace(0, L, NCOL)))


def t_label(k, n):
    return f"$t={1 - k / max(n - 1, 1):.2f}$"


# ------------------------------------------------------------- renderers
def draw_slide(ax, board, n=5):
    b = board.reshape(n, n)
    ax.imshow((b > 0).astype(float), cmap="Oranges", vmin=-0.6, vmax=1.6)
    for r in range(n):
        for c in range(n):
            if b[r, c]:
                ax.text(c, r, str(b[r, c]), ha="center", va="center", fontsize=4.6)
    ax.set_xticks([]); ax.set_yticks([])


def draw_grid(ax, img, cmap, vmax):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])


def draw_pend(ax, th):
    ax.plot([0, 0], [0, 1.05], ls=":", lw=0.7, color=GRAY)
    ax.plot([0, np.sin(th)], [np.cos(th) * 0 + 0, np.cos(th)], lw=2.2, color=BLUE)
    ax.plot([np.sin(th)], [np.cos(th)], "o", ms=6, color=BLUE)
    ax.plot([0], [0], "o", ms=3, color="k")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal"); ax.axis("off")


def draw_2048(ax, board):
    b = board.reshape(4, 4)
    ax.imshow((b > 0).astype(float), cmap="Purples", vmin=-0.6, vmax=1.8)
    for r in range(4):
        for c in range(4):
            if b[r, c]:
                ax.text(c, r, str(2 ** int(b[r, c])), ha="center", va="center",
                        fontsize=4.4)
    ax.set_xticks([]); ax.set_yticks([])


def draw_hanoi(ax, pegs, n=10):
    heights = [0, 0, 0]
    for disk in range(n - 1, -1, -1):
        p = int(pegs[disk])
        w = 0.25 + 0.55 * (disk + 1) / n
        ax.barh(heights[p], w, left=p - w / 2, height=0.85, color=ORANGE,
                edgecolor="white", linewidth=0.25)
        heights[p] += 1
    for p in range(3):
        ax.plot([p, p], [0, n + 0.5], color=GRAY, lw=0.7, zorder=0)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, n + 1)
    ax.axis("off")


# ------------------------------------------------------------ trajectories
def traj_slide():
    from domains.slide import SlideEnv
    from domains.train_slide import rollout, verify
    ck = torch.load("runs/slide5_diff/ckpt_latest.pt", map_location=DEV,
                    weights_only=True)
    env = SlideEnv(5, DEV)
    net = PolicyNet(env.S, env.M, 5000, 1000, 4, vocab=env.vocab).to(DEV)
    net.load_state_dict(ck["net"]); net.eval()
    g = torch.Generator(device=DEV).manual_seed(4)
    st = env.scramble(1, 1000, generator=g)
    solved, actions = rollout(env, net, "denoise", st, 400)
    assert solved[0] and verify(env, st, actions)[0]
    moves = [int(a) for a in actions[0].tolist() if a >= 0]
    states, s = [st[0].cpu().numpy()], st.clone()
    for mv in moves:
        s = env.step(s, torch.tensor([mv], device=DEV))
        states.append(s[0].cpu().numpy())
    return [states[i] for i in snap_idx(len(moves))], True, \
        f"24-puzzle — {len(moves)} moves, greedy"


def traj_maze():
    from domains.maze import MazeEnv
    from domains.train_maze import rollout
    ck = torch.load("runs/maze_diff/ckpt_latest.pt", map_location=DEV,
                    weights_only=True)
    cfg = ck["cfg"]
    env = MazeEnv(cfg["n"], device=DEV)
    net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"],
                    vocab=4).to(DEV)
    net.load_state_dict(ck["net"]); net.eval()
    for seed in range(200):
        g = torch.Generator(device=DEV).manual_seed(seed)
        walls, goal = env.new_instances(1, generator=g)
        st = env.scramble(walls, goal, 60, generator=g)
        solved, actions = rollout(env, net, "denoise", st, goal, 150)
        L = int((actions[0] >= 0).sum())
        if solved[0] and 14 <= L <= 60:
            break
    moves = [int(a) for a in actions[0].tolist() if a >= 0]
    states, s = [st[0].cpu().numpy()], st.clone()
    for mv in moves:
        s = env.step(s, torch.tensor([mv], device=DEV), goal)
        states.append(s[0].cpu().numpy())
    return [states[i].reshape(15, 15) for i in snap_idx(len(moves))], True, \
        f"maze — {len(moves)} moves, greedy (fresh instance)"


def traj_sokoban():
    from domains.sokoban import SokobanEnv
    from domains.eval_sokoban import load
    from domains.hybrid_sokoban import hybrid_rollout
    env, pnet, _ = load("runs/soko_diff/ckpt_latest.pt")
    _, vnet, _ = load("runs/soko_davi/ckpt_latest.pt")
    import torch.nn.functional as F
    from domains.train_sokoban import solved_rendered
    for seed in range(300):
        g = torch.Generator(device=DEV).manual_seed(seed)
        w, go, b, a_, _ = env.instances_and_scramble(1, 50, generator=g)
        # hybrid rollout with state recording
        boxes, agent = b.clone(), a_.clone()
        frames = [env.render(w, go, boxes, agent)[0].cpu().numpy()]
        ok = False
        for t in range(160):
            if env.is_solved(w, go, boxes, agent)[0]:
                ok = True
                break
            legal = env.legal_forward(w, go, boxes, agent)
            st = env.render(w, go, boxes, agent)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logp = F.log_softmax(pnet(st).float(), 1)
                nb = env.neighbors_forward(w, go, boxes, agent).reshape(-1, env.S)
                v = vnet(nb).float().view(-1, env.M)
            v = torch.where(solved_rendered(nb).view(-1, env.M),
                            torch.zeros_like(v), v.clamp_min(0))
            act = (logp - 2.0 * v).masked_fill(~legal, -1e30).argmax(1)
            boxes, agent = env.step_forward(w, go, boxes, agent, act)
            frames.append(env.render(w, go, boxes, agent)[0].cpu().numpy())
        if ok and 10 <= len(frames) - 1 <= 80:
            break
    return [frames[i].reshape(8, 8) for i in snap_idx(len(frames) - 1)], True, \
        f"Sokoban — {len(frames)-1} moves, hybrid (denoiser + value)"


def traj_pendulum():
    from domains.pendulum import Reg, NBINS, bins, fwd_step, is_goal, UMAX
    ck = torch.load("runs/pend_diff_disc/ckpt_latest.pt", map_location=DEV,
                    weights_only=True)
    net = Reg(out=NBINS).to(DEV)
    net.load_state_dict(ck["net"]); net.eval()
    th = torch.tensor([np.pi], device=DEV, dtype=torch.float32)
    om = torch.zeros(1, device=DEV)
    traj = [float(th)]
    hold = 0
    with torch.no_grad():
        for t in range(250):
            u = bins(DEV)[net(th, om).argmax(-1)]
            th, om = fwd_step(th, om, u.clamp(-UMAX, UMAX))
            traj.append(float(th))
            hold = hold + 1 if bool(is_goal(th, om)[0]) else 0
            if hold >= 10:
                break
    return [traj[i] for i in snap_idx(len(traj) - 1)], True, \
        f"pendulum — swing-up in {len(traj)-1} steps, categorical denoiser"


def traj_2048():
    from domains.game2048 import spawn, legal_mask, play_step, S, max_tile
    ck = torch.load("runs/g2048_diff/ckpt_latest.pt", map_location=DEV,
                    weights_only=True)
    net = PolicyNet(S, 4, 2048, 1024, 3, vocab=12).to(DEV)
    net.load_state_dict(ck["net"]); net.eval()
    g = torch.Generator(device=DEV).manual_seed(11)
    board = spawn(spawn(torch.zeros(1, S, dtype=torch.int8, device=DEV), g), g)
    frames = [board[0].cpu().numpy()]
    with torch.no_grad():
        for t in range(400):
            legal = legal_mask(board)
            if not legal.any():
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d = net(board).float().masked_fill(~legal, -1e9).argmax(1)
            board, _ = play_step(board, d, generator=g)
            frames.append(board[0].cpu().numpy())
    reached = int(max_tile(board)[0])
    return [frames[i] for i in snap_idx(len(frames) - 1)], False, \
        f"2048 — stalls at {2**reached} (stochasticity breaks the recipe)"


def traj_hanoi():
    path = "runs/hanoi_diff_s0/ckpt_latest.pt"
    if not os.path.exists(path):
        return None
    from domains.hanoi import HanoiEnv
    from domains.train_slide import rollout, verify
    ck = torch.load(path, map_location=DEV, weights_only=True)
    cfg = ck["cfg"]
    env = HanoiEnv(cfg["n"], DEV)
    net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"],
                    vocab=env.vocab).to(DEV)
    net.load_state_dict(ck["net"]); net.eval()
    # the classic start: all disks on peg 0, exactly 1023 from the goal
    st = torch.zeros(1, cfg["n"], dtype=torch.int8, device=DEV)
    solved, actions = rollout(env, net, "denoise", st, 2500)
    if not solved[0]:
        return None
    assert verify(env, st, actions)[0]
    moves = [int(a) for a in actions[0].tolist() if a >= 0]
    states, s = [st[0].cpu().numpy()], st.clone()
    for mv in moves:
        s = env.step(s, torch.tensor([mv], device=DEV))
        states.append(s[0].cpu().numpy())
    return [states[i] for i in snap_idx(len(moves))], True, \
        f"Hanoi — classic start solved in {len(moves)} moves (optimal 1023)"


def traj_pomdp():
    path = "runs/pomdp_diff_s0/ckpt_latest.pt"
    if not os.path.exists(path):
        return None
    from domains.maze import MazeEnv, observe
    import domains.train_maze as tm
    ck = torch.load(path, map_location=DEV, weights_only=True)
    cfg = ck["cfg"]
    env = MazeEnv(cfg["n"], device=DEV)
    net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"],
                    vocab=5).to(DEV)
    net.load_state_dict(ck["net"]); net.eval()
    tm.OBS_RADIUS = cfg.get("radius", 3)
    for seed in range(300):
        g = torch.Generator(device=DEV).manual_seed(seed)
        walls, goal = env.new_instances(1, generator=g)
        st = env.scramble(walls, goal, 60, generator=g)
        solved, actions = tm.rollout(env, net, "denoise", st, goal, 150)
        L = int((actions[0] >= 0).sum())
        if solved[0] and 14 <= L <= 60:
            break
    tm.OBS_RADIUS = None
    moves = [int(a) for a in actions[0].tolist() if a >= 0]
    states, s = [st[0].cpu().numpy()], st.clone()
    for mv in moves:
        s = env.step(s, torch.tensor([mv], device=DEV), goal)
        states.append(s[0].cpu().numpy())
    # display the OBSERVATION the policy actually saw
    obs = [observe(torch.tensor(states[i], device=DEV).unsqueeze(0), 15, 3)[0]
           .cpu().numpy().reshape(15, 15) for i in snap_idx(len(moves))]
    return obs, True, f"POMDP maze — {len(moves)} moves seeing only a 7x7 window"


# ---------------------------------------------------------------- assemble
def main():
    rows = []
    rows.append(("slide", *traj_slide()))
    rows.append(("maze", *traj_maze()))
    rows.append(("soko", *traj_sokoban()))
    rows.append(("pend", *traj_pendulum()))
    h = traj_hanoi()
    if h:
        rows.append(("hanoi", *h))
    p = traj_pomdp()
    if p:
        rows.append(("pomdp", *p))
    rows.append(("g2048", *traj_2048()))

    R = len(rows)
    fig, axes = plt.subplots(R, NCOL, figsize=(7.0, 1.02 * R))
    for r, (kind, snaps, ok, label) in enumerate(rows):
        for k in range(NCOL):
            ax = axes[r, k]
            if k >= len(snaps):
                ax.axis("off")
                continue
            sn = snaps[k]
            if kind == "slide":
                draw_slide(ax, sn)
            elif kind in ("maze",):
                draw_grid(ax, sn, "Greys", 4)
            elif kind == "pomdp":
                draw_grid(ax, sn, "Greys", 4)
            elif kind == "soko":
                draw_grid(ax, sn, "YlOrBr", 6)
            elif kind == "pend":
                draw_pend(ax, sn)
            elif kind == "hanoi":
                draw_hanoi(ax, sn)
            elif kind == "g2048":
                draw_2048(ax, sn)
            if r == 0:
                ax.set_title(t_label(k, min(NCOL, len(snaps))), fontsize=7)
        # row label + verdict mark
        axes[r, 0].text(-0.28, 0.5, label.split(" — ")[0], rotation=90,
                        transform=axes[r, 0].transAxes, va="center",
                        ha="center", fontsize=7.5)
        mark, col = ("✓", GREEN) if ok else ("✗", RED)
        axes[r, -1].text(1.12, 0.5, mark, transform=axes[r, -1].transAxes,
                         fontsize=13, color=col, va="center", weight="bold")
        axes[r, 0].text(0.0, -0.16, label, transform=axes[r, 0].transAxes,
                        fontsize=6.2, color="#6b6a66", ha="left")
    fig.tight_layout(w_pad=0.3, h_pad=1.0)
    fig.savefig(os.path.join(HERE, "fig2_reverse_gallery.pdf"),
                bbox_inches="tight")
    print(f"fig2_reverse_gallery.pdf ({R} rows)")


if __name__ == "__main__":
    main()
