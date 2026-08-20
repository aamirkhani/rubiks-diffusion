"""Standalone reference figure: the three classic diffusion visuals on ONE
t = 0 -> 1 grid --- spiral particle cloud (Sohl-Dickstein 2015), a photograph
under Gaussian noise (DDPM), and the Rubik's Cube noising schedule from the
companion paper (real engine states, rendered in 3D).

Output: fig_ref_cube.pdf / .png (kept separate from galleries A/B).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "video"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from make_video import draw_cube, spiral_pts, astro, env3

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLUE = "#2a78d6"
NCOL = 8
MAXDEPTH = 30
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})


def cube_walk_states():
    """One seeded noising walk; snapshot at depth round(t * MAXDEPTH)."""
    g = torch.Generator(device=DEV).manual_seed(11)
    states, s, prev = [env3.solved.cpu().numpy()], env3.solved_batch(1), -1
    for _ in range(MAXDEPTH):
        a = int(torch.randint(env3.M, (1,), device=DEV, generator=g))
        while a == (prev ^ 1):
            a = int(torch.randint(env3.M, (1,), device=DEV, generator=g))
        s = env3.step(s, torch.tensor([a], device=DEV))
        states.append(s[0].cpu().numpy())
        prev = a
    return states


def main():
    rng = np.random.default_rng(3)
    x0 = spiral_pts()
    eps = rng.standard_normal(x0.shape)
    img0 = astro()
    epsI = np.random.default_rng(0).standard_normal(img0.shape)
    cubes = cube_walk_states()

    fig, axes = plt.subplots(3, NCOL, figsize=(7.0, 3.1))
    for c in range(NCOL):
        t = c / (NCOL - 1)
        # row 0: spiral particles (quadratic growth, as in the galleries)
        ax = axes[0, c]
        sig = 0.55 * t ** 2
        pts = (1 - 0.55 * t ** 2) * x0 + sig * eps
        ax.scatter(pts[:, 0], pts[:, 1], s=0.9, c=BLUE, alpha=0.7, linewidths=0)
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"$t={t:.2f}$", fontsize=7)
        # row 1: photograph under Gaussian noise
        ax = axes[1, c]
        sI = 1.0 * t ** 2
        ax.imshow(np.clip(np.sqrt(max(1 - sI ** 2, 1e-4)) * img0 + sI * epsI,
                          0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        # row 2: the cube, scramble depth = round(t * MAXDEPTH)
        ax = axes[2, c]
        draw_cube(ax, cubes[int(round(t * MAXDEPTH))], lw=0.4)
    labels = ["particle diffusion\n(Sohl-Dickstein '15)",
              "image diffusion\n(DDPM)",
              "Rubik's Cube scramble\n(companion paper)"]
    for r, lab in enumerate(labels):
        axes[r, 0].text(-0.30, 0.5, lab, rotation=90, fontsize=6.4,
                        color="#4a4742", transform=axes[r, 0].transAxes,
                        va="center", ha="center")
    fig.tight_layout(w_pad=0.25, h_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_ref_cube.{ext}"),
                    bbox_inches="tight")
    print("fig_ref_cube.pdf/.png")


if __name__ == "__main__":
    main()
