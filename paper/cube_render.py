"""Matplotlib isometric cube renderer, reusing the engine's validated sticker
geometry so rendered states are exactly the tensor states."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matplotlib.collections import PolyCollection

from cube_env import _sticker_geometry

FACE_HEX = ["#f6f6f4", "#c41e3a", "#009b48", "#ffd500", "#ff5800", "#0046ad"]
EDGE = "#14130f"


def render_cube(ax, state, n=3, yaw=-0.62, pitch=-0.42, lw=0.9):
    """Draw sticker state (numpy int array [6*n*n]) on ax; returns nothing."""
    pos, nrm = _sticker_geometry(n)
    pos = pos - n / 2.0
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    def view(p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        x1 = x * cy + z * sy
        z1 = -x * sy + z * cy
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp
        return np.stack([x1, y2, z2], axis=-1)

    half = 0.44
    quads, colors, depths = [], [], []
    for i in range(6 * n * n):
        m = nrm[i]
        up = np.array([0, 0, 1.0]) if abs(m[1]) > 0.9 else np.array([0, 1.0, 0])
        u = np.cross(m, up); u /= np.linalg.norm(u)
        v = np.cross(m, u)
        c = pos[i]
        quad3 = np.stack([c + (u + v) * half, c + (u - v) * half,
                          c - (u + v) * half, c - (u - v) * half])
        q = view(quad3)
        quads.append(q[:, :2])
        depths.append(q[:, 2].mean())
        colors.append(FACE_HEX[int(state[i])])
    order = np.argsort(depths)
    pc = PolyCollection([quads[k] for k in order],
                        facecolors=[colors[k] for k in order],
                        edgecolors=EDGE, linewidths=lw, joinstyle="round")
    ax.add_collection(pc)
    lim = n * 0.95
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off")


def cube_image(state, n=3, px=240):
    """Render a state to an RGB float array in [0,1] (for the noising demo)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2), dpi=px // 2)
    fig.patch.set_facecolor("white")
    render_cube(ax, state, n, lw=1.4)
    fig.tight_layout(pad=0.05)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    plt.close(fig)
    return buf
