"""Vectorized Rubik's cube environments (2x2 & 3x3) — pure torch, GPU-friendly.

State: [B, S] int8 sticker colors (0..5), S = 6*n*n.
Face order U, R, F, D, L, B (Kociemba layout), each face row-major as seen
from outside the cube.

Every face turn is a fixed permutation of the S sticker indices. The move
tables are generated geometrically (rotate sticker positions/normals with a
rotation matrix and re-match them), so there are no hand-typed cycles to get
wrong. Applying a batch of moves is a single torch.gather:

    new_state = state.gather(1, MOVES[actions])

The 2x2 env restricts moves to the U, R, F faces. That pins the DLB corner in
place, which makes the solved state unique (no whole-cube rotations) and the
reachable state space exactly 7! * 3^6 = 3,674,160 states — small enough for
an exact BFS distance oracle.
"""
import numpy as np
import torch

FACE_NAMES = ["U", "R", "F", "D", "L", "B"]
FACE_AXES = {
    "U": (0, 1, 0),
    "R": (1, 0, 0),
    "F": (0, 0, 1),
    "D": (0, -1, 0),
    "L": (-1, 0, 0),
    "B": (0, 0, -1),
}


def _sticker_geometry(n):
    """Center position and outward normal of every sticker, face-major order."""
    pos, nrm = [], []
    for f in FACE_NAMES:
        for i in range(n):          # row (top->bottom as seen from outside)
            for j in range(n):      # col (left->right as seen from outside)
                if f == "U":
                    p, m = (j + .5, n, i + .5), (0, 1, 0)
                elif f == "R":
                    p, m = (n, n - (i + .5), n - (j + .5)), (1, 0, 0)
                elif f == "F":
                    p, m = (j + .5, n - (i + .5), n), (0, 0, 1)
                elif f == "D":
                    p, m = (j + .5, 0, n - (i + .5)), (0, -1, 0)
                elif f == "L":
                    p, m = (0, n - (i + .5), j + .5), (-1, 0, 0)
                else:  # B
                    p, m = (n - (j + .5), n - (i + .5), 0), (0, 0, -1)
                pos.append(p)
                nrm.append(m)
    return np.asarray(pos, dtype=np.float64), np.asarray(nrm, dtype=np.float64)


def _rot(axis, angle_deg):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    t = np.deg2rad(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * np.cos(t) + np.sin(t) * K + (1 - np.cos(t)) * np.outer(a, a)


def face_cw_perm(n, face):
    """Permutation for a clockwise quarter turn of `face` (viewed from outside).

    Returned as gather indices: new_state[i] = old_state[perm[i]].
    """
    pos, nrm = _sticker_geometry(n)
    axis = np.asarray(FACE_AXES[face], dtype=np.float64)
    center = np.full(3, n / 2.0)
    # Clockwise viewed from outside (looking back along the outward axis)
    # is a -90 degree right-hand rotation about that axis.
    R = _rot(axis, -90.0)
    d = (pos - center) @ axis
    in_layer = d > (n / 2.0 - 1.0 + 1e-9)

    new_pos, new_nrm = pos.copy(), nrm.copy()
    new_pos[in_layer] = (R @ (pos[in_layer] - center).T).T + center
    new_nrm[in_layer] = (R @ nrm[in_layer].T).T

    S = len(pos)
    perm = -np.ones(S, dtype=np.int64)
    for s in range(S):
        err = ((pos - new_pos[s]) ** 2).sum(1) + ((nrm - new_nrm[s]) ** 2).sum(1)
        t = int(err.argmin())
        assert err[t] < 1e-9, f"no target sticker for {face} src={s}"
        assert perm[t] == -1, f"double assignment in {face} perm"
        perm[t] = s
    return perm


class CubeEnv:
    def __init__(self, n, device="cuda", faces=None):
        if faces is None:
            faces = ["U", "R", "F"] if n == 2 else list(FACE_NAMES)
        self.n = n
        self.S = 6 * n * n
        self.device = torch.device(device)
        perms, names = [], []
        for f in faces:
            cw = face_cw_perm(n, f)
            perms.append(cw)
            names.append(f)
            perms.append(np.argsort(cw))  # inverse permutation = CCW turn
            names.append(f + "'")
        self.moves = torch.tensor(np.stack(perms), device=self.device)  # [M, S]
        self.move_names = names
        self.M = len(names)
        # inverse action index: CW<->CCW pairs are adjacent
        inv = torch.arange(self.M, device=self.device)
        self.inverse_action = inv ^ 1
        self.solved = (
            torch.arange(6, device=self.device)
            .repeat_interleave(n * n)
            .to(torch.int8)
        )

    def solved_batch(self, batch):
        return self.solved.unsqueeze(0).repeat(batch, 1)

    def step(self, states, actions):
        """states [B,S] int8, actions [B] long -> [B,S] int8."""
        return torch.gather(states, 1, self.moves[actions])

    def neighbors(self, states):
        """All single-move successors: [B, S] -> [B, M, S]."""
        B = states.shape[0]
        idx = self.moves.unsqueeze(0).expand(B, self.M, self.S)
        return torch.gather(states.unsqueeze(1).expand(B, self.M, self.S), 2, idx)

    def is_solved(self, states):
        return (states == self.solved).all(dim=1)

    def scramble(self, batch, depths, return_actions=False, generator=None):
        """Random walk from solved. depths: int or [B] long tensor.

        Avoids undoing the immediately preceding move so the walk depth is a
        meaningful (upper bound on) scramble distance.
        """
        if isinstance(depths, int):
            depths = torch.full((batch,), depths, device=self.device, dtype=torch.long)
        states = self.solved_batch(batch)
        prev = torch.full((batch,), -1, device=self.device, dtype=torch.long)
        maxd = int(depths.max().item())
        actions_hist = []
        for t in range(maxd):
            a = torch.randint(self.M, (batch,), device=self.device, generator=generator)
            # resample where a undoes the previous move
            for _ in range(8):
                bad = a == (prev ^ 1)
                if not bad.any():
                    break
                a = torch.where(
                    bad,
                    torch.randint(self.M, (batch,), device=self.device, generator=generator),
                    a,
                )
            active = t < depths
            nxt = self.step(states, a)
            states = torch.where(active.unsqueeze(1), nxt, states)
            prev = torch.where(active, a, prev)
            if return_actions:
                actions_hist.append(torch.where(active, a, torch.full_like(a, -1)))
        if return_actions:
            return states, torch.stack(actions_hist, 1) if actions_hist else torch.empty(batch, 0, dtype=torch.long, device=self.device)
        return states

    def action_index(self, name):
        return self.move_names.index(name)

    def apply_names(self, state, names):
        """Apply a move sequence (list of names) to a single state [S]."""
        s = state.unsqueeze(0)
        for nm in names:
            a = torch.tensor([self.action_index(nm)], device=self.device)
            s = self.step(s, a)
        return s.squeeze(0)

    def render_ansi(self, state):
        """Flat unfolded-cross text rendering of a single state [S]."""
        n = self.n
        cols = ["W", "R", "G", "Y", "O", "B"]  # U R F D L B
        s = state.cpu().numpy()
        face = lambda f: s[f * n * n:(f + 1) * n * n].reshape(n, n)
        U, R, F, D, L, B = (face(k) for k in range(6))
        pad = " " * (2 * n + 1)
        out = []
        for r in U:
            out.append(pad + " ".join(cols[c] for c in r))
        for i in range(n):
            row = []
            for M in (L, F, R, B):
                row.append(" ".join(cols[c] for c in M[i]))
            out.append("  ".join(row))
        for r in D:
            out.append(pad + " ".join(cols[c] for c in r))
        return "\n".join(out)


def pack_2x2(states):
    """[B,24] int8 -> [B] int64 base-6 key (6^24 < 2^63)."""
    powers = torch.tensor([6 ** i for i in range(24)], dtype=torch.int64, device=states.device)
    return (states.to(torch.int64) * powers).sum(1)


def unpack_2x2(keys):
    """[B] int64 -> [B,24] int8."""
    k = keys.clone()
    out = torch.empty(k.shape[0], 24, dtype=torch.int8, device=keys.device)
    for i in range(24):
        out[:, i] = (k % 6).to(torch.int8)
        k //= 6
    return out
