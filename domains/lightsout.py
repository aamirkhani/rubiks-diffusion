"""Lights Out 5x5, vectorized.

Domain 8: COMMUTATIVE structure. Pressing cell i toggles i and its
orthogonal neighbors; presses commute and pressing twice cancels, so a
solution is an unordered SET of cells. The "undo the last move" label is
therefore maximally ambiguous: every cell in some minimal solution set is an
equally valid answer. Exact oracle via GF(2): solvability = orthogonality to
the two quiet patterns; optimal length = minimum weight over the 4-coset
solution family.

State: [B, 25] int8 in {0,1}. Actions: 25 (press cell). Goal: all off.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

N = 5
S = N * N


def _press_masks(device):
    masks = torch.zeros(S, S, dtype=torch.int8, device=device)
    for i in range(S):
        r, c = i // N, i % N
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N:
                masks[i, rr * N + cc] = 1
    return masks


class LightsOutEnv:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.S = S
        self.M = S
        self.vocab = 2
        self.masks = _press_masks(self.device)
        self.solved = torch.zeros(S, dtype=torch.int8, device=self.device)
        # every action is its own inverse
        self.inverse_action = torch.arange(S, device=self.device)

    def solved_batch(self, B):
        return self.solved.unsqueeze(0).repeat(B, 1)

    def is_solved(self, states):
        return (states == 0).all(dim=1)

    def legal_mask(self, states):
        return torch.ones(states.shape[0], self.M, dtype=torch.bool,
                          device=self.device)

    def step(self, states, actions):
        return (states ^ self.masks[actions]).to(torch.int8)

    def neighbors(self, states):
        return (states.unsqueeze(1) ^ self.masks.unsqueeze(0)).to(torch.int8)

    def scramble(self, B, depths, return_actions=False, generator=None):
        """Random DISTINCT presses (repeating a press cancels; sample without
        replacement so walk depth == solution set size <= 25)."""
        if isinstance(depths, int):
            depths = torch.full((B,), depths, device=self.device,
                                dtype=torch.long)
        depths = depths.clamp(1, S)
        # random permutation per row; take first `depth` presses
        perm = torch.rand(B, S, device=self.device,
                          generator=generator).argsort(dim=1)
        take = torch.arange(S, device=self.device).unsqueeze(0) < \
            depths.unsqueeze(1)
        states = self.solved_batch(B)
        hist = []
        maxd = int(depths.max())
        for t in range(maxd):
            a = perm[:, t]
            active = take[:, t]
            nxt = self.step(states, a)
            states = torch.where(active.unsqueeze(1), nxt, states)
            if return_actions:
                hist.append(torch.where(active, a, torch.full_like(a, -1)))
        if return_actions:
            return states, torch.stack(hist, 1)
        return states


# ------------------------------------------------------- exact GF(2) oracle
def _gf2_setup():
    A = _press_masks("cpu").numpy().astype(np.uint8).T  # A[j,i]: press i hits j
    # solve A x = b over GF(2); find particular solution + null space
    M_ = A.copy()
    n = S
    piv_cols, ops = [], []
    row = 0
    aug = np.eye(n, dtype=np.uint8)
    for col in range(n):
        sel = None
        for r in range(row, n):
            if M_[r, col]:
                sel = r
                break
        if sel is None:
            continue
        M_[[row, sel]] = M_[[sel, row]]
        aug[[row, sel]] = aug[[sel, row]]
        for r in range(n):
            if r != row and M_[r, col]:
                M_[r] ^= M_[row]
                aug[r] ^= aug[row]
        piv_cols.append(col)
        row += 1
    # null space of A^T? We need null space of A (x with A x = 0)
    free_cols = [c for c in range(n) if c not in piv_cols]
    null_vecs = []
    for fc in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[fc] = 1
        for r, pc in enumerate(piv_cols):
            if M_[r, fc]:
                v[pc] = 1
        null_vecs.append(v)
    return A, M_, aug, piv_cols, np.array(null_vecs, dtype=np.uint8)


_A, _R, _AUG, _PIV, _NULL = _gf2_setup()


def optimal_lengths(states_cpu):
    """Exact minimal press-set size per state (or -1 if unsolvable)."""
    b = states_cpu.numpy().astype(np.uint8)
    out = np.full(b.shape[0], -1, dtype=np.int32)
    # particular solution via recorded row ops: x_piv from reduced system
    rb = (b @ _AUG.T) % 2                     # reduced RHS
    for k in range(b.shape[0]):
        x = np.zeros(S, dtype=np.uint8)
        ok = True
        for r in range(len(_PIV), S):
            if rb[k, r]:
                ok = False
                break
        if not ok:
            continue
        for r, pc in enumerate(_PIV):
            x[pc] = rb[k, r]
        best = None
        n_null = _NULL.shape[0]
        for m in range(1 << n_null):
            xx = x.copy()
            for j in range(n_null):
                if m >> j & 1:
                    xx ^= _NULL[j]
            w = int(xx.sum())
            best = w if best is None else min(best, w)
        out[k] = best
    return torch.tensor(out)


def run_tests(device="cuda"):
    env = LightsOutEnv(device)
    B = 2048
    g = torch.Generator(device=device).manual_seed(0)
    depths = torch.randint(1, S + 1, (B,), device=device, generator=g)
    st, acts = env.scramble(B, depths, return_actions=True, generator=g)
    # press-set property: re-pressing the same set solves it
    s = st.clone()
    for t in range(acts.shape[1]):
        a = acts[:, t]
        valid = a >= 0
        nxt = env.step(s, a.clamp_min(0))
        s = torch.where(valid.unsqueeze(1), nxt, s)
    assert env.is_solved(s).all(), "re-pressing the scramble set failed"
    # oracle: distance <= scramble set size; solvable
    d = optimal_lengths(st[:256].cpu())
    assert (d >= 0).all()
    assert (d <= depths[:256].cpu()).all()
    # null space of the 5x5 board has dimension 2 (known fact)
    assert _NULL.shape[0] == 2, _NULL.shape
    print(f"lights-out tests passed: press-set inverse; GF(2) oracle "
          f"(null dim 2, known); mean optimal {d.float().mean():.1f}")


if __name__ == "__main__":
    run_tests()
