"""English peg solitaire (33-cell cross board), vectorized.

Domain 10: NON-CONSERVATIVE irreversible dynamics. Every jump removes a peg
(the state shrinks monotonically), deadlocks are everywhere, and forward
moves cannot be undone. Noising kernel: UN-JUMPS from the single-center-peg
goal (adds pegs), so every generated instance is solvable by construction.
Also the transfer test for the Sokoban hybrid fix.

Board: 7x7 grid, [B, 49] int8: 0 empty, 1 peg, 2 invalid (corner) cell.
Actions: origin cell x direction = 49*4 = 196; jump p -> p+2d over p+d.
Goal: exactly one peg, at the center (cell 24).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

N = 7
S = N * N
CENTER = 24
INVALID = 2


def _valid_mask(device):
    v = torch.ones(S, dtype=torch.bool, device=device)
    for r in range(N):
        for c in range(N):
            if (r < 2 or r > 4) and (c < 2 or c > 4):
                v[r * N + c] = False
    return v


class PegEnv:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.S = S
        self.M = S * 4
        self.vocab = 3
        self.valid = _valid_mask(self.device)
        self.dr = torch.tensor([-1, 1, 0, 0], device=self.device)
        self.dc = torch.tensor([0, 0, -1, 1], device=self.device)
        # precompute over/land cells for each (cell, dir); -1 = off board
        over = torch.full((S, 4), -1, dtype=torch.long, device=self.device)
        land = torch.full((S, 4), -1, dtype=torch.long, device=self.device)
        for p in range(S):
            if not self.valid[p]:
                continue
            r, c = p // N, p % N
            for d in range(4):
                r1, c1 = r + int(self.dr[d]), c + int(self.dc[d])
                r2, c2 = r + 2 * int(self.dr[d]), c + 2 * int(self.dc[d])
                if 0 <= r2 < N and 0 <= c2 < N:
                    o, l = r1 * N + c1, r2 * N + c2
                    if self.valid[o] and self.valid[l]:
                        over[p, d], land[p, d] = o, l
        self.over, self.land = over, land
        g = torch.zeros(S, dtype=torch.int8, device=self.device)
        g[~self.valid] = INVALID
        g[CENTER] = 1
        self.goal_state = g

    def solved_batch(self, B):
        return self.goal_state.unsqueeze(0).repeat(B, 1)

    def is_solved(self, states):
        return ((states == 1).sum(1) == 1) & (states[:, CENTER] == 1)

    def legal_mask(self, states):
        """[B, 196] forward jumps."""
        B = states.shape[0]
        peg = states == 1
        emp = states == 0
        over = self.over.view(-1)                       # [S*4]
        land = self.land.view(-1)
        ok_geom = (over >= 0)
        ov = torch.where(ok_geom, over, torch.zeros_like(over))
        ld = torch.where(ok_geom, land, torch.zeros_like(land))
        src = peg.repeat_interleave(4, dim=1)           # [B, S*4] peg at origin
        m = src & peg[:, ov] & emp[:, ld] & ok_geom.unsqueeze(0)
        return m

    def step(self, states, actions):
        B = states.shape[0]
        rows = torch.arange(B, device=self.device)
        p = actions // 4
        d = actions % 4
        o = self.over[p, d]
        l = self.land[p, d]
        legal = (o >= 0)
        legal = legal & (states[rows, p] == 1)
        osafe = o.clamp_min(0)
        lsafe = l.clamp_min(0)
        legal = legal & (states[rows, osafe] == 1) & (states[rows, lsafe] == 0)
        out = states.clone()
        lr = rows[legal]
        out[lr, p[legal]] = 0
        out[lr, osafe[legal]] = 0
        out[lr, lsafe[legal]] = 1
        return out

    def neighbors(self, states):
        """Too many actions to materialize all; used only by DAVI via chunks."""
        raise NotImplementedError("use masked value lookahead instead")

    def scramble(self, B, depths, return_actions=False, generator=None):
        """Reverse construction: un-jumps from the goal. Returns states and
        the LAST forward action label (the jump that undoes the last un-jump)."""
        if isinstance(depths, int):
            depths = torch.full((B,), depths, device=self.device,
                                dtype=torch.long)
        states = self.solved_batch(B)
        rows = torch.arange(B, device=self.device)
        last = torch.full((B,), -1, dtype=torch.long, device=self.device)
        maxd = int(depths.max())
        for t in range(maxd):
            # un-jump legal: land cell has peg, origin and over empty
            peg = states == 1
            emp = states == 0
            over = self.over.view(-1)
            land = self.land.view(-1)
            ok = over >= 0
            ov = torch.where(ok, over, torch.zeros_like(over))
            ld = torch.where(ok, land, torch.zeros_like(land))
            m = emp.repeat_interleave(4, 1) & emp[:, ov] & peg[:, ld] & \
                ok.unsqueeze(0)
            u = torch.rand(B, self.M, device=self.device,
                           generator=generator).clamp_min(1e-9)
            a = (-torch.log(-torch.log(u))).masked_fill(~m, -1e9).argmax(1)
            any_ok = m.any(1)
            active = (t < depths) & any_ok
            p = a // 4
            d = a % 4
            o = self.over[p, d].clamp_min(0)
            l = self.land[p, d].clamp_min(0)
            ar = rows[active]
            states[ar, p[active]] = 1
            states[ar, o[active]] = 1
            states[ar, l[active]] = 0
            last = torch.where(active, a, last)
        if return_actions:
            return states, last
        return states


def run_tests(device="cuda"):
    env = PegEnv(device)
    B = 2048
    g = torch.Generator(device=device).manual_seed(0)
    # full-history generation + forward replay must re-solve
    states = env.solved_batch(B)
    rows = torch.arange(B, device=device)
    hist = []
    for t in range(20):
        peg = states == 1
        emp = states == 0
        over = env.over.view(-1)
        land = env.land.view(-1)
        ok = over >= 0
        ov = torch.where(ok, over, torch.zeros_like(over))
        ld = torch.where(ok, land, torch.zeros_like(land))
        m = emp.repeat_interleave(4, 1) & emp[:, ov] & peg[:, ld] & ok.unsqueeze(0)
        u = torch.rand(B, env.M, device=device, generator=g).clamp_min(1e-9)
        a = (-torch.log(-torch.log(u))).masked_fill(~m, -1e9).argmax(1)
        okr = m.any(1)
        p, d = a // 4, a % 4
        o = env.over[p, d].clamp_min(0)
        l = env.land[p, d].clamp_min(0)
        ar = rows[okr]
        states[ar, p[okr]] = 1
        states[ar, o[okr]] = 1
        states[ar, l[okr]] = 0
        hist.append(torch.where(okr, a, torch.full_like(a, -1)))
    for t in range(19, -1, -1):
        a = hist[t]
        valid = a >= 0
        nxt = env.step(states, a.clamp_min(0))
        states = torch.where(valid.unsqueeze(1), nxt, states)
    assert env.is_solved(states).all(), "peg replay failed"
    # peg count changes by exactly +1 per un-jump / -1 per jump
    st2, lab = env.scramble(512, 15, return_actions=True, generator=g)
    counts = (st2 == 1).sum(1)
    assert (counts <= 16).all() and (counts >= 1).all()
    legal = env.legal_mask(st2)
    r5 = torch.arange(512, device=device)
    has = lab >= 0
    assert legal[r5[has], lab[has]].all(), "last-label not a legal forward jump"
    print(f"peg solitaire tests passed: {B} reverse walks fully re-solved by "
          f"forward jumps; labels always legal")


if __name__ == "__main__":
    run_tests()
