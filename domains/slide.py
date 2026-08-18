"""Vectorized sliding-tile puzzle (8-puzzle / 15-puzzle / 24-puzzle).

Domain 1 of the beyond-groups study: removes the *group structure* assumption —
the legal action set depends on the state (where the blank is), so moves are
no longer a free group action, but remain deterministic and invertible.

State: [B, S] int8, value at each cell in row-major order, 0 = blank.
Goal:  [1, 2, ..., S-1, 0].
Actions (blank movement): 0=up, 1=down, 2=left, 3=right; inverse = a ^ 1 after
pairing (up,down), (left,right). Illegal actions are no-ops (masked in
training/search via legal_mask).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class SlideEnv:
    def __init__(self, n, device="cuda"):
        self.n = n
        self.S = n * n
        self.M = 4
        self.vocab = self.S
        self.device = torch.device(device)
        g = torch.arange(1, self.S + 1, device=self.device, dtype=torch.int8)
        g[-1] = 0
        self.solved = g
        self.inverse_action = torch.tensor([1, 0, 3, 2], device=self.device)
        # action deltas on the blank's (row, col): up, down, left, right
        self.drow = torch.tensor([-1, 1, 0, 0], device=self.device)
        self.dcol = torch.tensor([0, 0, -1, 1], device=self.device)

    def solved_batch(self, B):
        return self.solved.unsqueeze(0).repeat(B, 1)

    def blank_pos(self, states):
        return (states == 0).float().argmax(dim=1)

    def legal_mask(self, states):
        """[B, 4] bool — which blank moves stay on the board."""
        bp = self.blank_pos(states)
        r, c = bp // self.n, bp % self.n
        nr = r.unsqueeze(1) + self.drow.unsqueeze(0)
        nc = c.unsqueeze(1) + self.dcol.unsqueeze(0)
        return (nr >= 0) & (nr < self.n) & (nc >= 0) & (nc < self.n)

    def step(self, states, actions):
        """Swap blank with the neighbor in the action direction; illegal = no-op."""
        B = states.shape[0]
        bp = self.blank_pos(states).long()
        r, c = bp // self.n, bp % self.n
        nr, nc = r + self.drow[actions], c + self.dcol[actions]
        legal = (nr >= 0) & (nr < self.n) & (nc >= 0) & (nc < self.n)
        tp = (nr.clamp(0, self.n - 1) * self.n + nc.clamp(0, self.n - 1)).long()
        out = states.clone()
        rows = torch.arange(B, device=self.device)
        moved = out[rows, tp]
        out[rows, tp] = 0
        out[rows, bp] = moved
        return torch.where(legal.unsqueeze(1), out, states)

    def neighbors(self, states):
        """[B, 4, S] — successor under each action (no-op where illegal)."""
        B = states.shape[0]
        outs = []
        for a in range(4):
            aa = torch.full((B,), a, dtype=torch.long, device=self.device)
            outs.append(self.step(states, aa))
        return torch.stack(outs, dim=1)

    def is_solved(self, states):
        return (states == self.solved).all(dim=1)

    def scramble(self, B, depths, return_actions=False, generator=None):
        """Random walk of legal moves, no immediate backtrack."""
        if isinstance(depths, int):
            depths = torch.full((B,), depths, device=self.device, dtype=torch.long)
        states = self.solved_batch(B)
        prev = torch.full((B,), -1, dtype=torch.long, device=self.device)
        maxd = int(depths.max().item())
        hist = []
        for t in range(maxd):
            legal = self.legal_mask(states)
            # forbid undoing the previous move
            has_prev = prev >= 0
            legal[has_prev, self.inverse_action[prev[has_prev]]] = False
            # uniform sample over legal moves via Gumbel-argmax (fast, batched)
            u = torch.rand(B, 4, device=self.device, generator=generator).clamp_min(1e-9)
            gum = -torch.log(-torch.log(u))
            a = gum.masked_fill(~legal, -1e9).argmax(1)
            active = t < depths
            nxt = self.step(states, a)
            states = torch.where(active.unsqueeze(1), nxt, states)
            prev = torch.where(active, a, prev)
            if return_actions:
                hist.append(torch.where(active, a, torch.full_like(a, -1)))
        if return_actions:
            return states, torch.stack(hist, 1)
        return states


def pack(states, n):
    """[B, S] -> int64 key, base-S digits (fits: 25^25 > 2^63 only for n=5;
    use base-S with S<=16 exact; for n=5 use double-int hashing instead)."""
    S = n * n
    assert S <= 16, "exact packing only for 8/15-puzzle"
    powers = torch.tensor([S ** i for i in range(S)], dtype=torch.int64,
                          device=states.device)
    return (states.to(torch.int64) * powers).sum(1)


def unpack(keys, n):
    S = n * n
    k = keys.clone()
    out = torch.empty(k.shape[0], S, dtype=torch.int8, device=keys.device)
    for i in range(S):
        out[:, i] = (k % S).to(torch.int8)
        k //= S
    return out


def bfs_oracle(n, device="cuda", expected=None):
    """Exhaustive BFS from the goal (8-puzzle: 181,440 reachable states)."""
    env = SlideEnv(n, device)
    solved = env.solved_batch(1)
    frontier = solved
    visited = pack(solved, n)
    keys_all = [visited.clone()]
    dists_all = [torch.zeros(1, dtype=torch.int16, device=device)]
    d = 0
    while frontier.shape[0] > 0:
        nb = env.neighbors(frontier).reshape(-1, env.S)
        keys = pack(nb, n)
        order = keys.argsort()
        ks, st = keys[order], nb[order]
        first = torch.ones_like(ks, dtype=torch.bool)
        first[1:] = ks[1:] != ks[:-1]
        ks, st = ks[first], st[first]
        new = ~torch.isin(ks, visited, assume_unique=True)
        ks, frontier = ks[new], st[new]
        d += 1
        if ks.numel() == 0:
            break
        keys_all.append(ks)
        dists_all.append(torch.full((ks.numel(),), d, dtype=torch.int16, device=device))
        visited = torch.sort(torch.cat([visited, ks])).values
    keys = torch.cat(keys_all)
    dists = torch.cat(dists_all)
    order = keys.argsort()
    keys, dists = keys[order], dists[order]
    if expected is not None:
        assert keys.numel() == expected, f"{keys.numel()} != {expected}"
    return keys, dists


def run_tests(device="cuda"):
    for n in (3, 4, 5):
        env = SlideEnv(n, device)
        B = 512
        st = env.scramble(B, 50)
        # legal step then inverse returns to origin
        legal = env.legal_mask(st)
        a = torch.multinomial(legal.float(), 1).squeeze(1)
        back = env.step(env.step(st, a), env.inverse_action[a])
        assert torch.equal(back, st), f"{n}: inverse broken"
        # illegal actions are no-ops
        illegal_rows = (~legal).any(1).nonzero(as_tuple=True)[0]
        if illegal_rows.numel():
            r0 = illegal_rows[0]
            bad_a = (~legal[r0]).float().argmax()
            same = env.step(st[r0:r0+1], bad_a.view(1))
            assert torch.equal(same, st[r0:r0+1]), f"{n}: illegal not no-op"
        # scramble depth honored
        depths = torch.tensor([0, 30] * 8, device=device)
        s2 = env.scramble(16, depths)
        assert env.is_solved(s2)[::2].all()
        assert not env.is_solved(s2)[1::2].any()
        # exactly one blank always
        assert ((st == 0).sum(1) == 1).all()
        print(f"{n}x{n} slide env tests passed")
    # 8-puzzle reachable set is exactly 9!/2 = 181,440 with diameter 31
    keys, dists = bfs_oracle(3, device, expected=181440)
    dia = int(dists.max())
    assert dia == 31, f"8-puzzle diameter {dia} != 31"
    print(f"8-puzzle oracle: 181,440 states, diameter 31  ✓ (known values)")
    return keys, dists


if __name__ == "__main__":
    run_tests()
