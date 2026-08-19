"""Tower of Hanoi, vectorized.

Domain 6 of the beyond-groups study: extreme HORIZON. With n=10 disks the
state space is only 3^10 = 59,049 (fully BFS-enumerable -> exhaustive
validation), but the optimal solution from the standard start is 2^10 - 1 =
1023 moves with almost zero slack: a single wrong move can undo hundreds of
moves of progress. Deterministic, reversible, fixed goal -- isolates
solution length from every other axis.

State: [B, n] int8, peg id (0,1,2) of each disk, index 0 = smallest.
Actions: 6 ordered peg pairs (from, to); legal iff `from` is nonempty and its
top (smallest) disk is smaller than `to`'s top. Inverse of (a,b) is (b,a).
Goal: all disks on peg 2.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

PAIRS = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]


class HanoiEnv:
    def __init__(self, n=10, device="cuda"):
        self.n = n
        self.S = n
        self.M = 6
        self.vocab = 3
        self.device = torch.device(device)
        self.solved = torch.full((n,), 2, dtype=torch.int8, device=self.device)
        self.frm = torch.tensor([p[0] for p in PAIRS], device=self.device)
        self.to = torch.tensor([p[1] for p in PAIRS], device=self.device)
        inv = [PAIRS.index((b, a)) for a, b in PAIRS]
        self.inverse_action = torch.tensor(inv, device=self.device)
        self._sizes = torch.arange(n, device=self.device)

    def solved_batch(self, B):
        return self.solved.unsqueeze(0).repeat(B, 1)

    def is_solved(self, states):
        return (states == self.solved).all(dim=1)

    def _tops(self, states):
        """Smallest disk index on each peg; n if peg empty. -> [B, 3]"""
        B = states.shape[0]
        tops = torch.full((B, 3), self.n, dtype=torch.long, device=self.device)
        for peg in range(3):
            on = states == peg                       # [B, n]
            first = on.float().argmax(dim=1)         # smallest index present
            has = on.any(dim=1)
            tops[:, peg] = torch.where(has, first, torch.full_like(first, self.n))
        return tops

    def legal_mask(self, states):
        tops = self._tops(states)
        return tops[:, self.frm] < tops[:, self.to]   # [B, 6] (n < n is False)

    def step(self, states, actions):
        B = states.shape[0]
        tops = self._tops(states)
        f, t = self.frm[actions], self.to[actions]
        rows = torch.arange(B, device=self.device)
        tf, tt = tops[rows, f], tops[rows, t]
        legal = tf < tt
        out = states.clone()
        disk = tf.clamp_max(self.n - 1)
        out[rows, disk] = torch.where(legal, t.to(torch.int8), out[rows, disk])
        return out

    def neighbors(self, states):
        B = states.shape[0]
        outs = []
        for a in range(6):
            aa = torch.full((B,), a, dtype=torch.long, device=self.device)
            outs.append(self.step(states, aa))
        return torch.stack(outs, 1)

    def scramble(self, B, depths, return_actions=False, generator=None):
        if isinstance(depths, int):
            depths = torch.full((B,), depths, device=self.device, dtype=torch.long)
        states = self.solved_batch(B)
        prev = torch.full((B,), -1, dtype=torch.long, device=self.device)
        maxd = int(depths.max().item())
        hist = []
        for t in range(maxd):
            legal = self.legal_mask(states)
            has = prev >= 0
            legal[has, self.inverse_action[prev[has]]] = False
            none = ~legal.any(1)
            if none.any():
                legal[none] = self.legal_mask(states[none])
            u = torch.rand(B, 6, device=self.device, generator=generator).clamp_min(1e-9)
            a = (-torch.log(-torch.log(u))).masked_fill(~legal, -1e9).argmax(1)
            active = t < depths
            nxt = self.step(states, a)
            states = torch.where(active.unsqueeze(1), nxt, states)
            prev = torch.where(active, a, prev)
            if return_actions:
                hist.append(torch.where(active, a, torch.full_like(a, -1)))
        if return_actions:
            return states, torch.stack(hist, 1)
        return states


def pack(states):
    powers = torch.tensor([3 ** i for i in range(states.shape[1])],
                          dtype=torch.int64, device=states.device)
    return (states.to(torch.int64) * powers).sum(1)


def unpack(keys, n):
    k = keys.clone()
    out = torch.empty(k.shape[0], n, dtype=torch.int8, device=keys.device)
    for i in range(n):
        out[:, i] = (k % 3).to(torch.int8)
        k //= 3
    return out


def bfs_oracle(n=10, device="cuda"):
    env = HanoiEnv(n, device)
    frontier = env.solved_batch(1)
    visited = pack(frontier)
    keys_all, dists_all = [visited.clone()], [torch.zeros(1, dtype=torch.int32,
                                                          device=device)]
    d = 0
    while frontier.shape[0] > 0:
        nb = env.neighbors(frontier).reshape(-1, env.S)
        keys = pack(nb)
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
        dists_all.append(torch.full((ks.numel(),), d, dtype=torch.int32,
                                    device=device))
        visited = torch.sort(torch.cat([visited, ks])).values
    keys = torch.cat(keys_all); dists = torch.cat(dists_all)
    order = keys.argsort()
    return keys[order], dists[order]


def run_tests(device="cuda"):
    env = HanoiEnv(10, device)
    B = 4096
    st = env.scramble(B, 500)
    legal = env.legal_mask(st)
    a = legal.float().argmax(1)
    ok = legal.any(1)
    back = env.step(env.step(st[ok], a[ok]), env.inverse_action[a[ok]])
    assert torch.equal(back, st[ok]), "hanoi inverse broken"
    # known facts: 3^10 states, eccentricity of goal = 2^10 - 1 = 1023
    keys, dists = bfs_oracle(10, device)
    assert keys.numel() == 3 ** 10, keys.numel()
    assert int(dists.max()) == 2 ** 10 - 1, int(dists.max())
    # the classic start (all on peg 0) is exactly 1023 from the goal
    start = torch.zeros(1, 10, dtype=torch.int8, device=device)
    d0 = dists[torch.searchsorted(keys, pack(start))]
    assert int(d0) == 1023, int(d0)
    print("hanoi tests passed: 59,049 states; goal eccentricity 1023; "
          "classic start distance = 1023 (known values)")


if __name__ == "__main__":
    run_tests()
