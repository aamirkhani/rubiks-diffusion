"""Vectorized procedural gridworld mazes.

Domain 2 of the beyond-groups study: removes the *single fixed goal /
fixed environment* assumption. Every instance is a fresh random obstacle
grid with its own goal cell, so the denoiser must be CONDITIONAL —
p(a | agent, goal, walls) — the analogue of conditional diffusion.

State encoding (reuses the categorical backbone): [B, n*n] int8 grid,
  0 = open, 1 = wall, 2 = agent, 3 = goal   (agent covers the goal when on it,
  so `is_solved` = no goal marker visible).
Actions: 0=up 1=down 2=left 3=right (agent moves; wall/boundary = no-op).
Forward (noising) process: random walk of the agent AWAY from the goal
through open cells. Exact per-instance oracle: batched BFS distance field.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

OPEN, WALL, AGENT, GOAL, UNKNOWN = 0, 1, 2, 3, 4


class MazeEnv:
    def __init__(self, n=15, wall_p=0.25, device="cuda"):
        self.n = n
        self.S = n * n
        self.M = 4
        self.vocab = 4
        self.wall_p = wall_p
        self.device = torch.device(device)
        self.inverse_action = torch.tensor([1, 0, 3, 2], device=self.device)
        self.drow = torch.tensor([-1, 1, 0, 0], device=self.device)
        self.dcol = torch.tensor([0, 0, -1, 1], device=self.device)

    # ---------------------------------------------------------- instances
    def new_instances(self, B, generator=None):
        """Random wall grids + random goal cell (goal never on a wall)."""
        g = torch.rand(B, self.S, device=self.device, generator=generator)
        walls = (g < self.wall_p).to(torch.int8)
        goal = torch.randint(self.S, (B,), device=self.device, generator=generator)
        rows = torch.arange(B, device=self.device)
        walls[rows, goal] = 0
        return walls, goal

    def make_state(self, walls, agent, goal):
        st = walls.clone()
        rows = torch.arange(walls.shape[0], device=self.device)
        st[rows, goal] = GOAL
        st[rows, agent] = AGENT          # covers goal if agent==goal
        return st

    def agent_pos(self, states):
        return (states == AGENT).float().argmax(dim=1).long()

    def goal_visible(self, states):
        return (states == GOAL).any(dim=1)

    def is_solved(self, states):
        return ~self.goal_visible(states)

    def legal_mask(self, states):
        ap = self.agent_pos(states)
        r, c = ap // self.n, ap % self.n
        nr = r.unsqueeze(1) + self.drow.unsqueeze(0)
        nc = c.unsqueeze(1) + self.dcol.unsqueeze(0)
        inb = (nr >= 0) & (nr < self.n) & (nc >= 0) & (nc < self.n)
        tp = (nr.clamp(0, self.n - 1) * self.n + nc.clamp(0, self.n - 1)).long()
        tgt = torch.gather(states, 1, tp)
        return inb & (tgt != WALL) & (tgt != WALL)

    def step(self, states, actions, goal):
        """Move agent; restores the goal marker when leaving the goal cell."""
        B = states.shape[0]
        ap = self.agent_pos(states)
        r, c = ap // self.n, ap % self.n
        nr, nc = r + self.drow[actions], c + self.dcol[actions]
        inb = (nr >= 0) & (nr < self.n) & (nc >= 0) & (nc < self.n)
        tp = (nr.clamp(0, self.n - 1) * self.n + nc.clamp(0, self.n - 1)).long()
        rows = torch.arange(B, device=self.device)
        tgt = states[rows, tp]
        legal = inb & (tgt != WALL)
        out = states.clone()
        # vacate: goal marker back if we were sitting on the goal, else open
        out[rows, ap] = torch.where(ap == goal, torch.tensor(GOAL, dtype=torch.int8,
                                    device=self.device),
                                    torch.tensor(OPEN, dtype=torch.int8,
                                    device=self.device))
        out[rows, tp] = AGENT
        return torch.where(legal.unsqueeze(1), out, states)

    def neighbors(self, states, goal):
        outs = []
        B = states.shape[0]
        for a in range(4):
            aa = torch.full((B,), a, dtype=torch.long, device=self.device)
            outs.append(self.step(states, aa, goal))
        return torch.stack(outs, dim=1)

    # ------------------------------------------------------------ noising
    def scramble(self, walls, goal, depths, return_actions=False, generator=None):
        """Random walk from the goal through open cells (the forward process)."""
        B = walls.shape[0]
        if isinstance(depths, int):
            depths = torch.full((B,), depths, device=self.device, dtype=torch.long)
        states = self.make_state(walls, goal, goal)      # agent starts ON goal
        prev = torch.full((B,), -1, dtype=torch.long, device=self.device)
        maxd = int(depths.max().item())
        hist = []
        for t in range(maxd):
            legal = self.legal_mask(states)
            has = prev >= 0
            legal[has, self.inverse_action[prev[has]]] = False
            # rows with no legal move (fully walled): allow backtrack again
            none = ~legal.any(dim=1)
            if none.any():
                legal[none] = self.legal_mask(states[none])
            still_none = ~legal.any(dim=1)
            legal[still_none, 0] = True                  # will be a no-op
            u = torch.rand(B, 4, device=self.device, generator=generator).clamp_min(1e-9)
            gum = -torch.log(-torch.log(u))
            a = gum.masked_fill(~legal, -1e9).argmax(1)
            active = t < depths
            nxt = self.step(states, a, goal)
            states = torch.where(active.unsqueeze(1), nxt, states)
            moved = active & ~still_none
            prev = torch.where(moved, a, prev)
            if return_actions:
                hist.append(torch.where(moved, a, torch.full_like(a, -1)))
        if return_actions:
            return states, torch.stack(hist, 1)
        return states

    # ------------------------------------------------------------- oracle
    def bfs_field(self, walls, goal):
        """Exact distance-to-goal field per instance: [B, S] int32 (-1 = unreachable)."""
        B = walls.shape[0]
        n = self.n
        INF = 10 ** 6
        dist = torch.full((B, self.S), INF, device=self.device, dtype=torch.int32)
        rows = torch.arange(B, device=self.device)
        dist[rows, goal] = 0
        open_mask = walls == 0
        d2 = dist.view(B, n, n)
        for _ in range(2 * self.S):
            up = torch.full_like(d2, INF); up[:, :-1, :] = d2[:, 1:, :]
            dn = torch.full_like(d2, INF); dn[:, 1:, :] = d2[:, :-1, :]
            lf = torch.full_like(d2, INF); lf[:, :, :-1] = d2[:, :, 1:]
            rt = torch.full_like(d2, INF); rt[:, :, 1:] = d2[:, :, :-1]
            best = torch.minimum(torch.minimum(up, dn), torch.minimum(lf, rt)) + 1
            new = torch.minimum(d2, best.view(B, n, n))
            new = torch.where(open_mask.view(B, n, n), new, d2)
            if torch.equal(new, d2):
                break
            d2 = new
        dist = d2.reshape(B, self.S)
        dist[rows, goal] = 0
        return torch.where(dist >= INF, torch.full_like(dist, -1), dist)


def run_tests(device="cuda"):
    env = MazeEnv(15, device=device)
    g = torch.Generator(device=device).manual_seed(0)
    walls, goal = env.new_instances(4096, generator=g)
    # scramble then check: agent ends on an open reachable cell; goal restored
    st, acts = env.scramble(walls, goal, 40, return_actions=True, generator=g)
    assert ((st == AGENT).sum(1) == 1).all()
    solved = env.is_solved(st)
    field = env.bfs_field(walls, goal)
    ap = env.agent_pos(st)
    rows = torch.arange(walls.shape[0], device=device)
    d = field[rows, ap]
    assert (d[~solved] > 0).all(), "unsolved state with zero/unreachable distance"
    assert (d >= 0).all(), "walk reached unreachable cell?!"
    # walk length upper-bounds true distance
    walk_len = (acts >= 0).sum(1)
    assert (d <= walk_len).all(), "true distance exceeds walk length!"
    # step/inverse consistency on legal moves
    legal = env.legal_mask(st)
    a = legal.float().argmax(1)
    ok = legal.any(1)
    fwd = env.step(st[ok], a[ok], goal[ok])
    back = env.step(fwd, env.inverse_action[a[ok]], goal[ok])
    assert torch.equal(back, st[ok]), "maze inverse broken"
    # oracle sanity on an empty maze: distance = manhattan
    w0 = torch.zeros(1, env.S, dtype=torch.int8, device=device)
    g0 = torch.tensor([0], device=device)
    f0 = env.bfs_field(w0, g0)
    exp = torch.tensor([[r + c for c in range(15)] for r in range(15)],
                       device=device, dtype=torch.int32).reshape(1, -1)
    assert torch.equal(f0, exp), "BFS field != manhattan on empty grid"
    print("maze env tests passed (4096 instances, oracle-checked)")


if __name__ == "__main__":
    run_tests()


def observe(states, n, radius):
    """POMDP view: cells beyond Chebyshev `radius` of the agent -> UNKNOWN.
    The goal cell stays visible as a compass (goal direction is knowable)."""
    import torch as _t
    B = states.shape[0]
    ap = (states == AGENT).float().argmax(dim=1)
    ar, ac = ap // n, ap % n
    rr = _t.arange(n, device=states.device)
    gr = rr.view(1, n, 1).expand(B, n, n)
    gc = rr.view(1, 1, n).expand(B, n, n)
    dist = _t.maximum((gr - ar.view(-1, 1, 1)).abs(),
                      (gc - ac.view(-1, 1, 1)).abs()).reshape(B, -1)
    hidden = (dist > radius) & (states != GOAL)
    out = states.clone()
    out[hidden] = UNKNOWN
    return out
