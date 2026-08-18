"""Vectorized Sokoban.

Domain 3 of the beyond-groups study: removes the *invertible actions*
assumption. Forward dynamics push boxes (irreversible: you cannot pull);
the noising process uses PULLS — reverse dynamics available only to the
training generator. A pull (or plain move) in direction d is undone by the
forward move in direction -d (which pushes the box back if one was pulled),
so the denoising label is always inverse_action[d].

Structured state: (walls [B,S] bool, goals [B,S] bool, boxes [B,S] bool,
agent [B] long). Rendered grid for the nets: int8 with
  0 open, 1 wall, 2 box, 3 goal, 4 box-on-goal, 5 agent, 6 agent-on-goal.
Solved: every box sits on a goal (no rendered value 2).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class SokobanEnv:
    def __init__(self, n=8, n_boxes=3, wall_p=0.12, device="cuda"):
        self.n = n
        self.S = n * n
        self.M = 4
        self.vocab = 7
        self.n_boxes = n_boxes
        self.wall_p = wall_p
        self.device = torch.device(device)
        self.inverse_action = torch.tensor([1, 0, 3, 2], device=self.device)
        self.dr = torch.tensor([-1, 1, 0, 0], device=self.device)
        self.dc = torch.tensor([0, 0, -1, 1], device=self.device)

    # ------------------------------------------------------------ helpers
    def _shift_ok(self, pos, d):
        """target cell of moving from pos in direction d; validity mask."""
        r, c = pos // self.n, pos % self.n
        nr, nc = r + self.dr[d], c + self.dc[d]
        ok = (nr >= 0) & (nr < self.n) & (nc >= 0) & (nc < self.n)
        return (nr.clamp(0, self.n - 1) * self.n + nc.clamp(0, self.n - 1)).long(), ok

    def render(self, walls, goals, boxes, agent):
        B = walls.shape[0]
        st = torch.zeros(B, self.S, dtype=torch.int8, device=self.device)
        st[walls] = 1
        st[goals] = 3
        st[boxes & goals] = 4
        st[boxes & ~goals] = 2
        rows = torch.arange(B, device=self.device)
        on_goal = goals[rows, agent]
        st[rows, agent] = torch.where(on_goal, torch.tensor(6, dtype=torch.int8,
                                      device=self.device),
                                      torch.tensor(5, dtype=torch.int8,
                                      device=self.device))
        return st

    def is_solved(self, walls, goals, boxes, agent):
        return (boxes == goals).all(dim=1) | (boxes & ~goals).sum(1).eq(0) & \
               (goals & ~boxes).sum(1).eq(0)

    # ------------------------------------------------------- forward step
    def legal_forward(self, walls, goals, boxes, agent):
        """[B,4] — plain move or push allowed."""
        B = walls.shape[0]
        rows = torch.arange(B, device=self.device)
        out = torch.zeros(B, 4, dtype=torch.bool, device=self.device)
        for d in range(4):
            dd = torch.full((B,), d, dtype=torch.long, device=self.device)
            t1, ok1 = self._shift_ok(agent, dd)
            t2, ok2 = self._shift_ok(t1, dd)
            w1 = walls[rows, t1]
            b1 = boxes[rows, t1]
            w2 = walls[rows, t2]
            b2 = boxes[rows, t2]
            plain = ok1 & ~w1 & ~b1
            push = ok1 & ~w1 & b1 & ok2 & ~w2 & ~b2
            out[:, d] = plain | push
        return out

    def step_forward(self, walls, goals, boxes, agent, actions):
        """Apply moves/pushes; illegal = no-op. Returns (boxes, agent)."""
        B = walls.shape[0]
        rows = torch.arange(B, device=self.device)
        t1, ok1 = self._shift_ok(agent, actions)
        t2, ok2 = self._shift_ok(t1, actions)
        w1, b1 = walls[rows, t1], boxes[rows, t1]
        w2, b2 = walls[rows, t2], boxes[rows, t2]
        plain = ok1 & ~w1 & ~b1
        push = ok1 & ~w1 & b1 & ok2 & ~w2 & ~b2
        legal = plain | push
        new_boxes = boxes.clone()
        do_push = push
        new_boxes[rows[do_push], t1[do_push]] = False
        new_boxes[rows[do_push], t2[do_push]] = True
        new_agent = torch.where(legal, t1, agent)
        return new_boxes, new_agent

    def neighbors_forward(self, walls, goals, boxes, agent):
        """Rendered successor states under each forward action: [B,4,S]."""
        outs = []
        B = walls.shape[0]
        for d in range(4):
            dd = torch.full((B,), d, dtype=torch.long, device=self.device)
            nb, na = self.step_forward(walls, goals, boxes, agent, dd)
            outs.append(self.render(walls, goals, nb, na))
        return torch.stack(outs, dim=1)

    # ----------------------------------------------------- reverse (pull)
    def instances_and_scramble(self, B, depths, p_pull=0.7, return_labels=True,
                               generator=None):
        """Generate solvable instances by pulling from a solved layout.

        Returns (walls, goals, boxes, agent, labels) where labels is the
        forward action that undoes the LAST noising op (denoising target).
        """
        n, S = self.n, self.S
        dev = self.device
        rows = torch.arange(B, device=dev)
        # walls: border + sparse interior
        walls = torch.rand(B, S, device=dev, generator=generator) < self.wall_p
        grid = walls.view(B, n, n)
        grid[:, 0, :] = True; grid[:, -1, :] = True
        grid[:, :, 0] = True; grid[:, :, -1] = True
        walls = grid.view(B, S)
        # goals: n_boxes random open cells; boxes start ON goals (solved)
        open_w = (~walls).float()
        goals = torch.zeros(B, S, dtype=torch.bool, device=dev)
        gsel = torch.multinomial(open_w, self.n_boxes, generator=generator)
        goals[rows.unsqueeze(1), gsel] = True
        boxes = goals.clone()
        # agent: random open non-box cell
        aw = (~walls & ~boxes).float()
        agent = torch.multinomial(aw, 1, generator=generator).squeeze(1)

        if isinstance(depths, int):
            depths = torch.full((B,), depths, device=dev, dtype=torch.long)
        maxd = int(depths.max().item())
        last = torch.full((B,), -1, dtype=torch.long, device=dev)
        for t in range(maxd):
            # candidate noising ops per direction: plain move or pull
            legal_any = torch.zeros(B, 4, dtype=torch.bool, device=dev)
            pullable = torch.zeros(B, 4, dtype=torch.bool, device=dev)
            t1s = []
            for d in range(4):
                dd = torch.full((B,), d, dtype=torch.long, device=dev)
                t1, ok1 = self._shift_ok(agent, dd)
                back, okb = self._shift_ok(agent, self.inverse_action[dd])
                free1 = ok1 & ~walls[rows, t1] & ~boxes[rows, t1]
                has_box_behind = okb & boxes[rows, back]
                legal_any[:, d] = free1
                pullable[:, d] = free1 & has_box_behind
                t1s.append(t1)
            u = torch.rand(B, 4, device=dev, generator=generator).clamp_min(1e-9)
            gum = -torch.log(-torch.log(u))
            a = gum.masked_fill(~legal_any, -1e9).argmax(1)
            stuck = ~legal_any.any(1)
            active = (t < depths) & ~stuck
            t1 = torch.stack(t1s, 1)[rows, a]
            do_pull = pullable[rows, a] & \
                (torch.rand(B, device=dev, generator=generator) < p_pull)
            back, _ = self._shift_ok(agent, self.inverse_action[a])
            ap = active & do_pull
            boxes[rows[ap], back[ap]] = False
            boxes[rows[ap], agent[ap]] = True
            agent = torch.where(active, t1, agent)
            last = torch.where(active, a, last)
        labels = self.inverse_action[last.clamp_min(0)]
        labels = torch.where(last >= 0, labels, torch.full_like(labels, -1))
        return walls, goals, boxes, agent, labels


def run_tests(device="cuda"):
    env = SokobanEnv(8, 3, device=device)
    g = torch.Generator(device=device).manual_seed(0)
    B = 4096
    walls, goals, boxes, agent, labels = env.instances_and_scramble(
        B, 25, generator=g)
    # box/goal counts conserved
    assert (boxes.sum(1) == env.n_boxes).all()
    assert (goals.sum(1) == env.n_boxes).all()
    # agent never on wall or box
    rows = torch.arange(B, device=device)
    assert not walls[rows, agent].any()
    assert not boxes[rows, agent].any()
    # replay: undoing with forward moves must re-solve every instance.
    # noising recorded only the last op, so regenerate with full histories:
    torch.manual_seed(1)
    g2 = torch.Generator(device=device).manual_seed(2)
    B2 = 2048
    env2 = SokobanEnv(8, 3, device=device)
    # manual small-scale generator with history
    w, go, bx, ag, _ = env2.instances_and_scramble(B2, 0, generator=g2)
    hist = []
    depths = 20
    for t in range(depths):
        w_, go_, bx_, ag_, lab = None, None, None, None, None
        # single noising step via depths=1 style: reuse method with depth 1
        # (walls/goals fixed; boxes/agent evolve)
        # inline single step: replicate the loop body via depths=1 call is not
        # state-preserving, so do a 1-step scramble manually:
        legal_any = torch.zeros(B2, 4, dtype=torch.bool, device=device)
        pullable = torch.zeros(B2, 4, dtype=torch.bool, device=device)
        rows2 = torch.arange(B2, device=device)
        t1s = []
        for d in range(4):
            dd = torch.full((B2,), d, dtype=torch.long, device=device)
            t1, ok1 = env2._shift_ok(ag, dd)
            back, okb = env2._shift_ok(ag, env2.inverse_action[dd])
            free1 = ok1 & ~w[rows2, t1] & ~bx[rows2, t1]
            legal_any[:, d] = free1
            pullable[:, d] = free1 & okb & bx[rows2, back]
            t1s.append(t1)
        u = torch.rand(B2, 4, device=device).clamp_min(1e-9)
        a = (-torch.log(-torch.log(u))).masked_fill(~legal_any, -1e9).argmax(1)
        ok = legal_any.any(1)
        t1 = torch.stack(t1s, 1)[rows2, a]
        do_pull = pullable[rows2, a] & (torch.rand(B2, device=device) < 0.7)
        back, _ = env2._shift_ok(ag, env2.inverse_action[a])
        ap = ok & do_pull
        bx[rows2[ap], back[ap]] = False
        bx[rows2[ap], ag[ap]] = True
        ag = torch.where(ok, t1, ag)
        hist.append(torch.where(ok, a, torch.full_like(a, -1)))
    # now undo in reverse order with forward dynamics
    for t in range(depths - 1, -1, -1):
        a = hist[t]
        valid = a >= 0
        fa = env2.inverse_action[a.clamp_min(0)]
        nb, na = env2.step_forward(w, go, bx, ag, fa)
        bx = torch.where(valid.unsqueeze(1), nb, bx)
        ag = torch.where(valid, na, ag)
    solved = env2.is_solved(w, go, bx, ag)
    assert solved.all(), f"replay failed on {int((~solved).sum())}/{B2}"
    print(f"sokoban tests passed: counts ok; {B2} pulled scrambles fully "
          f"un-done by forward pushes (solvability by construction verified)")


if __name__ == "__main__":
    run_tests()
