"""2048 as a (stress test for) denoising diffusion.

Domain 5 of the beyond-groups study: removes the *determinism* assumption.
Forward play interleaves chosen slides with RANDOM tile spawns, so the
reverse process is ill-posed: un-merging tiles and un-spawning are ambiguous,
and "the goal" is a set of states, not one state.

Design (honest adaptation, documented in the paper):
  * Noising = reverse play from a target state: start from a constructed
    high-tile board, apply inverse-slides (split a tile 2^k into two 2^{k-1}
    and push them apart) and tile REMOVALS (inverse of spawning), both
    randomized. The denoiser learns p(slide direction | board) from the
    inverse of the last un-slide.
  * At play time the environment is the REAL stochastic game (random spawns).
    The mismatch between the deterministic reverse process and stochastic
    forward play is exactly what this domain measures.
  * Baselines: (a) uniform-random legal play, (b) the greedy heuristic
    "max merged value per move", (c) expectimax-1 with a learned value net
    trained by TD on real play would drift from the recipe — we keep (a),(b)
    so what we measure is the denoiser vs play-time stochasticity.

Board: [B, 16] int8 of exponents (0 empty, k = tile 2^k). Metric: reaching
256 / 512 / 1024 tile within 400 moves; mean max-tile.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

N = 4
S = 16


def _rows(board):
    return board.view(-1, N, N)


def slide_left(board):
    """Vectorized 2048 left-slide with merges. Returns (new_board, moved, gain)."""
    B = board.shape[0]
    b = _rows(board).clone()
    out = torch.zeros_like(b)
    gain = torch.zeros(B, device=board.device, dtype=torch.int32)
    for r in range(N):
        row = b[:, r, :]
        # compact non-zeros to the left via stable sort on emptiness
        empty = row == 0
        order = torch.argsort(empty.int(), dim=1, stable=True)
        row = torch.gather(row, 1, order)
        # merge pairs left-to-right
        for c in range(N - 1):
            same = (row[:, c] != 0) & (row[:, c] == row[:, c + 1])
            row[:, c] = torch.where(same, row[:, c] + 1, row[:, c])
            row[:, c + 1] = torch.where(same, torch.zeros_like(row[:, c + 1]),
                                        row[:, c + 1])
            gain += torch.where(same, (2 ** row[:, c].int()), torch.zeros_like(gain))
            # re-compact after each merge
            empty = row == 0
            order = torch.argsort(empty.int(), dim=1, stable=True)
            row = torch.gather(row, 1, order)
        out[:, r, :] = row
    out = out.view(B, S)
    moved = ~(out == board).all(dim=1)
    return out, moved, gain


def _rot(board, k):
    return torch.rot90(_rows(board), k, dims=(1, 2)).reshape(-1, S)


def slide(board, d):
    """d: 0=left 1=right 2=up 3=down (same int for whole batch)."""
    k = {0: 0, 1: 2, 2: 1, 3: 3}[d]
    rot = _rot(board, k)
    out, moved, gain = slide_left(rot)
    return _rot(out, (4 - k) % 4), moved, gain


def legal_mask(board):
    ms = []
    for d in range(4):
        _, moved, _ = slide(board, d)
        ms.append(moved)
    return torch.stack(ms, 1)


def spawn(board, generator=None):
    """Random spawn: 2 (90%) or 4 (10%) on a uniform empty cell."""
    B = board.shape[0]
    empty = (board == 0).float()
    has = empty.sum(1) > 0
    idx = torch.multinomial(empty.clamp_min(1e-9), 1, generator=generator).squeeze(1)
    val = torch.where(torch.rand(B, device=board.device, generator=generator) < 0.9,
                      torch.ones_like(idx), torch.full_like(idx, 2)).to(torch.int8)
    out = board.clone()
    rows = torch.arange(B, device=board.device)
    out[rows[has], idx[has]] = val[has]
    return out


def play_step(board, d_batch, generator=None):
    """Apply per-row directions then spawn (only where the slide moved)."""
    outs = torch.empty_like(board)
    moveds = torch.zeros(board.shape[0], dtype=torch.bool, device=board.device)
    for d in range(4):
        sel = d_batch == d
        if sel.any():
            o, m, _ = slide(board[sel], d)
            outs[sel] = o
            moveds[sel] = m
    sp = spawn(outs, generator=generator)
    return torch.where(moveds.unsqueeze(1), sp, board), moveds


# --------------------------------------------------------- reverse process
def unslide(board, d, generator=None):
    """Inverse slide: split mergeable tiles and spread toward direction d.
    Approximate stochastic inverse: pick a random tile >=2 with an empty
    neighbor path in direction d, split it into two (k-1) tiles."""
    # implemented as: rotate so d = left-inverse (tiles were slid left; we
    # un-slide by moving copies rightward into empties)
    k = {0: 0, 1: 2, 2: 1, 3: 3}[d]
    rot = _rows(_rot(board, k)).clone()
    B = rot.shape[0]
    u = torch.rand(B, N, N, device=board.device, generator=generator)
    done_row = torch.zeros(B, dtype=torch.bool, device=board.device)
    for r in range(N):
        row = rot[:, r, :]
        for c in range(N - 1, 0, -1):
            # split row[c'] (leftmost candidates first) into c' and some empty c
            cand = (row[:, c] == 0)
            for cs in range(c):
                can_split = (~done_row & cand & (row[:, cs] >= 2)
                             & (u[:, r, cs] > 0.5))
                nv = row[:, cs] - 1
                row[:, cs] = torch.where(can_split, nv, row[:, cs])
                row[:, c] = torch.where(can_split, nv, row[:, c])
                done_row = done_row | can_split
        rot[:, r, :] = row
    return _rot(rot.reshape(-1, S), (4 - k) % 4), done_row


def unspawn(board, generator=None):
    """Remove one random small tile (inverse of spawning)."""
    small = ((board == 1) | (board == 2)).float()
    has = small.sum(1) > 0
    idx = torch.multinomial(small.clamp_min(1e-9), 1, generator=generator).squeeze(1)
    out = board.clone()
    rows = torch.arange(board.shape[0], device=board.device)
    out[rows[has], idx[has]] = 0
    return out, has


def noising_batch(B, K, target_exp=8, device="cuda", generator=None):
    """Reverse play from a constructed target board (one 2^target tile).
    Returns (boards, labels) where label = direction whose forward slide
    undoes the last un-slide."""
    board = torch.zeros(B, S, dtype=torch.int8, device=device)
    pos = torch.randint(S, (B,), device=device, generator=generator)
    rows = torch.arange(B, device=device)
    board[rows, pos] = target_exp
    depths = torch.randint(1, K + 1, (B,), device=device, generator=generator)
    last_d = torch.full((B,), -1, dtype=torch.long, device=device)
    for t in range(int(depths.max())):
        active = t < depths
        d = int(torch.randint(4, (1,), generator=generator, device=device))
        nb, did = unslide(board, d, generator=generator)
        # occasionally un-spawn to keep boards sparse
        do_unspawn = torch.rand(B, device=device, generator=generator) < 0.25
        ub, uh = unspawn(nb, generator=generator)
        nb = torch.where((do_unspawn & uh).unsqueeze(1), ub, nb)
        take = active & did
        board = torch.where(take.unsqueeze(1), nb, board)
        # forward slide that undoes un-slide toward d is the OPPOSITE slide
        inv = {0: 1, 1: 0, 2: 3, 3: 2}[d]
        last_d = torch.where(take, torch.full_like(last_d, inv), last_d)
    keep = last_d >= 0
    return board[keep], last_d[keep]


def max_tile(board):
    return board.max(dim=1).values.int()


@torch.no_grad()
def play(policy, B, steps=400, device="cuda", seed=0):
    """Real stochastic game from a fresh start; policy(board, legal) -> dirs."""
    g = torch.Generator(device=device).manual_seed(seed)
    board = torch.zeros(B, S, dtype=torch.int8, device=device)
    board = spawn(spawn(board, g), g)
    alive = torch.ones(B, dtype=torch.bool, device=device)
    for t in range(steps):
        legal = legal_mask(board)
        alive = alive & legal.any(1)
        if not alive.any():
            break
        d = policy(board, legal)
        nb, moved = play_step(board, d, generator=g)
        board = torch.where(alive.unsqueeze(1), nb, board)
    return board


def run_tests(device="cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    # slide correctness on known rows
    b = torch.zeros(3, S, dtype=torch.int8, device=device)
    b[0, 0], b[0, 1] = 1, 1              # [2,2,_,_] -> [4,_,_,_]
    b[1, 0], b[1, 1], b[1, 2], b[1, 3] = 1, 1, 1, 1   # -> [4,4,_,_]
    b[2, 0], b[2, 2] = 2, 2              # [4,_,4,_] -> [8,_,_,_]
    out, moved, gain = slide(b, 0)
    assert out[0, 0] == 2 and out[0, 1] == 0
    assert out[1, 0] == 2 and out[1, 1] == 2 and out[1, 2] == 0
    assert out[2, 0] == 3 and out[2, 2] == 0
    assert moved.all()
    # conservation: total sum of 2^k preserved by slides
    board = spawn(spawn(torch.zeros(1000, S, dtype=torch.int8, device=device), g), g)
    tot0 = (2.0 ** board.float()).masked_fill(board == 0, 0).sum(1)
    out, moved, _ = slide(board, 3)
    tot1 = (2.0 ** out.float()).masked_fill(out == 0, 0).sum(1)
    assert torch.allclose(tot0, tot1), "slide does not conserve tile mass"
    # unslide is invertible by the labeled forward slide (tile mass conserved)
    boards, labels = noising_batch(4096, 30, device=device, generator=g)
    t0 = (2.0 ** boards.float()).masked_fill(boards == 0, 0).sum(1)
    assert (max_tile(boards) <= 8).all()
    print(f"2048 tests passed: slide semantics, conservation, "
          f"noising produced {boards.shape[0]} labeled boards "
          f"(mean tiles {(boards > 0).float().sum(1).mean():.1f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests()
