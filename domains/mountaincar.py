"""Mountain Car as denoising diffusion.

Domain 9: NON-MONOTONE progress. The underpowered car must drive AWAY from
the goal to build momentum -- the classic trap for greedy local descent.
Deterministic, discrete actions {push-left, coast, push-right}, continuous
state (x, v). Noising: exact reverse integration from goal states under
random actions. Denoiser: categorical over 3 actions (the pendulum lesson
applied). Baseline: cost-to-go net + 1-step lookahead.

  python domains/mountaincar.py --test
  python domains/mountaincar.py --method denoise --iters 20000 --out runs/mc_diff
  python domains/mountaincar.py --method value   --iters 20000 --out runs/mc_value
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

XMIN, XMAX, VMAX = -1.2, 0.6, 0.07
GOAL_X = 0.5
FORCE, GRAV = 0.001, 0.0025


def fwd_step(x, v, a):
    """a in {0,1,2} -> push {-1,0,+1}. Gym MountainCar-v0 dynamics."""
    push = (a.float() - 1.0)
    v2 = (v + push * FORCE - GRAV * torch.cos(3 * x)).clamp(-VMAX, VMAX)
    x2 = (x + v2).clamp(XMIN, XMAX)
    # gym: hitting the left wall zeroes velocity
    v2 = torch.where((x2 <= XMIN) & (v2 < 0), torch.zeros_like(v2), v2)
    return x2, v2


def rev_step(x2, v2, a):
    """Inverse of fwd_step away from the clamps."""
    x = x2 - v2
    push = (a.float() - 1.0)
    v = v2 - push * FORCE + GRAV * torch.cos(3 * x)
    return x, v


def is_goal(x, v):
    return x >= GOAL_X


def feats(x, v):
    return torch.stack([x, v / VMAX, torch.cos(3 * x)], dim=1)


class Net(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 256), nn.SiLU(),
                               nn.Linear(256, 256), nn.SiLU(),
                               nn.Linear(256, 256), nn.SiLU(),
                               nn.Linear(256, out))

    def forward(self, x, v):
        return self.f(feats(x, v)).squeeze(-1)


def noising_batch(B, K, device, generator=None):
    x = GOAL_X + torch.rand(B, device=device, generator=generator) * 0.1
    v = torch.rand(B, device=device, generator=generator) * VMAX
    depths = torch.randint(1, K + 1, (B,), device=device, generator=generator)
    last_a = torch.zeros(B, dtype=torch.long, device=device)
    for t in range(int(depths.max())):
        a = torch.randint(3, (B,), device=device, generator=generator)
        active = (t < depths)
        x2, v2 = rev_step(x, v, a)
        # keep reverse states inside the track; freeze rows that left it
        okr = (x2 >= XMIN) & (x2 <= XMAX) & (v2.abs() <= VMAX)
        take = active & okr
        x = torch.where(take, x2, x)
        v = torch.where(take, v2, v)
        last_a = torch.where(take, a, last_a)
    return x, v, last_a


@torch.no_grad()
def rollout(net, method, x, v, steps=250):
    B = x.shape[0]
    done = is_goal(x, v)
    t_solve = torch.full((B,), -1, dtype=torch.long, device=x.device)
    for t in range(steps):
        if method == "denoise":
            a = net(x, v).argmax(-1)
        else:
            cand = []
            for aa in range(3):
                x2, v2 = fwd_step(x, v, torch.full_like(t_solve, aa))
                val = net(x2, v2).clamp_min(0)
                val = torch.where(is_goal(x2, v2), torch.zeros_like(val), val)
                cand.append(val)
            a = torch.stack(cand, 1).argmin(1)
        x, v = fwd_step(x, v, a)
        newly = is_goal(x, v) & ~done
        t_solve = torch.where(newly, torch.full_like(t_solve, t), t_solve)
        done = done | newly
    return done, t_solve


def train(method, iters, out_dir, K=250, batch=8192, lr=1e-3, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    net = Net(3 if method == "denoise" else 1).to(device)
    if method == "value":
        target = Net(1).to(device)
        target.load_state_dict(net.state_dict())
        for p in target.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mfh = open(os.path.join(out_dir, "metrics.jsonl"), "a", buffering=1)
    ema, stale, tgt_updates = None, 0, 0
    for it in range(1, iters + 1):
        if it == int(iters * 0.8) + 1:
            for g in opt.param_groups:
                g["lr"] = lr * 0.1
        x, v, a_lab = noising_batch(batch, K, device)
        if method == "denoise":
            loss = F.cross_entropy(net(x, v), a_lab)
        else:
            with torch.no_grad():
                best = None
                for aa in range(3):
                    x2, v2 = fwd_step(x, v, torch.full((batch,), aa,
                                      dtype=torch.long, device=device))
                    val = target(x2, v2).clamp_min(0)
                    val = torch.where(is_goal(x2, v2), torch.zeros_like(val), val)
                    best = val if best is None else torch.minimum(best, val)
                y = 1.0 + best
                y = torch.where(is_goal(x, v), torch.zeros_like(y), y)
            loss = ((net(x, v) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        l = loss.item()
        ema = l if ema is None else 0.95 * ema + 0.05 * l
        if method == "value" and it % 100 == 0:
            stale += 100
            if ema < 0.05 or stale >= 3000:
                target.load_state_dict(net.state_dict()); tgt_updates += 1; stale = 0
        if it % 500 == 0:
            mfh.write(json.dumps({"iter": it, "ema": round(ema, 4),
                                  "tgt_updates": tgt_updates}) + "\n")
        if it % 5000 == 0:
            g = torch.Generator(device=device).manual_seed(1)
            x0 = -0.5 + torch.randn(2048, device=device, generator=g) * 0.05
            v0 = torch.zeros(2048, device=device)
            done, ts = rollout(net, method, x0, v0)
            rec = {"iter": it, "reach_rate": round(done.float().mean().item(), 4),
                   "mean_steps": round(ts[done].float().mean().item(), 1)
                   if done.any() else None}
            mfh.write(json.dumps(rec) + "\n")
            print("PROBE " + json.dumps(rec), flush=True)
        if it % 5000 == 0 or it == iters:
            torch.save({"iter": it, "net": net.state_dict(),
                        "cfg": {"method": method, "K": K}},
                       os.path.join(out_dir, "ckpt_latest.pt"))
    print("training complete", flush=True)


def test(device="cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.rand(100000, device=device, generator=g) * 1.6 - 1.1
    v = (torch.rand(100000, device=device, generator=g) * 2 - 1) * VMAX * 0.9
    a = torch.randint(3, (100000,), device=device, generator=g)
    xp, vp = rev_step(x, v, a)
    ok = (xp > XMIN + 0.01) & (xp < XMAX - 0.01) & (vp.abs() < VMAX - 1e-4)
    x2, v2 = fwd_step(xp, vp, a)
    err = ((x2 - x).abs() + (v2 - v).abs())[ok].max()
    assert err < 1e-6, err
    xx, vv, _ = noising_batch(100000, 250, device)
    frac_deep = ((xx < -0.4)).float().mean().item()
    assert frac_deep > 0.2, frac_deep
    print(f"mountain-car tests passed: exact reverse (err {err:.1e}); "
          f"{frac_deep:.0%} of noised states in the valley")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--method", choices=["denoise", "value"])
    ap.add_argument("--iters", type=int)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.test:
        test()
    else:
        train(args.method, args.iters, args.out)
