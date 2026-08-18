"""Pendulum swing-up as continuous denoising diffusion.

Domain 4 of the beyond-groups study: removes the *discreteness* assumption.
Continuous state (theta, omega), continuous torque u in [-2, 2]. The goal is
the upright equilibrium. Forward (noising) process: start at the goal and
integrate the dynamics BACKWARD in time under random torques — the exact
inverse of the semi-implicit Euler forward step, so every noising trajectory
is a valid forward trajectory read in reverse. The denoiser regresses the
torque that undoes the last noise increment (literal epsilon-prediction).

Baseline ("value"): cost-to-go net trained by bootstrapping over a torque
grid, solved by 1-step lookahead — the continuous analogue of DAVI.

Underactuated: max torque 2 < mgl = 10, so direct lifting is impossible and
swing-up (energy pumping) is required — the task is nontrivial.

  python domains/pendulum.py --test
  python domains/pendulum.py --method denoise --iters 30000 --out runs/pend_diff
  python domains/pendulum.py --method value   --iters 30000 --out runs/pend_value
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

G, L, MASS, DT, UMAX = 10.0, 1.0, 1.0, 0.05, 2.0
OMEGA_MAX = 8.0


def wrap(th):
    return (th + torch.pi) % (2 * torch.pi) - torch.pi


def fwd_step(th, om, u):
    """Semi-implicit Euler (gym-style). theta=0 is upright."""
    acc = 3 * G / (2 * L) * torch.sin(th) + 3.0 / (MASS * L ** 2) * u
    om2 = (om + DT * acc).clamp(-OMEGA_MAX, OMEGA_MAX)
    th2 = th + DT * om2
    return th2, om2


def rev_step(th2, om2, u):
    """Exact inverse of fwd_step (when omega stays inside the clamp)."""
    th = th2 - DT * om2
    acc = 3 * G / (2 * L) * torch.sin(th) + 3.0 / (MASS * L ** 2) * u
    om = om2 - DT * acc
    return th, om


def feats(th, om):
    return torch.stack([torch.cos(th), torch.sin(th), om / OMEGA_MAX], dim=1)


class Reg(nn.Module):
    def __init__(self, out=1, h=256):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out))

    def forward(self, th, om):
        return self.f(feats(th, om)).squeeze(-1)


def noising_batch(B, K, device, generator=None):
    """Reverse-time walks from the goal. Returns (th, om, label_u, depth)."""
    th = torch.randn(B, device=device, generator=generator) * 0.05
    om = torch.randn(B, device=device, generator=generator) * 0.05
    depths = torch.randint(1, K + 1, (B,), device=device, generator=generator)
    last_u = torch.zeros(B, device=device)
    for t in range(int(depths.max())):
        u = (torch.rand(B, device=device, generator=generator) * 2 - 1) * UMAX
        active = t < depths
        th2, om2 = rev_step(th, om, u)
        th = torch.where(active, th2, th)
        om = torch.where(active, om2, om)
        last_u = torch.where(active, u, last_u)
    return wrap(th), om.clamp(-OMEGA_MAX, OMEGA_MAX), last_u, depths


def is_goal(th, om):
    return (wrap(th).abs() < 0.15) & (om.abs() < 0.6)


NBINS = 21
BIN_CENTERS = None  # set lazily per device


def bins(device):
    global BIN_CENTERS
    if BIN_CENTERS is None or BIN_CENTERS.device != torch.device(device):
        BIN_CENTERS = torch.linspace(-UMAX, UMAX, NBINS, device=device)
    return BIN_CENTERS


@torch.no_grad()
def rollout(net, method, th, om, steps=250, ugrid=None):
    """Greedy control; success = reach AND HOLD the goal for 10 steps."""
    B = th.shape[0]
    hold = torch.zeros(B, dtype=torch.long, device=th.device)
    done = torch.zeros(B, dtype=torch.bool, device=th.device)
    t_solve = torch.full((B,), -1, dtype=torch.long, device=th.device)
    for t in range(steps):
        if method == "denoise":
            u = net(th, om).clamp(-UMAX, UMAX)
        elif method == "denoise-disc":
            u = bins(th.device)[net(th, om).argmax(-1)]
        else:
            # 1-step lookahead over torque grid, pick min predicted cost
            cand = []
            for ug in ugrid:
                th2, om2 = fwd_step(th, om, torch.full_like(th, ug))
                v = net(th2, om2).clamp_min(0)
                v = torch.where(is_goal(th2, om2), torch.zeros_like(v), v)
                cand.append(v)
            u_idx = torch.stack(cand, 1).argmin(1)
            u = ugrid[u_idx]
        th, om = fwd_step(th, om, u)
        atg = is_goal(th, om)
        hold = torch.where(atg, hold + 1, torch.zeros_like(hold))
        newly = (hold >= 10) & ~done
        t_solve = torch.where(newly, torch.full_like(t_solve, t), t_solve)
        done = done | newly
    return done, t_solve


def train(method, iters, out_dir, K=200, batch=8192, lr=1e-3, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    net = Reg(out=NBINS if method == "denoise-disc" else 1).to(device)
    ugrid = torch.linspace(-UMAX, UMAX, 9, device=device)
    if method == "value":
        target = Reg().to(device)
        target.load_state_dict(net.state_dict())
        for p in target.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mfh = open(os.path.join(out_dir, "metrics.jsonl"), "a", buffering=1)
    ema, stale, tgt_updates = None, 0, 0
    t_last, it_last = time.time(), 0
    for it in range(1, iters + 1):
        if it == int(iters * 0.8) + 1:
            for g in opt.param_groups:
                g["lr"] = lr * 0.1
        th, om, u_lab, depths = noising_batch(batch, K, device)
        if method == "denoise":
            loss = ((net(th, om) - u_lab) ** 2).mean()
        elif method == "denoise-disc":
            lab = torch.bucketize(u_lab, bins(device)) .clamp(0, NBINS - 1)
            loss = torch.nn.functional.cross_entropy(net(th, om), lab)
        else:
            with torch.no_grad():
                best = None
                for ug in ugrid:
                    th2, om2 = fwd_step(th, om, torch.full_like(th, ug))
                    v = target(th2, om2).clamp_min(0)
                    v = torch.where(is_goal(th2, om2), torch.zeros_like(v), v)
                    best = v if best is None else torch.minimum(best, v)
                y = 1.0 + best
                y = torch.where(is_goal(th, om), torch.zeros_like(y), y)
            loss = ((net(th, om) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        l = loss.item()
        ema = l if ema is None else 0.95 * ema + 0.05 * l
        if method == "value" and it % 100 == 0:
            stale += 100
            if ema < 0.05 or stale >= 3000:
                target.load_state_dict(net.state_dict()); tgt_updates += 1; stale = 0
        if it % 200 == 0:
            now = time.time()
            ips = (it - it_last) / max(now - t_last, 1e-9)
            t_last, it_last = now, it
            mfh.write(json.dumps({"iter": it, "loss": round(l, 4),
                                  "ema": round(ema, 4), "ips": round(ips, 1),
                                  "tgt_updates": tgt_updates}) + "\n")
        if it % 5000 == 0:
            g = torch.Generator(device=device).manual_seed(1)
            th0 = torch.pi + torch.randn(2048, device=device, generator=g) * 0.1
            om0 = torch.randn(2048, device=device, generator=g) * 0.1
            done, ts = rollout(net, method, th0.clone(), om0.clone(), ugrid=ugrid)
            rec = {"iter": it,
                   "swingup_rate": round(done.float().mean().item(), 4),
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
    th = torch.rand(100000, device=device, generator=g) * 6 - 3
    om = torch.rand(100000, device=device, generator=g) * 12 - 6
    u = (torch.rand(100000, device=device, generator=g) * 2 - 1) * UMAX
    thp, omp = rev_step(th, om, u)
    th2, om2 = fwd_step(thp, omp, u)
    inside = om2.abs() < OMEGA_MAX - 1e-3   # clamp-free region
    err = ((th2 - th).abs() + (om2 - om).abs())[inside].max()
    assert err < 1e-4, f"reverse != inverse of forward: {err}"
    # noising reaches the hanging region (coverage sanity)
    th, om, _, _ = noising_batch(200000, 200, device)
    frac_low = ((wrap(th).abs() > 2.6)).float().mean().item()
    assert frac_low > 0.05, f"noising never reaches hanging region ({frac_low:.2%})"
    print(f"pendulum tests passed: exact reverse dynamics (err {err:.1e}), "
          f"{frac_low:.1%} of noised states near hanging")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--method", choices=["denoise", "denoise-disc", "value"])
    ap.add_argument("--iters", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.test:
        test()
    else:
        train(args.method, args.iters, args.out)
