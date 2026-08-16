"""DAVI (Deep Approximate Value Iteration) trainer — DeepCubeA-style.

Targets: y(s) = 0 if solved else 1 + min_a V_target(step(s, a)), with
V_target(solved neighbor) = 0. Target net syncs to the online net whenever
the recent loss falls below a threshold (or after a staleness cap).
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from cube_env import CubeEnv
from model import ValueNet
from solve import greedy_solve

PRESETS = {
    "2x2": dict(n=2, iters=20000, batch=10000, K=14, h1=1024, h2=512, blocks=2,
                lr=1e-3, loss_thresh=0.05, check_every=100, max_stale=2000,
                probe_depths=[3, 5, 8, 11, 14], probe_every=1000, ckpt_every=2000),
    "3x3": dict(n=3, iters=250000, batch=10000, K=30, h1=5000, h2=1000, blocks=4,
                lr=1e-3, loss_thresh=0.05, check_every=100, max_stale=5000,
                probe_depths=[5, 10, 14, 18, 22, 26, 30], probe_every=5000, ckpt_every=2000,
                compile=True),
}


def probe(env, net, depths, n_each=512):
    """Greedy solve rate by scramble depth."""
    was_training = net.training
    net.eval()
    rates = {}
    for d in depths:
        st = env.scramble(n_each, d)
        max_steps = min(2 * d + 12, 60)
        solved, _, _ = greedy_solve(env, net, st, max_steps)
        rates[str(d)] = round(solved.float().mean().item(), 4)
    if was_training:
        net.train()
    return rates


def train(cfg, out_dir, resume=None, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    env = CubeEnv(cfg["n"], device)
    net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"]).to(device)
    target = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    start_iter, tgt_updates = 0, 0

    if resume:
        ck = torch.load(resume, map_location=device, weights_only=True)
        net.load_state_dict(ck["net"])
        target.load_state_dict(ck["target"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"]
        tgt_updates = ck.get("tgt_updates", 0)
        print(f"resumed from {resume} at iter {start_iter}")
    else:
        target.load_state_dict(net.state_dict())

    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # compiled wrappers share parameters with the eager modules; used only for
    # the two fixed-shape training passes (probes/eval keep the eager net)
    if cfg.get("compile"):
        torch.backends.cuda.matmul.allow_tf32 = True
        net_fwd = torch.compile(net)
        tgt_fwd = torch.compile(target)
        print("torch.compile enabled", flush=True)
    else:
        net_fwd, tgt_fwd = net, target

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    mfh = open(metrics_path, "a", buffering=1)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"model params: {nparams/1e6:.1f}M  batch {cfg['batch']}  K={cfg['K']}  device={device}", flush=True)

    B, M, S, K = cfg["batch"], env.M, env.S, cfg["K"]
    ema_loss, stale = None, 0
    t_last, it_last = time.time(), start_iter
    net.train()

    for it in range(start_iter + 1, cfg["iters"] + 1):
        # lr decay: x0.1 for the final 20%
        if it == int(cfg["iters"] * 0.8) + 1:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * 0.1

        depths = torch.randint(1, K + 1, (B,), device=device)
        states = env.scramble(B, depths)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            nb = env.neighbors(states).reshape(B * M, S)
            v_next = tgt_fwd(nb).float().view(B, M).clamp_min(0)
            solved_nb = env.is_solved(nb).view(B, M)
            v_next = torch.where(solved_nb, torch.zeros_like(v_next), v_next)
            y = 1.0 + v_next.min(dim=1).values
            y = torch.where(env.is_solved(states), torch.zeros_like(y), y)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net_fwd(states)
            loss = F.mse_loss(pred.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        l = loss.item()
        ema_loss = l if ema_loss is None else 0.95 * ema_loss + 0.05 * l

        if it % cfg["check_every"] == 0:
            stale += cfg["check_every"]
            if ema_loss < cfg["loss_thresh"] or stale >= cfg["max_stale"]:
                target.load_state_dict(net.state_dict())
                tgt_updates += 1
                stale = 0

        if it % 50 == 0:
            now = time.time()
            ips = (it - it_last) / max(now - t_last, 1e-9)
            t_last, it_last = now, it
            rec = {"iter": it, "loss": round(l, 4), "ema": round(ema_loss, 4),
                   "tgt_updates": tgt_updates, "ips": round(ips, 2),
                   "lr": opt.param_groups[0]["lr"]}
            mfh.write(json.dumps(rec) + "\n")
            if it % 500 == 0:
                print(json.dumps(rec), flush=True)

        if it % cfg["probe_every"] == 0:
            rates = probe(env, net, cfg["probe_depths"])
            rec = {"iter": it, "probe": rates, "tgt_updates": tgt_updates}
            mfh.write(json.dumps(rec) + "\n")
            print("PROBE " + json.dumps(rec), flush=True)
            net.train()

        if it % cfg["ckpt_every"] == 0 or it == cfg["iters"]:
            ck = {"iter": it, "net": net.state_dict(), "target": target.state_dict(),
                  "opt": opt.state_dict(), "cfg": cfg, "tgt_updates": tgt_updates}
            torch.save(ck, os.path.join(out_dir, "ckpt_latest.pt"))
            if it % 20000 == 0:
                torch.save(ck, os.path.join(out_dir, f"ckpt_{it}.pt"))

    print("training complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--K", type=int, help="override max scramble walk depth")
    ap.add_argument("--resume")
    args = ap.parse_args()
    cfg = dict(PRESETS[args.preset])
    if args.iters:
        cfg["iters"] = args.iters
    if args.K:
        cfg["K"] = args.K
    train(cfg, args.out, resume=args.resume)
