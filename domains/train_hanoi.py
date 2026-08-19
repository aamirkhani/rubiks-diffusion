"""Hanoi trainers (denoise vs DAVI). Tiny nets, tiny states, brutal horizon.

  python domains/train_hanoi.py --method denoise --iters 30000 --out runs/hanoi_diff
  python domains/train_hanoi.py --method davi    --iters 30000 --out runs/hanoi_davi
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from domains.hanoi import HanoiEnv
from domains.train_slide import rollout, verify          # env-API compatible
from model import PolicyNet, ValueNet


def train(method, iters, out_dir, n=10, K=1200, batch=8192, lr=1e-3,
          device="cuda", h1=1024, h2=512, blocks=2):
    os.makedirs(out_dir, exist_ok=True)
    env = HanoiEnv(n, device)
    if method == "denoise":
        net = PolicyNet(env.S, env.M, h1, h2, blocks, vocab=env.vocab).to(device)
    else:
        net = ValueNet(env.S, h1, h2, blocks, vocab=env.vocab).to(device)
        target = ValueNet(env.S, h1, h2, blocks, vocab=env.vocab).to(device)
        target.load_state_dict(net.state_dict())
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
        tgt_fwd = target
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mfh = open(os.path.join(out_dir, "metrics.jsonl"), "a", buffering=1)
    print(f"{method} hanoi n={n}: "
          f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M params, K={K}",
          flush=True)

    POOL = 16
    pool_states, pool_labels, pool_ptr = None, None, 0

    def refill():
        nonlocal pool_states, pool_labels, pool_ptr
        NB = batch * POOL
        depths = torch.randint(1, K + 1, (NB,), device=device)
        if method == "denoise":
            sts, acts = env.scramble(NB, depths, return_actions=True)
            last = acts[torch.arange(NB, device=device), depths - 1]
            keep = last >= 0
            pool_states = sts[keep]
            pool_labels = env.inverse_action[last[keep]]
        else:
            pool_states = env.scramble(NB, depths)
            pool_labels = None
        pool_ptr = 0

    ema, stale, tgt_updates = None, 0, 0
    net.train()
    for it in range(1, iters + 1):
        if it == int(iters * 0.8) + 1:
            for g in opt.param_groups:
                g["lr"] = lr * 0.1
        if pool_states is None or pool_ptr + batch > pool_states.shape[0]:
            refill()
        sl = slice(pool_ptr, pool_ptr + batch)
        pool_ptr += batch
        states = pool_states[sl]
        if method == "denoise":
            loss = F.cross_entropy(net(states).float(), pool_labels[sl])
        else:
            with torch.no_grad():
                nb = env.neighbors(states).reshape(-1, env.S)
                v = tgt_fwd(nb).float().view(batch, env.M).clamp_min(0)
                v = torch.where(env.is_solved(nb).view(batch, env.M),
                                torch.zeros_like(v), v)
                v = v.masked_fill(~env.legal_mask(states), 1e9)
                y = 1.0 + v.min(dim=1).values
                y = torch.where(env.is_solved(states), torch.zeros_like(y), y)
            loss = F.mse_loss(net(states).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        l = loss.item()
        ema = l if ema is None else 0.95 * ema + 0.05 * l
        if method == "davi" and it % 100 == 0:
            stale += 100
            if ema < 0.05 or stale >= 4000:
                target.load_state_dict(net.state_dict()); tgt_updates += 1; stale = 0
        if it % 500 == 0:
            mfh.write(json.dumps({"iter": it, "ema": round(ema, 4),
                                  "tgt_updates": tgt_updates}) + "\n")
        if it % 5000 == 0:
            net.eval()
            rates = {}
            for d in (100, 300, 600, 1200):
                st = env.scramble(512, d)
                solved, _ = rollout(env, net, method, st, max_steps=2500)
                rates[str(d)] = round(solved.float().mean().item(), 4)
            net.train()
            mfh.write(json.dumps({"iter": it, "probe": rates}) + "\n")
            print("PROBE " + json.dumps({"iter": it, "probe": rates}), flush=True)
        if it % 5000 == 0 or it == iters:
            torch.save({"iter": it, "net": net.state_dict(),
                        "cfg": {"n": n, "method": method, "h1": h1, "h2": h2,
                                "blocks": blocks, "K": K}},
                       os.path.join(out_dir, "ckpt_latest.pt"))
    print("training complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["denoise", "davi"], required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    train(args.method, args.iters, args.out)
