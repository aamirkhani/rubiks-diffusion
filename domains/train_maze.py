"""Conditional trainers for the maze domain (denoise vs DAVI), matched setup.

  python domains/train_maze.py --method denoise --iters 60000 --out runs/maze_diff
  python domains/train_maze.py --method davi    --iters 60000 --out runs/maze_davi
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from domains.maze import MazeEnv
from model import PolicyNet, ValueNet


@torch.no_grad()
def rollout(env, net, method, states, goal, max_steps):
    B = states.shape[0]
    dev = states.device
    states = states.clone()
    done = env.is_solved(states)
    actions = torch.full((B, max_steps), -1, dtype=torch.long, device=dev)
    prev = torch.full((B,), -1, dtype=torch.long, device=dev)
    for t in range(max_steps):
        if done.all():
            break
        idx = (~done).nonzero(as_tuple=True)[0]
        sub, gl = states[idx], goal[idx]
        legal = env.legal_mask(sub)
        p = prev[idx]
        has = p >= 0
        legal[has, env.inverse_action[p[has]]] = False
        none = ~legal.any(1)
        if none.any():
            legal[none] = env.legal_mask(sub[none])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if method == "denoise":
                a = net(sub).float().masked_fill(~legal, -1e9).argmax(1)
            else:
                nb = env.neighbors(sub, gl).reshape(-1, env.S)
                v = net(nb).float().view(-1, env.M)
                v = torch.where(env.is_solved(nb).view(-1, env.M),
                                torch.zeros_like(v), v.clamp_min(0))
                a = v.masked_fill(~legal, 1e9).argmin(1)
        states[idx] = env.step(sub, a, gl)
        actions[idx, t] = a
        prev[idx] = a
        done = done | env.is_solved(states)
    return env.is_solved(states), actions


def train(method, iters, out_dir, n=15, K=60, batch=8192, lr=1e-3,
          device="cuda", h1=2048, h2=1024, blocks=3, resume=None):
    os.makedirs(out_dir, exist_ok=True)
    env = MazeEnv(n, device=device)
    if method == "denoise":
        net = PolicyNet(env.S, env.M, h1, h2, blocks, vocab=env.vocab).to(device)
    else:
        net = ValueNet(env.S, h1, h2, blocks, vocab=env.vocab).to(device)
        target = ValueNet(env.S, h1, h2, blocks, vocab=env.vocab).to(device)
        target.load_state_dict(net.state_dict())
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    start_iter = 0
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=True)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"]
        if method == "davi":
            target.load_state_dict(ck["target"])
    torch.backends.cuda.matmul.allow_tf32 = True
    net_fwd = torch.compile(net)
    tgt_fwd = torch.compile(target) if method == "davi" else None

    mfh = open(os.path.join(out_dir, "metrics.jsonl"), "a", buffering=1)
    print(f"{method} maze {n}x{n}: "
          f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M params", flush=True)

    POOL = 16
    pool = None; pool_ptr = 0

    def refill():
        nonlocal pool, pool_ptr
        NB = batch * POOL
        walls, goal = env.new_instances(NB)
        depths = torch.randint(1, K + 1, (NB,), device=device)
        if method == "denoise":
            sts, acts = env.scramble(walls, goal, depths, return_actions=True)
            # label = inverse of LAST actual move (some steps may be stuck no-ops)
            last = torch.full((NB,), -1, dtype=torch.long, device=device)
            for t in range(acts.shape[1]):
                last = torch.where(acts[:, t] >= 0, acts[:, t], last)
            keep = last >= 0
            pool = (sts[keep], goal[keep], env.inverse_action[last[keep]])
        else:
            sts = env.scramble(walls, goal, depths)
            pool = (sts, goal, None)
        pool_ptr = 0

    ema = None; stale = 0; tgt_updates = 0
    t_last, it_last = time.time(), start_iter
    net.train()
    for it in range(start_iter + 1, iters + 1):
        if it == int(iters * 0.8) + 1:
            for g in opt.param_groups:
                g["lr"] = lr * 0.1
        if pool is None or pool_ptr + batch > pool[0].shape[0]:
            refill()
        sl = slice(pool_ptr, pool_ptr + batch)
        pool_ptr += batch
        states, goal = pool[0][sl], pool[1][sl]
        if method == "denoise":
            labels = pool[2][sl]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net_fwd(states).float(), labels)
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                nb = env.neighbors(states, goal).reshape(-1, env.S)
                v = tgt_fwd(nb).float().view(states.shape[0], env.M).clamp_min(0)
                v = torch.where(env.is_solved(nb).view(states.shape[0], env.M),
                                torch.zeros_like(v), v)
                v = v.masked_fill(~env.legal_mask(states), 1e9)
                y = 1.0 + v.min(dim=1).values
                y = torch.where(env.is_solved(states), torch.zeros_like(y), y)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(net_fwd(states).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        l = loss.item()
        ema = l if ema is None else 0.95 * ema + 0.05 * l
        if method == "davi" and it % 100 == 0:
            stale += 100
            if ema < 0.05 or stale >= 4000:
                target.load_state_dict(net.state_dict()); tgt_updates += 1; stale = 0
        if it % 100 == 0:
            now = time.time()
            ips = (it - it_last) / max(now - t_last, 1e-9)
            t_last, it_last = now, it
            mfh.write(json.dumps({"iter": it, "loss": round(l, 4),
                                  "ema": round(ema, 4), "ips": round(ips, 2),
                                  "tgt_updates": tgt_updates}) + "\n")
        if it % 5000 == 0:
            net.eval()
            rates = {}
            for d in (10, 25, 40, 60):
                walls, goal = env.new_instances(512)
                st = env.scramble(walls, goal, d)
                solved, _ = rollout(env, net, method, st, goal, max_steps=3 * d + 20)
                rates[str(d)] = round(solved.float().mean().item(), 4)
            net.train()
            mfh.write(json.dumps({"iter": it, "probe": rates}) + "\n")
            print("PROBE " + json.dumps({"iter": it, "probe": rates}), flush=True)
        if it % 2000 == 0 or it == iters:
            ck = {"iter": it, "net": net.state_dict(), "opt": opt.state_dict(),
                  "cfg": {"n": n, "method": method, "h1": h1, "h2": h2,
                          "blocks": blocks, "K": K}}
            if method == "davi":
                ck["target"] = target.state_dict()
            torch.save(ck, os.path.join(out_dir, "ckpt_latest.pt"))
    print("training complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["denoise", "davi"], required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume")
    args = ap.parse_args()
    train(args.method, args.iters, args.out, resume=args.resume)
