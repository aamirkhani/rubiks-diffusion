"""Matched-compute trainers for the sliding-tile domain: DAVI (value) and the
diffusion-style denoiser, both legality-aware. Mirrors the cube protocol:
same backbone, batch, optimizer, scramble distribution.

  python domains/train_slide.py --method denoise --n 5 --iters 200000 --out runs/slide5_diff
  python domains/train_slide.py --method davi    --n 5 --iters 100000 --out runs/slide5_davi
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from domains.slide import SlideEnv
from model import PolicyNet, ValueNet


@torch.no_grad()
def greedy_probe(env, net, method, depths, n_each=512, max_steps=None):
    net.eval()
    rates = {}
    for d in depths:
        st = env.scramble(n_each, d)
        ms = max_steps or min(4 * d + 20, 400)
        solved = rollout(env, net, method, st, ms)[0]
        rates[str(d)] = round(solved.float().mean().item(), 4)
    net.train()
    return rates


@torch.no_grad()
def rollout(env, net, method, states, max_steps):
    """Greedy rollout for either method, legality- and backtrack-masked."""
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
        sub = states[idx]
        legal = env.legal_mask(sub)
        p = prev[idx]
        has = p >= 0
        legal[has, env.inverse_action[p[has]]] = False
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if method == "denoise":
                score = net(sub).float()
                score = score.masked_fill(~legal, -1e9)
                a = score.argmax(1)
            else:  # davi: pick legal neighbor with smallest value (0 if solved)
                nb = env.neighbors(sub).reshape(-1, env.S)
                v = net(nb).float().view(-1, env.M)
                v = torch.where(env.is_solved(nb).view(-1, env.M),
                                torch.zeros_like(v), v.clamp_min(0))
                v = v.masked_fill(~legal, 1e9)
                a = v.argmin(1)
        states[idx] = env.step(sub, a)
        actions[idx, t] = a
        prev[idx] = a
        done = done | env.is_solved(states)
    return env.is_solved(states), actions


@torch.no_grad()
def verify(env, start, actions):
    s = start.clone()
    for t in range(actions.shape[1]):
        a = actions[:, t]
        valid = a >= 0
        if not valid.any():
            break
        nxt = env.step(s, torch.where(valid, a, torch.zeros_like(a)))
        s = torch.where(valid.unsqueeze(1), nxt, s)
    return env.is_solved(s)


def train(method, n, iters, out_dir, K, batch=10000, lr=1e-3, device="cuda",
          compile_net=True, h1=5000, h2=1000, blocks=4, resume=None):
    os.makedirs(out_dir, exist_ok=True)
    env = SlideEnv(n, device)
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
        print(f"resumed at {start_iter}", flush=True)

    if compile_net:
        torch.backends.cuda.matmul.allow_tf32 = True
        net_fwd = torch.compile(net)
        tgt_fwd = torch.compile(target) if method == "davi" else None
        print("torch.compile on", flush=True)
    else:
        net_fwd = net
        tgt_fwd = target if method == "davi" else None

    mfh = open(os.path.join(out_dir, "metrics.jsonl"), "a", buffering=1)
    npar = sum(p.numel() for p in net.parameters())
    print(f"{method} slide {n}x{n}: {npar/1e6:.1f}M params, K={K}, batch {batch}",
          flush=True)
    probe_depths = [10, 30, 60, 120, 200] if n == 5 else [5, 10, 15, 22, 31]
    ema, stale, tgt_updates = None, 0, 0
    t_last, it_last = time.time(), start_iter
    net.train()

    # scramble pool: amortize the K-step walk over POOL training iterations
    POOL = 64
    pool_states, pool_labels, pool_ptr = None, None, 0

    def refill_pool():
        nonlocal pool_states, pool_labels, pool_ptr
        NB = batch * POOL
        depths = torch.randint(1, K + 1, (NB,), device=device)
        if method == "denoise":
            sts, acts = env.scramble(NB, depths, return_actions=True)
            last = acts[torch.arange(NB, device=device), depths - 1]
            pool_labels = env.inverse_action[last]
        else:
            sts = env.scramble(NB, depths)
            pool_labels = None
        pool_states = sts
        pool_ptr = 0

    for it in range(start_iter + 1, iters + 1):
        if it == int(iters * 0.8) + 1:
            for g in opt.param_groups:
                g["lr"] = lr * 0.1
        if pool_states is None or pool_ptr + batch > pool_states.shape[0]:
            refill_pool()
        sl = slice(pool_ptr, pool_ptr + batch)
        pool_ptr += batch
        if method == "denoise":
            states, labels = pool_states[sl], pool_labels[sl]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net_fwd(states).float(), labels)
        else:
            states = pool_states[sl]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                nb = env.neighbors(states).reshape(-1, env.S)
                v = tgt_fwd(nb).float().view(batch, env.M).clamp_min(0)
                v = torch.where(env.is_solved(nb).view(batch, env.M),
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
            if ema < 0.05 or stale >= 5000:
                target.load_state_dict(net.state_dict())
                tgt_updates += 1
                stale = 0
        if it % 100 == 0:
            now = time.time()
            ips = (it - it_last) / max(now - t_last, 1e-9)
            t_last, it_last = now, it
            mfh.write(json.dumps({"iter": it, "loss": round(l, 4),
                                  "ema": round(ema, 4), "ips": round(ips, 2),
                                  "tgt_updates": tgt_updates}) + "\n")
            if it % 2000 == 0:
                print(json.dumps({"iter": it, "ema": round(ema, 4),
                                  "ips": round(ips, 2)}), flush=True)
        if it % 10000 == 0:
            rates = greedy_probe(env, net, method, probe_depths)
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
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--K", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h1", type=int, default=5000)
    ap.add_argument("--h2", type=int, default=1000)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--resume")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    K = args.K or (300 if args.n == 5 else 31 if args.n == 3 else 120)
    train(args.method, args.n, args.iters, args.out, K,
          h1=args.h1, h2=args.h2, blocks=args.blocks,
          compile_net=not args.no_compile, resume=args.resume)
