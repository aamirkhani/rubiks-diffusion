"""Trainers for Lights Out and Peg Solitaire (denoise vs DAVI).

  python domains/train_misc.py --domain lightsout --method denoise --iters 20000 --out runs/lo_diff
  python domains/train_misc.py --domain pegs      --method davi    --iters 30000 --out runs/peg_davi
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from model import PolicyNet, ValueNet


def get_env(domain, device):
    if domain == "lightsout":
        from domains.lightsout import LightsOutEnv
        return LightsOutEnv(device)
    from domains.pegsolitaire import PegEnv
    return PegEnv(device)


@torch.no_grad()
def all_successors(env, states):
    """[B, M, S] successor states via env.step per action (chunked over M)."""
    B = states.shape[0]
    outs = []
    for a in range(env.M):
        aa = torch.full((B,), a, dtype=torch.long, device=states.device)
        outs.append(env.step(states, aa))
    return torch.stack(outs, 1)


@torch.no_grad()
def rollout(env, net, method, states, max_steps, forbid_repeat=False):
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
        if forbid_repeat:
            p = prev[idx]
            has = p >= 0
            legal[has, p[has]] = False
        dead = ~legal.any(1)
        if dead.any():
            legal[dead, 0] = True   # no legal move: dummy (will not solve)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if method == "denoise":
                a = net(sub).float().masked_fill(~legal, -1e9).argmax(1)
            else:
                nb = all_successors(env, sub)
                v = net(nb.reshape(-1, env.S)).float().view(-1, env.M)
                v = torch.where(env.is_solved(nb.reshape(-1, env.S))
                                .view(-1, env.M), torch.zeros_like(v),
                                v.clamp_min(0))
                a = v.masked_fill(~legal, 1e9).argmin(1)
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


def train(domain, method, iters, out_dir, batch=None, lr=1e-3, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    env = get_env(domain, device)
    K = 25 if domain == "lightsout" else 31
    batch = batch or (8192 if domain == "lightsout" else 4096)
    h1, h2, blocks = 2048, 1024, 3
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
    mfh = open(os.path.join(out_dir, "metrics.jsonl"), "a", buffering=1)
    print(f"{method} {domain}: "
          f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M params",
          flush=True)

    POOL = 16
    ps, pl, pp = None, None, 0

    def refill():
        nonlocal ps, pl, pp
        NB = batch * POOL
        depths = torch.randint(1, K + 1, (NB,), device=device)
        if domain == "lightsout":
            sts, acts = env.scramble(NB, depths, return_actions=True)
            # ANY pressed cell that is still 'on-set' works; use the last one
            last = torch.full((NB,), -1, dtype=torch.long, device=device)
            for t in range(acts.shape[1]):
                last = torch.where(acts[:, t] >= 0, acts[:, t], last)
        else:
            sts, last = env.scramble(NB, depths, return_actions=True)
        keep = last >= 0
        ps, pl, pp = sts[keep], last[keep], 0

    ema, stale, tgt_updates = None, 0, 0
    net.train()
    for it in range(1, iters + 1):
        if it == int(iters * 0.8) + 1:
            for g in opt.param_groups:
                g["lr"] = lr * 0.1
        if ps is None or pp + batch > ps.shape[0]:
            refill()
        sl = slice(pp, pp + batch)
        pp += batch
        states, labels = ps[sl], pl[sl]
        if method == "denoise":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net(states).float(), labels)
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                nb = all_successors(env, states)
                v = target(nb.reshape(-1, env.S)).float().view(batch, env.M)
                v = torch.where(env.is_solved(nb.reshape(-1, env.S))
                                .view(batch, env.M), torch.zeros_like(v),
                                v.clamp_min(0))
                v = v.masked_fill(~env.legal_mask(states), 1e9)
                y = 1.0 + v.min(1).values
                y = torch.where(env.is_solved(states), torch.zeros_like(y), y)
                y = y.clamp_max(60.0)     # dead states (no legal move): cap
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
            for d in ((5, 10, 15, 25) if domain == "lightsout"
                      else (5, 10, 15, 25)):
                st = env.scramble(512, d)
                if isinstance(st, tuple):
                    st = st[0]
                solved, _ = rollout(env, net, method, st, 3 * d + 15,
                                    forbid_repeat=(domain == "lightsout"))
                rates[str(d)] = round(solved.float().mean().item(), 4)
            net.train()
            mfh.write(json.dumps({"iter": it, "probe": rates}) + "\n")
            print("PROBE " + json.dumps({"iter": it, "probe": rates}), flush=True)
        if it % 5000 == 0 or it == iters:
            torch.save({"iter": it, "net": net.state_dict(),
                        "cfg": {"domain": domain, "method": method, "h1": h1,
                                "h2": h2, "blocks": blocks, "K": K}},
                       os.path.join(out_dir, "ckpt_latest.pt"))
    print("training complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["lightsout", "pegs"], required=True)
    ap.add_argument("--method", choices=["denoise", "davi"], required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    train(args.domain, args.method, args.iters, args.out)
