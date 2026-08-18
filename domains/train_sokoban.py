"""Sokoban trainers (denoise vs DAVI), matched setup. Non-invertible forward
dynamics; noising via pulls happens inside instance generation.

  python domains/train_sokoban.py --method denoise --iters 80000 --out runs/soko_diff
  python domains/train_sokoban.py --method davi    --iters 80000 --out runs/soko_davi
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from domains.sokoban import SokobanEnv
from model import PolicyNet, ValueNet


def solved_rendered(st):
    return (st != 2).all(dim=1)


@torch.no_grad()
def rollout(env, net, method, walls, goals, boxes, agent, max_steps):
    B = walls.shape[0]
    dev = walls.device
    boxes, agent = boxes.clone(), agent.clone()
    done = env.is_solved(walls, goals, boxes, agent)
    actions = torch.full((B, max_steps), -1, dtype=torch.long, device=dev)
    for t in range(max_steps):
        if done.all():
            break
        idx = (~done).nonzero(as_tuple=True)[0]
        w, g, b, a_ = walls[idx], goals[idx], boxes[idx], agent[idx]
        legal = env.legal_forward(w, g, b, a_)
        st = env.render(w, g, b, a_)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if method == "denoise":
                act = net(st).float().masked_fill(~legal, -1e9).argmax(1)
            else:
                nb = env.neighbors_forward(w, g, b, a_).reshape(-1, env.S)
                v = net(nb).float().view(-1, env.M)
                v = torch.where(solved_rendered(nb).view(-1, env.M),
                                torch.zeros_like(v), v.clamp_min(0))
                act = v.masked_fill(~legal, 1e9).argmin(1)
        nb_, na_ = env.step_forward(w, g, b, a_, act)
        boxes[idx], agent[idx] = nb_, na_
        actions[idx, t] = act
        done = done | env.is_solved(walls, goals, boxes, agent)
    return env.is_solved(walls, goals, boxes, agent), actions


def train(method, iters, out_dir, n=8, n_boxes=3, K=40, batch=8192, lr=1e-3,
          device="cuda", h1=2048, h2=1024, blocks=3, resume=None):
    os.makedirs(out_dir, exist_ok=True)
    env = SokobanEnv(n, n_boxes, device=device)
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
    print(f"{method} sokoban {n}x{n}/{n_boxes} boxes: "
          f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M params", flush=True)

    POOL = 8
    pool, pool_ptr = None, 0

    def refill():
        nonlocal pool, pool_ptr
        NB = batch * POOL
        depths = torch.randint(1, K + 1, (NB,), device=device)
        w, g, b, a_, lab = env.instances_and_scramble(NB, depths)
        keep = lab >= 0
        pool = (w[keep], g[keep], b[keep], a_[keep], lab[keep])
        pool_ptr = 0

    ema, stale, tgt_updates = None, 0, 0
    t_last, it_last = time.time(), start_iter
    net.train()
    for it in range(start_iter + 1, iters + 1):
        if it == int(iters * 0.8) + 1:
            for gg in opt.param_groups:
                gg["lr"] = lr * 0.1
        if pool is None or pool_ptr + batch > pool[0].shape[0]:
            refill()
        sl = slice(pool_ptr, pool_ptr + batch)
        pool_ptr += batch
        w, g, b, a_, lab = (x[sl] for x in pool)
        st = env.render(w, g, b, a_)
        if method == "denoise":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net_fwd(st).float(), lab)
        else:
            Bs = st.shape[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                nb = env.neighbors_forward(w, g, b, a_).reshape(-1, env.S)
                v = tgt_fwd(nb).float().view(Bs, env.M).clamp_min(0)
                v = torch.where(solved_rendered(nb).view(Bs, env.M),
                                torch.zeros_like(v), v)
                v = v.masked_fill(~env.legal_forward(w, g, b, a_), 1e9)
                y = 1.0 + v.min(dim=1).values
                y = torch.where(env.is_solved(w, g, b, a_), torch.zeros_like(y), y)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(net_fwd(st).float(), y)
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
            for d in (5, 12, 25, 40):
                w, g, b, a_, _ = env.instances_and_scramble(512, d)
                solved, _ = rollout(env, net, method, w, g, b, a_,
                                    max_steps=4 * d + 20)
                rates[str(d)] = round(solved.float().mean().item(), 4)
            net.train()
            mfh.write(json.dumps({"iter": it, "probe": rates}) + "\n")
            print("PROBE " + json.dumps({"iter": it, "probe": rates}), flush=True)
        if it % 2000 == 0 or it == iters:
            ck = {"iter": it, "net": net.state_dict(), "opt": opt.state_dict(),
                  "cfg": {"n": n, "n_boxes": n_boxes, "method": method,
                          "h1": h1, "h2": h2, "blocks": blocks, "K": K}}
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
