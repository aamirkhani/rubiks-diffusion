"""2048: train the denoiser on deterministic reverse play, evaluate in the
REAL stochastic game against random and greedy-merge baselines.

  python domains/train_2048.py --iters 40000 --out runs/g2048_diff
  python domains/train_2048.py --eval --ckpt runs/g2048_diff/ckpt_latest.pt
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from domains.game2048 import (S, noising_batch, legal_mask, play_step, spawn,
                              slide, max_tile)
from model import PolicyNet

VOCAB = 12  # exponents 0..11 (up to 2048 tile)


def train(iters, out_dir, K=40, batch=8192, lr=1e-3, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    net = PolicyNet(S, 4, 2048, 1024, 3, vocab=VOCAB).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    torch.backends.cuda.matmul.allow_tf32 = True
    net_fwd = torch.compile(net)
    mfh = open(os.path.join(out_dir, "metrics.jsonl"), "a", buffering=1)
    print(f"2048 denoiser: {sum(p.numel() for p in net.parameters())/1e6:.1f}M "
          f"params", flush=True)
    POOL = 4
    pool, pool_ptr = None, 0

    def refill():
        nonlocal pool, pool_ptr
        boards, labels = noising_batch(batch * POOL, K, device=device)
        pool, pool_ptr = (boards, labels), 0

    ema = None
    t_last, it_last = time.time(), 0
    for it in range(1, iters + 1):
        if it == int(iters * 0.8) + 1:
            for g in opt.param_groups:
                g["lr"] = lr * 0.1
        if pool is None or pool_ptr + batch > pool[0].shape[0]:
            refill()
        sl = slice(pool_ptr, pool_ptr + batch)
        pool_ptr += batch
        boards, labels = pool[0][sl], pool[1][sl]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(net_fwd(boards).float(), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        l = loss.item()
        ema = l if ema is None else 0.95 * ema + 0.05 * l
        if it % 200 == 0:
            now = time.time()
            ips = (it - it_last) / max(now - t_last, 1e-9)
            t_last, it_last = now, it
            mfh.write(json.dumps({"iter": it, "loss": round(l, 4),
                                  "ema": round(ema, 4),
                                  "ips": round(ips, 1)}) + "\n")
        if it % 5000 == 0:
            res = evaluate(net, device=device, B=512, seed=it)
            mfh.write(json.dumps({"iter": it, "probe": res}) + "\n")
            print("PROBE " + json.dumps(res), flush=True)
        if it % 5000 == 0 or it == iters:
            torch.save({"iter": it, "net": net.state_dict(),
                        "cfg": {"K": K}}, os.path.join(out_dir, "ckpt_latest.pt"))
    print("training complete", flush=True)


@torch.no_grad()
def run_policy(policy_fn, B, steps, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    board = torch.zeros(B, S, dtype=torch.int8, device=device)
    board = spawn(spawn(board, g), g)
    alive = torch.ones(B, dtype=torch.bool, device=device)
    for t in range(steps):
        legal = legal_mask(board)
        alive = alive & legal.any(1)
        if not alive.any():
            break
        d = policy_fn(board, legal)
        nb, _ = play_step(board, d, generator=g)
        board = torch.where(alive.unsqueeze(1), nb, board)
    mt = max_tile(board)
    return {
        "mean_max_tile": round((2.0 ** mt.float()).mean().item(), 1),
        "reach_128": round((mt >= 7).float().mean().item(), 4),
        "reach_256": round((mt >= 8).float().mean().item(), 4),
        "reach_512": round((mt >= 9).float().mean().item(), 4),
    }


@torch.no_grad()
def evaluate(net, device="cuda", B=2048, steps=400, seed=0):
    net.eval()

    def pol_net(board, legal):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(board).float()
        return lg.masked_fill(~legal, -1e9).argmax(1)

    def pol_random(board, legal):
        u = torch.rand(board.shape[0], 4, device=device).clamp_min(1e-9)
        return (-torch.log(-torch.log(u))).masked_fill(~legal, -1e9).argmax(1)

    def pol_greedy_merge(board, legal):
        gains = []
        for d in range(4):
            _, moved, gain = slide(board, d)
            gains.append(torch.where(moved, gain, torch.full_like(gain, -1)))
        return torch.stack(gains, 1).float().masked_fill(~legal, -1e9).argmax(1)

    out = {"denoiser": run_policy(pol_net, B, steps, device, seed),
           "random": run_policy(pol_random, B, steps, device, seed + 1),
           "greedy_merge": run_policy(pol_greedy_merge, B, steps, device, seed + 2)}
    net.train()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ckpt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.eval:
        ck = torch.load(args.ckpt, map_location="cuda", weights_only=True)
        net = PolicyNet(S, 4, 2048, 1024, 3, vocab=VOCAB).cuda()
        net.load_state_dict(ck["net"])
        res = evaluate(net, B=2048, seed=999)
        res_out = {"domain": "2048 (stochastic spawns; goal = tile set)",
                   "iter": ck["iter"], **res}
        print(json.dumps(res_out, indent=2))
        os.makedirs("paper2_data", exist_ok=True)
        with open("paper2_data/domain5_2048.json", "w") as f:
            json.dump(res_out, f, indent=2)
        print("saved paper2_data/domain5_2048.json")
    else:
        train(args.iters, args.out)
