"""Diffusion-style trainer: the forward process scrambles the solved cube with
random moves (noise); the net learns the reverse process by predicting the
move that undoes the LAST scramble step (noise prediction). Depth schedule is
linear (uniform 1..K). No target net, no bootstrapping — pure supervised.
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from cube_env import CubeEnv
from model import PolicyNet
from solve_policy import policy_greedy

PRESETS = {
    "2x2": dict(n=2, iters=40000, batch=10000, K=18, h1=1024, h2=512, blocks=2,
                lr=1e-3, probe_depths=[3, 5, 8, 11, 14], probe_every=2000,
                ckpt_every=2000),
    "3x3": dict(n=3, iters=500000, batch=10000, K=30, h1=5000, h2=1000, blocks=4,
                lr=1e-3, probe_depths=[5, 10, 14, 18, 22, 26, 30], probe_every=10000,
                ckpt_every=2000, compile=True),
}


def probe(env, net, depths, n_each=512):
    was_training = net.training
    net.eval()
    rates = {}
    for d in depths:
        st = env.scramble(n_each, d)
        solved, _, _ = policy_greedy(env, net, st, max_steps=min(2 * d + 12, 60))
        rates[str(d)] = round(solved.float().mean().item(), 4)
    if was_training:
        net.train()
    return rates


def train(cfg, out_dir, resume=None, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    env = CubeEnv(cfg["n"], device)
    net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"],
                    t_dim=cfg.get("t_dim", 0)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    start_iter = 0
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=True)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"]
        print(f"resumed from {resume} at iter {start_iter}", flush=True)

    if cfg.get("compile"):
        torch.backends.cuda.matmul.allow_tf32 = True
        net_fwd = torch.compile(net)
        print("torch.compile enabled", flush=True)
    else:
        net_fwd = net

    mfh = open(os.path.join(out_dir, "metrics.jsonl"), "a", buffering=1)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"model params: {nparams/1e6:.1f}M  batch {cfg['batch']}  K={cfg['K']} "
          f"(linear schedule)  device={device}", flush=True)

    B, K = cfg["batch"], cfg["K"]
    ema_loss, ema_acc = None, None
    t_last, it_last = time.time(), start_iter
    net.train()

    for it in range(start_iter + 1, cfg["iters"] + 1):
        if it == int(cfg["iters"] * 0.8) + 1:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * 0.1

        sched = cfg.get("schedule", "uniform")
        if sched == "uniform":
            depths = torch.randint(1, K + 1, (B,), device=device)
        else:
            u = torch.rand(B, device=device)
            u = u * u if sched == "shallow" else u.sqrt()   # "deep"
            depths = (1 + (K * u).long()).clamp(1, K)
        states, acts = env.scramble(B, depths, return_actions=True)
        last = acts[torch.arange(B, device=device), depths - 1]
        labels = env.inverse_action[last]          # the "noise" to predict

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net_fwd(states, depths) if cfg.get("t_dim") else net_fwd(states)
            loss = F.cross_entropy(logits.float(), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        l = loss.item()
        ema_loss = l if ema_loss is None else 0.95 * ema_loss + 0.05 * l

        if it % 50 == 0:
            with torch.no_grad():
                acc = (logits.argmax(1) == labels).float().mean().item()
            ema_acc = acc if ema_acc is None else 0.9 * ema_acc + 0.1 * acc
            now = time.time()
            ips = (it - it_last) / max(now - t_last, 1e-9)
            t_last, it_last = now, it
            rec = {"iter": it, "loss": round(l, 4), "ema": round(ema_loss, 4),
                   "label_acc": round(acc, 4), "ips": round(ips, 2),
                   "lr": opt.param_groups[0]["lr"]}
            mfh.write(json.dumps(rec) + "\n")
            if it % 500 == 0:
                print(json.dumps(rec), flush=True)

        if it % cfg["probe_every"] == 0:
            rates = probe(env, net, cfg["probe_depths"])
            rec = {"iter": it, "probe": rates}
            mfh.write(json.dumps(rec) + "\n")
            print("PROBE " + json.dumps(rec), flush=True)
            net.train()

        if it % cfg["ckpt_every"] == 0 or it == cfg["iters"]:
            torch.save({"iter": it, "net": net.state_dict(), "opt": opt.state_dict(),
                        "cfg": cfg}, os.path.join(out_dir, "ckpt_latest.pt"))
            if it % 50000 == 0:
                torch.save({"iter": it, "net": net.state_dict(), "cfg": cfg},
                           os.path.join(out_dir, f"ckpt_{it}.pt"))

    print("training complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--K", type=int)
    ap.add_argument("--resume")
    args = ap.parse_args()
    cfg = dict(PRESETS[args.preset])
    if args.iters:
        cfg["iters"] = args.iters
    if args.K:
        cfg["K"] = args.K
    train(cfg, args.out, resume=args.resume)
