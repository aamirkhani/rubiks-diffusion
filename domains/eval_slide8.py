"""Exhaustive 8-puzzle comparison: greedy sweep over ALL 181,440 reachable
states for both methods, verified, plus oracle move-quality. Writes
paper2_data/domain1_slide8.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from domains.slide import SlideEnv, bfs_oracle, pack, unpack
from domains.train_slide import rollout, verify
from model import PolicyNet, ValueNet

DEV = "cuda"


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEV, weights_only=True)
    cfg = ck["cfg"]
    env = SlideEnv(cfg["n"], DEV)
    if cfg["method"] == "denoise":
        net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"],
                        vocab=env.vocab).to(DEV)
    else:
        net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"],
                       vocab=env.vocab).to(DEV)
    net.load_state_dict(ck["net"])
    net.eval()
    return env, net, cfg


def sweep(env, net, method, keys, dists):
    N = keys.numel()
    solved_tot, len_sum, opt_sum = 0, 0, 0
    fails = []
    B = 100000
    t0 = time.time()
    for s0 in range(0, N, B):
        st = unpack(keys[s0:s0 + B], env.n)
        solved, actions = rollout(env, net, method, st, max_steps=80)
        ok = verify(env, st, actions) & solved
        assert torch.equal(ok, solved)
        solved_tot += int(solved.sum())
        len_sum += int((actions[solved] >= 0).sum())
        opt_sum += int(dists[s0:s0 + B][solved].sum())
        if (~solved).any():
            fails.append(st[~solved])
    return {
        "greedy_all_states": round(solved_tot / N, 6),
        "solved": solved_tot, "total": N,
        "avg_len": round(len_sum / max(solved_tot, 1), 2),
        "avg_optimal": round(opt_sum / max(solved_tot, 1), 2),
        "sweep_secs": round(time.time() - t0, 1),
    }, fails


def move_quality(env, net, method, keys, dists, n=100000):
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randint(keys.numel(), (n,), device=DEV, generator=g)
    st = unpack(keys[idx], env.n)
    d0 = dists[idx].long()
    legal = env.legal_mask(st)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if method == "denoise":
            a = net(st).float().masked_fill(~legal, -1e9).argmax(1)
        else:
            nb = env.neighbors(st).reshape(-1, env.S)
            v = net(nb).float().view(-1, env.M)
            v = torch.where(env.is_solved(nb).view(-1, env.M),
                            torch.zeros_like(v), v.clamp_min(0))
            a = v.masked_fill(~legal, 1e9).argmin(1)
    nxt = env.step(st, a)
    d1 = dists[torch.searchsorted(keys, pack(nxt, env.n))].long()
    good = d1 < d0
    return round(good[d0 > 0].float().mean().item(), 4)


def main():
    keys, dists = bfs_oracle(3, DEV, expected=181440)
    out = {"domain": "8-puzzle (slide 3x3)", "states": 181440, "diameter": 31}
    for tag, ckpt in (("denoise", "runs/slide3_diff/ckpt_latest.pt"),
                      ("davi", "runs/slide3_davi/ckpt_latest.pt")):
        env, net, cfg = load(ckpt)
        res, fails = sweep(env, net, cfg["method"], keys, dists)
        res["move_quality"] = move_quality(env, net, cfg["method"], keys, dists)
        out[tag] = res
        print(tag, json.dumps(res), flush=True)
    os.makedirs("paper2_data", exist_ok=True)
    with open("paper2_data/domain1_slide8.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved paper2_data/domain1_slide8.json")


if __name__ == "__main__":
    main()
