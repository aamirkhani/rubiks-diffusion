"""Ablations for the paper, all on the 2x2 where the exact oracle enables
full-state-space measurement:

  A. depth (noise) schedule: uniform K=18 (main), uniform K=14, shallow-, deep-biased
  B. timestep conditioning: t-conditioned net vs unconditioned (main)

Each variant: 40k iters, then greedy sweep over ALL 3,674,160 states +
argmax-move-quality vs oracle. Results -> ablations_2x2.json
"""
import json

import torch

from bfs_2x2 import load_oracle
from cube_env import CubeEnv, unpack_2x2, pack_2x2
from denoise import PRESETS, train
from model import PolicyNet
from solve import verify_solutions
from solve_policy import policy_logits
from cube_env import CubeEnv

DEV = "cuda"

VARIANTS = {
    "uniform_K18": {},                                  # main run (reuses runs/2x2_diff)
    "uniform_K14": {"K": 14},
    "shallow_K18": {"schedule": "shallow"},
    "deep_K18": {"schedule": "deep"},
    "tcond_K18": {"t_dim": 18},
}


@torch.no_grad()
def greedy_anneal(env, net, states, max_steps, K, t_cond):
    """Greedy rollout; for t-conditioned nets anneal t from K down to 1."""
    net.eval()
    B = states.shape[0]
    states = states.clone()
    done = env.is_solved(states)
    actions = torch.full((B, max_steps), -1, dtype=torch.long, device=DEV)
    prev = torch.full((B,), -1, dtype=torch.long, device=DEV)
    for t in range(max_steps):
        if done.all():
            break
        idx = (~done).nonzero(as_tuple=True)[0]
        sub = states[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if t_cond:
                tt = torch.full((sub.shape[0],), max(K - t, 1), device=DEV)
                logits = net(sub, tt).float()
            else:
                logits = net(sub).float()
        p = prev[idx]
        has = p >= 0
        logits[has, env.inverse_action[p[has]]] = -1e9
        a = logits.argmax(1)
        states[idx] = env.step(sub, a)
        actions[idx, t] = a
        prev[idx] = a
        done = done | env.is_solved(states)
    return env.is_solved(states), actions


def full_sweep(env, net, keys, dists, K, t_cond):
    N = keys.numel()
    solved_total = 0
    B = 200000
    for start in range(0, N, B):
        st = unpack_2x2(keys[start:start + B])
        solved, actions = greedy_anneal(env, net, st, 40, K, t_cond)
        ok = verify_solutions(env, st, actions) & solved
        assert torch.equal(ok, solved)
        solved_total += int(solved.sum())
    return solved_total / N


def move_quality(env, net, keys, dists, n=100000, t_cond=False, K=18):
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randint(keys.numel(), (n,), device=DEV, generator=g)
    st = unpack_2x2(keys[idx])
    d_true = dists[idx].long()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = net(st).float() if not t_cond else net(st, None).float()
    a = logits.argmax(1)
    nxt = env.step(st, a)
    d_next = dists[torch.searchsorted(keys, pack_2x2(nxt))].long()
    good = d_next < d_true
    return round(good[d_true > 0].float().mean().item(), 4)


def main():
    env = CubeEnv(2, DEV)
    keys, dists = load_oracle(device=DEV)
    results = {}
    for name, overrides in VARIANTS.items():
        out_dir = "runs/2x2_diff" if name == "uniform_K18" else f"runs/abl_{name}"
        cfg = dict(PRESETS["2x2"])
        cfg.update(overrides)
        if name != "uniform_K18":
            print(f"\n=== training {name} ===", flush=True)
            train(cfg, out_dir)
        ck = torch.load(f"{out_dir}/ckpt_latest.pt", map_location=DEV, weights_only=True)
        net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"],
                        t_dim=cfg.get("t_dim", 0)).to(DEV)
        net.load_state_dict(ck["net"])
        net.eval()
        t_cond = bool(cfg.get("t_dim", 0))
        rate = full_sweep(env, net, keys, dists, cfg["K"], t_cond)
        mq = move_quality(env, net, keys, dists, t_cond=t_cond, K=cfg["K"])
        results[name] = {"greedy_all_states": round(rate, 6), "move_quality": mq,
                         "K": cfg["K"], "schedule": cfg.get("schedule", "uniform"),
                         "t_conditioned": t_cond}
        print(f"{name}: greedy over all states {rate:.4%}, move quality {mq:.2%}", flush=True)
    with open("ablations_2x2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved ablations_2x2.json")


if __name__ == "__main__":
    main()
