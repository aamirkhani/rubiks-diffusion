"""Sokoban hybrid: act by argmax over legal actions of
    log p_denoiser(a|s)  -  lambda * V_davi(s')
combining the denoiser's local direction with the value net's forward-dynamics
deadlock awareness. Zero additional training. Selects lambda on a validation
split, reports on the same protocol as eval_sokoban, and appends results to
paper2_data/domain3_sokoban.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from domains.sokoban import SokobanEnv
from domains.eval_sokoban import load
from domains.train_sokoban import solved_rendered

DEV = "cuda"


@torch.no_grad()
def hybrid_rollout(env, pnet, vnet, lam, walls, goals, boxes, agent, max_steps):
    B = walls.shape[0]
    boxes, agent = boxes.clone(), agent.clone()
    done = env.is_solved(walls, goals, boxes, agent)
    for t in range(max_steps):
        if done.all():
            break
        idx = (~done).nonzero(as_tuple=True)[0]
        w, g, b, a_ = walls[idx], goals[idx], boxes[idx], agent[idx]
        legal = env.legal_forward(w, g, b, a_)
        st = env.render(w, g, b, a_)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logp = F.log_softmax(pnet(st).float(), 1)
            nb = env.neighbors_forward(w, g, b, a_).reshape(-1, env.S)
            v = vnet(nb).float().view(-1, env.M)
        v = torch.where(solved_rendered(nb).view(-1, env.M),
                        torch.zeros_like(v), v.clamp_min(0))
        score = logp - lam * v
        act = score.masked_fill(~legal, -1e30).argmax(1)
        nb_, na_ = env.step_forward(w, g, b, a_, act)
        boxes[idx], agent[idx] = nb_, na_
        done = done | env.is_solved(walls, goals, boxes, agent)
    return env.is_solved(walls, goals, boxes, agent)


def main():
    envp, pnet, _ = load("runs/soko_diff/ckpt_latest.pt")
    envv, vnet, _ = load("runs/soko_davi/ckpt_latest.pt")
    env = envp

    # lambda selection on a validation split (held-out seed)
    g = torch.Generator(device=DEV).manual_seed(777)
    w, go, b, a_, _ = env.instances_and_scramble(2000, 40, generator=g)
    best_lam, best_rate = None, -1
    for lam in (0.1, 0.25, 0.5, 1.0, 2.0):
        solved = hybrid_rollout(env, pnet, vnet, lam, w, go, b, a_, 200)
        r = solved.float().mean().item()
        print(f"  val lambda={lam}: {r:.4f}", flush=True)
        if r > best_rate:
            best_lam, best_rate = lam, r

    res = {"lambda": best_lam, "val_rate_depth40": round(best_rate, 4)}
    for depth in (10, 25, 40, 60):
        gg = torch.Generator(device=DEV).manual_seed(1000 + depth)  # same as eval
        w, go, b, a_, _ = env.instances_and_scramble(5000, depth, generator=gg)
        t0 = time.time()
        solved = hybrid_rollout(env, pnet, vnet, best_lam, w, go, b, a_,
                                4 * depth + 40)
        res[f"depth_{depth}"] = {
            "solve_rate": round(solved.float().mean().item(), 4),
            "secs": round(time.time() - t0, 1)}
    print("hybrid", json.dumps(res), flush=True)

    path = "paper2_data/domain3_sokoban.json"
    out = json.load(open(path))
    out["hybrid_denoise_proposes_value_scores"] = res
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"updated {path}")


if __name__ == "__main__":
    main()
