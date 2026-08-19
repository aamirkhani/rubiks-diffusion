"""Maze head-to-head on fresh procedural instances with exact per-instance
oracles. Starts sampled uniformly from reachable cells (true generalization
test: net never saw these mazes). Writes paper2_data/domain2_maze.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from domains.maze import MazeEnv, AGENT, GOAL
from domains.train_maze import rollout
from model import PolicyNet, ValueNet

DEV = "cuda"


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEV, weights_only=True)
    cfg = ck["cfg"]
    env = MazeEnv(cfg["n"], device=DEV)
    if cfg["method"] == "denoise":
        net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"],
                        vocab=env.vocab).to(DEV)
    else:
        net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"],
                       vocab=env.vocab).to(DEV)
    net.load_state_dict(ck["net"])
    net.eval()
    return env, net, cfg


def main(B=20000, seed=123):
    out = {"domain": "procedural mazes 15x15 (conditional, per-instance goal)",
           "n_eval_instances": B,
           "start_sampling": "uniform over reachable cells (exact BFS field)"}
    for tag, ckpt in (("denoise", "runs/maze_diff/ckpt_latest.pt"),
                      ("davi", "runs/maze_davi/ckpt_latest.pt")):
        env, net, cfg = load(ckpt)
        g = torch.Generator(device=DEV).manual_seed(seed)
        walls, goal = env.new_instances(B, generator=g)
        field = env.bfs_field(walls, goal)
        # uniform start over reachable, non-goal cells
        w = ((field > 0)).float()
        ok = w.sum(1) > 0
        walls, goal, field, w = walls[ok], goal[ok], field[ok], w[ok]
        start = torch.multinomial(w, 1, generator=g).squeeze(1)
        rows = torch.arange(walls.shape[0], device=DEV)
        d_opt = field[rows, start]
        st = env.make_state(walls, start, goal)

        t0 = time.time()
        solved, actions = rollout(env, net, cfg["method"], st, goal,
                                  max_steps=int(d_opt.max().item()) * 3 + 20)
        lens = (actions[solved] >= 0).sum(1).float()
        res = {
            "iter": None,
            "solve_rate": round(solved.float().mean().item(), 4),
            "avg_len": round(lens.mean().item(), 2) if solved.any() else None,
            "avg_optimal": round(d_opt[solved].float().mean().item(), 2),
            "optimality_ratio": round(
                (lens / d_opt[solved].float()).mean().item(), 3)
            if solved.any() else None,
            "eval_secs": round(time.time() - t0, 1),
        }
        out[tag] = res
        print(tag, json.dumps(res), flush=True)
    os.makedirs("paper2_data", exist_ok=True)
    with open("paper2_data/domain2_maze.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved paper2_data/domain2_maze.json")


if __name__ == "__main__":
    main()
