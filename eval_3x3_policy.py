"""3x3 evaluation for the diffusion-style solver: greedy rollout + log-prob
beam on fully scrambled cubes, replay-verified."""
import argparse
import json
import time

import torch

from cube_env import CubeEnv
from model import PolicyNet
from solve import verify_solutions
from solve_policy import policy_greedy, policy_beam_solve

DEV = "cuda"


def main(ckpt_path, out_json, n_cubes=1000, scramble_depth=100, width=2048,
         max_depth=60, seed=1234):
    env = CubeEnv(3, DEV)
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=True)
    cfg = ck["cfg"]
    net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"]).to(DEV)
    net.load_state_dict(ck["net"])
    net.eval()

    g = torch.Generator(device=DEV).manual_seed(seed)
    scrambles = env.scramble(n_cubes, scramble_depth, generator=g)

    t0 = time.time()
    solved_g, len_g, act_g = policy_greedy(env, net, scrambles, max_steps=max_depth)
    ok = verify_solutions(env, scrambles, act_g) & solved_g
    assert torch.equal(ok, solved_g)
    n_g = int(solved_g.sum())
    print(f"greedy rollout: {n_g}/{n_cubes} "
          f"avg len {len_g[solved_g].float().mean().item():.1f} ({time.time()-t0:.0f}s)", flush=True)

    need = ~solved_g
    beam_solved = 0
    beam_lens = []
    if need.any():
        t0 = time.time()
        fails = scrambles[need]
        solved_b, actions_b = policy_beam_solve(env, net, fails, width=width, max_depth=max_depth)
        okb = verify_solutions(env, fails, actions_b)
        assert torch.equal(okb, solved_b), "beam solution failed replay!"
        beam_solved = int(solved_b.sum())
        beam_lens = (actions_b[solved_b] >= 0).sum(1).tolist()
        print(f"beam {width}: {beam_solved}/{int(need.sum())} ({time.time()-t0:.0f}s)", flush=True)

    total_solved = n_g + beam_solved
    all_lens = len_g[solved_g].tolist() + beam_lens
    report = {
        "ckpt": ckpt_path, "iter": ck["iter"],
        "method": "diffusion-style (inverse-move prediction)",
        "n_cubes": n_cubes, "scramble_depth": scramble_depth, "beam_width": width,
        "greedy_solved": n_g, "beam_solved": beam_solved,
        "solve_rate": round(total_solved / n_cubes, 4),
        "avg_solution_len": round(sum(all_lens) / max(len(all_lens), 1), 2),
        "max_solution_len": max(all_lens) if all_lens else None,
    }
    print(json.dumps(report, indent=2))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/3x3_diff/ckpt_latest.pt")
    ap.add_argument("--out", default="runs/3x3_diff/eval_report.json")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--max-depth", type=int, default=60)
    args = ap.parse_args()
    main(args.ckpt, args.out, n_cubes=args.n, scramble_depth=args.depth,
         width=args.width, max_depth=args.max_depth)
