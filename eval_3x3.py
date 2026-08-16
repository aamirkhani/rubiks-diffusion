"""3x3 evaluation: beam-search solve rate on fully scrambled cubes.

Scrambles are long random walks (default 100 moves — far past the ~26 QTM
diameter, i.e. effectively fully scrambled). Every solution is replay-verified.
"""
import argparse
import json
import time

import torch

from cube_env import CubeEnv
from model import ValueNet
from solve import greedy_solve, beam_solve, verify_solutions

DEV = "cuda"


def main(ckpt_path, out_json, n_cubes=1000, scramble_depth=100, width=2048,
         max_depth=60, seed=1234):
    env = CubeEnv(3, DEV)
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=True)
    cfg = ck["cfg"]
    net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"]).to(DEV)
    net.load_state_dict(ck["net"])
    net.eval()

    g = torch.Generator(device=DEV).manual_seed(seed)
    scrambles = env.scramble(n_cubes, scramble_depth, generator=g)

    # cheap first pass: batched greedy
    t0 = time.time()
    solved_g, len_g, act_g = greedy_solve(env, net, scrambles, max_steps=max_depth)
    ok = verify_solutions(env, scrambles, act_g) & solved_g
    assert torch.equal(ok, solved_g)
    print(f"greedy: {int(solved_g.sum())}/{n_cubes} "
          f"avg len {len_g[solved_g].float().mean().item():.1f} ({time.time()-t0:.0f}s)", flush=True)

    # beam search on the rest
    need = (~solved_g).nonzero(as_tuple=True)[0]
    beam_solved = 0
    beam_lens = []
    t0 = time.time()
    for j, i in enumerate(need.tolist()):
        sol = beam_solve(env, net, scrambles[i], width=width, max_depth=max_depth)
        if sol is not None:
            a = torch.tensor(sol, device=DEV).unsqueeze(0)
            assert verify_solutions(env, scrambles[i].unsqueeze(0), a)[0], "beam sol failed replay!"
            beam_solved += 1
            beam_lens.append(len(sol))
        if (j + 1) % 50 == 0:
            print(f"  beam {j+1}/{need.numel()}: solved {beam_solved} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    total_solved = int(solved_g.sum()) + beam_solved
    all_lens = len_g[solved_g].tolist() + beam_lens
    report = {
        "ckpt": ckpt_path, "iter": ck["iter"], "n_cubes": n_cubes,
        "scramble_depth": scramble_depth, "beam_width": width,
        "greedy_solved": int(solved_g.sum()),
        "beam_solved": beam_solved,
        "solve_rate": round(total_solved / n_cubes, 4),
        "avg_solution_len": round(sum(all_lens) / max(len(all_lens), 1), 2),
        "max_solution_len": max(all_lens) if all_lens else None,
    }
    print(json.dumps(report, indent=2))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/3x3_v1/ckpt_latest.pt")
    ap.add_argument("--out", default="runs/3x3_v1/eval_report.json")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--max-depth", type=int, default=60)
    args = ap.parse_args()
    main(args.ckpt, args.out, n_cubes=args.n, scramble_depth=args.depth,
         width=args.width, max_depth=args.max_depth)
