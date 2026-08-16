"""Beam-width compute-scaling scan for both methods on the 3x3 (and 2x2),
using the same batched no-dedup beam for fairness. 200 fresh scrambles per
point, replay-verified. Results -> width_scan.json
"""
import json
import time

import torch

from cube_env import CubeEnv
from model import PolicyNet, ValueNet
from solve import batched_beam_solve, verify_solutions
from solve_policy import policy_beam_solve

DEV = "cuda"
WIDTHS = [1, 8, 32, 128, 512, 2048]


def scan(env, solver, n=200, depth=100, seed=99):
    out = {}
    g = torch.Generator(device=DEV).manual_seed(seed)
    scrambles = env.scramble(n, depth, generator=g)
    for w in WIDTHS:
        t0 = time.time()
        solved, actions = solver(scrambles, w)
        ok = verify_solutions(env, scrambles, actions)
        assert torch.equal(ok, solved)
        lens = (actions[solved] >= 0).sum(1).float()
        out[str(w)] = {"rate": round(solved.float().mean().item(), 4),
                       "avg_len": round(lens.mean().item(), 2) if solved.any() else None,
                       "secs": round(time.time() - t0, 1)}
        print(f"  width {w:>5}: {out[str(w)]}", flush=True)
    return out


def main():
    results = {}
    for n_cube, davi_ckpt, diff_ckpt in (
            (3, "runs/3x3_v1/ckpt_latest.pt", "runs/3x3_diff/ckpt_latest.pt"),
            (2, "runs/2x2_v1/ckpt_latest.pt", "runs/2x2_diff/ckpt_latest.pt")):
        env = CubeEnv(n_cube, DEV)

        ck = torch.load(davi_ckpt, map_location=DEV, weights_only=True)
        vnet = ValueNet(env.S, ck["cfg"]["h1"], ck["cfg"]["h2"], ck["cfg"]["blocks"]).to(DEV)
        vnet.load_state_dict(ck["net"]); vnet.eval()
        print(f"{n_cube}x{n_cube} DAVI:", flush=True)
        results[f"{n_cube}x{n_cube}_davi"] = scan(
            env, lambda s, w: batched_beam_solve(env, vnet, s, width=w, max_depth=60))

        ck = torch.load(diff_ckpt, map_location=DEV, weights_only=True)
        pnet = PolicyNet(env.S, env.M, ck["cfg"]["h1"], ck["cfg"]["h2"], ck["cfg"]["blocks"]).to(DEV)
        pnet.load_state_dict(ck["net"]); pnet.eval()
        print(f"{n_cube}x{n_cube} diffusion:", flush=True)
        results[f"{n_cube}x{n_cube}_diffusion"] = scan(
            env, lambda s, w: policy_beam_solve(env, pnet, s, width=w, max_depth=60))

    with open("width_scan.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved width_scan.json")


if __name__ == "__main__":
    main()
