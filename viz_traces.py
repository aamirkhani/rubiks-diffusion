"""Generate solve traces (JSON) for the visualizer: scramble -> solved,
with per-step sticker states so the viewer needs no cube logic."""
import argparse
import json

import torch

from cube_env import CubeEnv
from model import ValueNet, PolicyNet
from solve import greedy_solve, batched_beam_solve, verify_solutions
from solve_policy import policy_greedy, policy_beam_solve

DEV = "cuda"


def make_traces(n, ckpt_path, n_traces, scramble_depth, out_path, seed=7, method="value"):
    env = CubeEnv(n, DEV)
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=True)
    cfg = ck["cfg"]
    if method == "policy":
        net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"]).to(DEV)
    else:
        net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"]).to(DEV)
    net.load_state_dict(ck["net"])
    net.eval()

    g = torch.Generator(device=DEV).manual_seed(seed)
    traces = []
    tries = 0
    while len(traces) < n_traces and tries < n_traces * 4:
        tries += 1
        st, scr_actions = env.scramble(1, scramble_depth, return_actions=True, generator=g)
        beam = policy_beam_solve if method == "policy" else batched_beam_solve
        solved, actions = beam(env, net, st, width=512, max_depth=60, chunk=256)
        if not solved[0]:
            grd = policy_greedy if method == "policy" else greedy_solve
            solved_g, lengths, actions_g = grd(env, net, st, max_steps=60)
            if not solved_g[0]:
                continue
            actions = actions_g
        sol = [int(a) for a in actions[0].tolist() if a >= 0]
        a = torch.tensor(sol, device=DEV).unsqueeze(0)
        assert verify_solutions(env, st, a)[0]

        # per-step states
        states = [st[0].tolist()]
        s = st.clone()
        for mv in sol:
            s = env.step(s, torch.tensor([mv], device=DEV))
            states.append(s[0].tolist())
        traces.append({
            "n": n,
            "scramble": [env.move_names[int(x)] for x in scr_actions[0].tolist() if x >= 0],
            "solution": [env.move_names[m] for m in sol],
            "states": states,
        })
    with open(out_path, "w") as f:
        json.dump(traces, f)
    print(f"wrote {len(traces)} traces to {out_path} "
          f"(avg solution {sum(len(t['solution']) for t in traces)/len(traces):.1f} moves)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["value", "policy"], default="value")
    args = ap.parse_args()
    make_traces(args.n, args.ckpt, args.count, args.depth, args.out, method=args.method)
