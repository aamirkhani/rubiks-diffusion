"""Full validation of the trained 2x2 solver against the exact BFS oracle.

1. Value accuracy: mean |V(s) - d*(s)| per true depth, over uniform states.
2. Solve sweep over the ENTIRE state space (all 3,674,160 states) with batched
   greedy descent; beam-search fallback for any state greedy misses.
3. Every solution is replay-verified in the environment.
4. Optimality gap vs exact distances.
"""
import argparse
import json
import time

import torch

from bfs_2x2 import load_oracle
from cube_env import CubeEnv, unpack_2x2, pack_2x2
from model import ValueNet
from solve import greedy_solve, batched_beam_solve, verify_solutions, value

DEV = "cuda"


def main(ckpt_path, out_json, full_sweep=True, n_uniform=100000):
    env = CubeEnv(2, DEV)
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=True)
    cfg = ck["cfg"]
    net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"]).to(DEV)
    net.load_state_dict(ck["net"])
    net.eval()
    keys, dists = load_oracle(device=DEV)
    N = keys.numel()
    report = {"ckpt": ckpt_path, "iter": ck["iter"]}

    # ---- 1. value accuracy on uniform random states ----
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randint(N, (n_uniform,), device=DEV, generator=g)
    st = unpack_2x2(keys[idx])
    d_true = dists[idx].long()
    v = value(net, st)
    mae_by_depth = {}
    for d in range(0, 15):
        m = d_true == d
        if m.any():
            mae_by_depth[d] = round((v[m] - d_true[m].float()).abs().mean().item(), 3)
    overall_mae = round((v - d_true.float()).abs().mean().item(), 4)
    report["value_mae_overall"] = overall_mae
    report["value_mae_by_depth"] = mae_by_depth
    print(f"value MAE overall: {overall_mae}")
    print(f"value MAE by true depth: {mae_by_depth}")

    # ---- 2. solve sweep ----
    scope = "ALL 3,674,160 states" if full_sweep else f"{n_uniform:,} uniform states"
    print(f"\nsolve sweep over {scope} (greedy, beam fallback, replay-verified)")
    t0 = time.time()
    total = solved_total = 0
    fail_states = []
    len_sum = 0
    opt_sum = 0
    B = 200000
    src = range(0, N, B) if full_sweep else [0]
    if not full_sweep:
        pool_idx = idx[:n_uniform]

    for start in src:
        if full_sweep:
            batch_keys = keys[start:start + B]
        else:
            batch_keys = keys[pool_idx]
        st = unpack_2x2(batch_keys)
        d_true = dists[torch.searchsorted(keys, batch_keys)].long()
        solved, lengths, actions = greedy_solve(env, net, st, max_steps=40)
        ok = verify_solutions(env, st, actions) & solved
        assert torch.equal(ok, solved), "greedy claimed solve failed replay!"
        total += st.shape[0]
        solved_total += int(solved.sum())
        len_sum += int(lengths[solved].sum())
        opt_sum += int(d_true[solved].sum())
        if (~solved).any():
            fail_states.append(st[~solved].cpu())
        if full_sweep and (start // B) % 5 == 0:
            print(f"  {min(start+B,N):>9,}/{N:,} greedy-solved so far: "
                  f"{solved_total}/{total} ({time.time()-t0:.0f}s)", flush=True)

    greedy_rate = solved_total / total
    report["greedy"] = {
        "scope": scope, "solved": solved_total, "total": total,
        "rate": round(greedy_rate, 6),
        "avg_len": round(len_sum / max(solved_total, 1), 3),
        "avg_optimal": round(opt_sum / max(solved_total, 1), 3),
    }
    print(f"greedy: {solved_total:,}/{total:,} = {greedy_rate:.4%}  "
          f"avg len {report['greedy']['avg_len']} vs optimal {report['greedy']['avg_optimal']}  "
          f"({time.time()-t0:.0f}s)")

    # ---- 3. beam fallback on greedy failures (batched, escalating width) ----
    beam_ok_total = 0
    beam_len_sum = 0
    if fail_states:
        fails = torch.cat(fail_states).to(DEV)
        stages = []
        for width, mdepth in ((32, 22), (256, 22), (2048, 26), (8192, 40)):
            if fails.shape[0] == 0:
                break
            print(f"\nbeam width {width} on {fails.shape[0]:,} remaining failures...", flush=True)
            t1 = time.time()
            solved_b, actions_b = batched_beam_solve(env, net, fails, width=width, max_depth=mdepth)
            ok = verify_solutions(env, fails, actions_b)
            assert torch.equal(ok, solved_b), "beam claimed solve failed replay!"
            n_ok = int(solved_b.sum())
            beam_ok_total += n_ok
            beam_len_sum += int((actions_b[solved_b] >= 0).sum())
            stages.append({"width": width, "attempted": int(fails.shape[0]),
                           "solved": n_ok, "secs": round(time.time() - t1, 1)})
            print(f"  solved {n_ok:,}/{fails.shape[0]:,} ({time.time()-t1:.0f}s)")
            fails = fails[~solved_b]
        report["beam_fallback"] = {"stages": stages, "unsolved": int(fails.shape[0]),
                                   "avg_beam_len": round(beam_len_sum / max(beam_ok_total, 1), 2)}
        if fails.shape[0] > 0:
            print(f"UNSOLVED after all beam stages: {fails.shape[0]}", flush=True)
            torch.save(fails.cpu(), out_json.replace(".json", "_unsolved.pt"))
            d_fail = dists[torch.searchsorted(keys, pack_2x2(fails))].long()
            hist = torch.bincount(d_fail, minlength=15).tolist()
            report["unsolved_true_depth_hist"] = {str(d): c for d, c in enumerate(hist) if c}
            print(f"unsolved true-depth histogram: {report['unsolved_true_depth_hist']}", flush=True)
    report["combined_rate"] = round((solved_total + beam_ok_total) / total, 8)

    print(f"\nCOMBINED SOLVE RATE: {report['combined_rate']:.6%} of {scope}")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report saved to {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/2x2_v1/ckpt_latest.pt")
    ap.add_argument("--out", default="runs/2x2_v1/eval_report.json")
    ap.add_argument("--quick", action="store_true", help="uniform sample instead of all states")
    args = ap.parse_args()
    main(args.ckpt, args.out, full_sweep=not args.quick)
