"""Full validation of the diffusion-style 2x2 solver against the exact oracle.

Mirrors eval_2x2.py: sweep ALL 3,674,160 states with greedy rollout, then
escalating log-prob beam stages; every solution replay-verified. Adds a
move-quality metric: how often the argmax move actually reduces true distance
(checkable exactly thanks to the oracle).
"""
import argparse
import json
import time

import torch

from bfs_2x2 import load_oracle
from cube_env import CubeEnv, unpack_2x2, pack_2x2
from model import PolicyNet
from solve import verify_solutions
from solve_policy import policy_greedy, policy_beam_solve, policy_logits

DEV = "cuda"


def main(ckpt_path, out_json, n_uniform=100000):
    env = CubeEnv(2, DEV)
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=True)
    cfg = ck["cfg"]
    net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"]).to(DEV)
    net.load_state_dict(ck["net"])
    net.eval()
    keys, dists = load_oracle(device=DEV)
    N = keys.numel()
    report = {"ckpt": ckpt_path, "iter": ck["iter"], "method": "diffusion-style (inverse-move prediction)"}

    # ---- move quality vs oracle on uniform states ----
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randint(N, (n_uniform,), device=DEV, generator=g)
    st = unpack_2x2(keys[idx])
    d_true = dists[idx].long()
    logits = policy_logits(net, st)
    a = logits.argmax(1)
    nxt = env.step(st, a)
    d_next = dists[torch.searchsorted(keys, pack_2x2(nxt))].long()
    good = (d_next < d_true) | ((d_true == 0) & (d_next == 0))
    acc_by_depth = {}
    for d in range(1, 15):
        m = d_true == d
        if m.any():
            acc_by_depth[d] = round(good[m].float().mean().item(), 4)
    report["argmax_move_reduces_distance"] = {
        "overall": round(good[d_true > 0].float().mean().item(), 4),
        "by_depth": acc_by_depth,
    }
    print(f"argmax move reduces true distance: {report['argmax_move_reduces_distance']['overall']:.2%}")
    print(f"by depth: {acc_by_depth}")

    # ---- solve sweep over ALL states ----
    print(f"\nsolve sweep over ALL {N:,} states (greedy rollout, beam fallback, replay-verified)", flush=True)
    t0 = time.time()
    total = solved_total = 0
    len_sum = opt_sum = 0
    fail_states = []
    B = 200000
    for start in range(0, N, B):
        batch_keys = keys[start:start + B]
        st = unpack_2x2(batch_keys)
        d_true = dists[start:start + B].long()
        solved, lengths, actions = policy_greedy(env, net, st, max_steps=40)
        ok = verify_solutions(env, st, actions) & solved
        assert torch.equal(ok, solved), "greedy claimed solve failed replay!"
        total += st.shape[0]
        solved_total += int(solved.sum())
        len_sum += int(lengths[solved].sum())
        opt_sum += int(d_true[solved].sum())
        if (~solved).any():
            fail_states.append(st[~solved])
        if (start // B) % 5 == 0:
            print(f"  {min(start+B,N):>9,}/{N:,} greedy-solved: {solved_total}/{total} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    greedy_rate = solved_total / total
    report["greedy"] = {"solved": solved_total, "total": total, "rate": round(greedy_rate, 6),
                        "avg_len": round(len_sum / max(solved_total, 1), 3),
                        "avg_optimal": round(opt_sum / max(solved_total, 1), 3)}
    print(f"greedy rollout: {solved_total:,}/{total:,} = {greedy_rate:.4%} "
          f"avg len {report['greedy']['avg_len']} vs optimal {report['greedy']['avg_optimal']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    beam_ok_total = 0
    beam_len_sum = 0
    if fail_states:
        fails = torch.cat(fail_states)
        stages = []
        for width, mdepth in ((32, 22), (256, 22), (2048, 26), (8192, 40), (32768, 40)):
            if fails.shape[0] == 0:
                break
            print(f"\nbeam width {width} on {fails.shape[0]:,} remaining...", flush=True)
            t1 = time.time()
            solved_b, actions_b = policy_beam_solve(env, net, fails, width=width, max_depth=mdepth)
            ok = verify_solutions(env, fails, actions_b)
            assert torch.equal(ok, solved_b), "beam claimed solve failed replay!"
            n_ok = int(solved_b.sum())
            beam_ok_total += n_ok
            beam_len_sum += int((actions_b[solved_b] >= 0).sum())
            stages.append({"width": width, "attempted": int(fails.shape[0]),
                           "solved": n_ok, "secs": round(time.time() - t1, 1)})
            print(f"  solved {n_ok:,}/{fails.shape[0]:,} ({time.time()-t1:.0f}s)", flush=True)
            fails = fails[~solved_b]
        report["beam_fallback"] = {"stages": stages, "unsolved": int(fails.shape[0]),
                                   "avg_beam_len": round(beam_len_sum / max(beam_ok_total, 1), 2)}
        if fails.shape[0] > 0:
            torch.save(fails.cpu(), out_json.replace(".json", "_unsolved.pt"))
            d_fail = dists[torch.searchsorted(keys, pack_2x2(fails))].long()
            hist = torch.bincount(d_fail, minlength=15).tolist()
            report["unsolved_true_depth_hist"] = {str(d): c for d, c in enumerate(hist) if c}
            print(f"UNSOLVED: {fails.shape[0]}  depth hist: {report['unsolved_true_depth_hist']}", flush=True)
    report["combined_rate"] = round((solved_total + beam_ok_total) / total, 8)
    print(f"\nCOMBINED SOLVE RATE: {report['combined_rate']:.6%} of ALL {N:,} states")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report saved to {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/2x2_diff/ckpt_latest.pt")
    ap.add_argument("--out", default="runs/2x2_diff/eval_report.json")
    args = ap.parse_args()
    main(args.ckpt, args.out)
