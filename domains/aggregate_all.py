"""Aggregate every trained seed across all 10 domains into
paper2_data/seed_variance.json and write domain JSONs for domains 6-10.
Run after sweeps complete. Idempotent; skips missing checkpoints.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import PolicyNet, ValueNet

DEV = "cuda"
OUT = {}


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(sum(vals) / len(vals), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "n_seeds": len(vals), "per_seed": [round(v, 4) for v in vals]}


def seeds_for(pattern):
    outs = []
    for p in sorted(glob.glob(pattern)):
        ck = os.path.join(p, "ckpt_latest.pt")
        if os.path.exists(ck):
            outs.append(ck)
    return outs


# ---------------------------------------------------------------- slide3
def eval_slide3():
    from domains.slide import SlideEnv, bfs_oracle
    from domains.eval_slide8 import sweep, load
    keys, dists = bfs_oracle(3, DEV, expected=181440)
    res = {}
    for tag, pat, s0 in (("denoise", "runs/slide3_diff_s*", "runs/slide3_diff"),
                         ("davi", "runs/slide3_davi_s*", "runs/slide3_davi")):
        rates, lens = [], []
        for ck in [os.path.join(s0, "ckpt_latest.pt")] + seeds_for(pat):
            if not os.path.exists(ck):
                continue
            env, net, cfg = load(ck)
            r, _ = sweep(env, net, cfg["method"], keys, dists)
            rates.append(r["greedy_all_states"])
            lens.append(r["avg_len"])
        res[tag] = {"greedy_all_states": agg(rates), "avg_len": agg(lens)}
    return res


# ------------------------------------------------------------------ maze
def eval_maze_seeds(pomdp=False):
    from domains.maze import MazeEnv
    import domains.train_maze as tm
    res = {}
    pre = "pomdp" if pomdp else "maze"
    for tag, pat, s0 in ((f"denoise", f"runs/{pre}_diff_s*", f"runs/{pre}_diff"),
                         (f"davi", f"runs/{pre}_davi_s*", f"runs/{pre}_davi")):
        rates = []
        cks = seeds_for(pat)
        if os.path.exists(os.path.join(s0, "ckpt_latest.pt")):
            cks = [os.path.join(s0, "ckpt_latest.pt")] + cks
        for ck in dict.fromkeys(cks):
            c = torch.load(ck, map_location=DEV, weights_only=True)
            cfg = c["cfg"]
            env = MazeEnv(cfg["n"], device=DEV)
            vocab = 5 if cfg.get("radius") else 4
            cls = PolicyNet if cfg["method"] == "denoise" else ValueNet
            if cfg["method"] == "denoise":
                net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"],
                                cfg["blocks"], vocab=vocab).to(DEV)
            else:
                net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"],
                               vocab=vocab).to(DEV)
            net.load_state_dict(c["net"])
            net.eval()
            tm.OBS_RADIUS = cfg.get("radius")
            g = torch.Generator(device=DEV).manual_seed(123)
            walls, goal = env.new_instances(20000, generator=g)
            field = env.bfs_field(walls, goal)
            w = (field > 0).float()
            ok = w.sum(1) > 0
            walls, goal, w = walls[ok], goal[ok], w[ok]
            start = torch.multinomial(w, 1, generator=g).squeeze(1)
            st = env.make_state(walls, start, goal)
            solved, _ = tm.rollout(env, net, cfg["method"], st, goal, 150)
            rates.append(solved.float().mean().item())
            tm.OBS_RADIUS = None
        res[tag] = {"solve_rate": agg(rates)}
    return res


# --------------------------------------------------------------- sokoban
def eval_soko_seeds():
    from domains.eval_sokoban import load
    from domains.train_sokoban import rollout
    res = {}
    for tag, pat, s0 in (("denoise", "runs/soko_diff_s*", "runs/soko_diff"),
                         ("davi", "runs/soko_davi_s*", "runs/soko_davi")):
        rates = []
        cks = [os.path.join(s0, "ckpt_latest.pt")] + seeds_for(pat)
        for ck in dict.fromkeys(cks):
            if not os.path.exists(ck):
                continue
            env, net, cfg = load(ck)
            g = torch.Generator(device=DEV).manual_seed(1060)
            w, go, b, a_, _ = env.instances_and_scramble(5000, 60, generator=g)
            solved, _ = rollout(env, net, cfg["method"], w, go, b, a_, 280)
            rates.append(solved.float().mean().item())
        res[tag] = {"solve_rate_depth60": agg(rates)}
    return res


# -------------------------------------------------------------- pendulum
def eval_pend_seeds():
    from domains.pendulum import Reg, NBINS, rollout as prroll
    import numpy as np
    res = {}
    for tag, pat, s0, out_dim, method in (
            ("denoise_mse", "runs/pend_diff_s*", "runs/pend_diff", 1, "denoise"),
            ("denoise_disc", "runs/pend_disc_s*", "runs/pend_diff_disc", NBINS,
             "denoise-disc"),
            ("value", "runs/pend_value_s*", "runs/pend_value", 1, "value")):
        rates = []
        cks = [os.path.join(s0, "ckpt_latest.pt")] + seeds_for(pat)
        for ck in dict.fromkeys(cks):
            if not os.path.exists(ck):
                continue
            c = torch.load(ck, map_location=DEV, weights_only=True)
            net = Reg(out=out_dim).to(DEV)
            net.load_state_dict(c["net"])
            net.eval()
            g = torch.Generator(device=DEV).manual_seed(1)
            th0 = torch.pi + torch.randn(2048, device=DEV, generator=g) * 0.1
            om0 = torch.randn(2048, device=DEV, generator=g) * 0.1
            ug = torch.linspace(-2, 2, 9, device=DEV)
            done, _ = prroll(net, method, th0, om0, ugrid=ug)
            rates.append(done.float().mean().item())
        res[tag] = {"swingup_rate": agg(rates)}
    return res


# ------------------------------------------------------------------ 2048
def eval_2048_seeds():
    from domains.train_2048 import evaluate, VOCAB
    from domains.game2048 import S
    res = {"denoiser_reach256": [], "random_reach256": None}
    cks = [os.path.join("runs/g2048_diff", "ckpt_latest.pt")] + \
        seeds_for("runs/g2048_diff_s*")
    rates = []
    for ck in dict.fromkeys(cks):
        if not os.path.exists(ck):
            continue
        c = torch.load(ck, map_location=DEV, weights_only=True)
        net = PolicyNet(S, 4, 2048, 1024, 3, vocab=VOCAB).to(DEV)
        net.load_state_dict(c["net"])
        r = evaluate(net, B=2048, seed=999)
        rates.append(r["denoiser"]["reach_256"])
        res["random_reach256"] = r["random"]["reach_256"]
        res["greedy_merge_reach256"] = r["greedy_merge"]["reach_256"]
    res["denoiser_reach256"] = agg(rates)
    return res


# ----------------------------------------------------------------- hanoi
def eval_hanoi_seeds():
    from domains.hanoi import HanoiEnv, bfs_oracle, pack, unpack
    from domains.train_slide import rollout, verify
    keys, dists = bfs_oracle(10, DEV)
    env = HanoiEnv(10, DEV)
    res = {}
    for tag, pat in (("denoise", "runs/hanoi_diff_s*"),
                     ("davi", "runs/hanoi_davi_s*")):
        rates, classic = [], []
        for ck in seeds_for(pat):
            c = torch.load(ck, map_location=DEV, weights_only=True)
            cfg = c["cfg"]
            if cfg["method"] == "denoise":
                net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"],
                                cfg["blocks"], vocab=env.vocab).to(DEV)
            else:
                net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"],
                               vocab=env.vocab).to(DEV)
            net.load_state_dict(c["net"])
            net.eval()
            # exhaustive sweep over all 59,049 states
            st = unpack(keys, 10)
            solved, actions = rollout(env, net, cfg["method"], st, 2500)
            ok = verify(env, st, actions) & solved
            assert torch.equal(ok, solved)
            rates.append(solved.float().mean().item())
            start = torch.zeros(1, 10, dtype=torch.int8, device=DEV)
            s1, a1 = rollout(env, net, cfg["method"], start, 2500)
            classic.append(int((a1[0] >= 0).sum()) if s1[0] else None)
        res[tag] = {"greedy_all_states": agg(rates),
                    "classic_start_len": [c for c in classic]}
    return res


# ------------------------------------------------------------- lightsout
def eval_lo_seeds():
    from domains.lightsout import LightsOutEnv, optimal_lengths
    from domains.train_misc import rollout, verify
    env = LightsOutEnv(DEV)
    res = {}
    for tag, pat in (("denoise", "runs/lo_diff_s*"), ("davi", "runs/lo_davi_s*")):
        rates, opt_ratio = [], []
        for ck in seeds_for(pat):
            c = torch.load(ck, map_location=DEV, weights_only=True)
            cfg = c["cfg"]
            if cfg["method"] == "denoise":
                net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"],
                                cfg["blocks"], vocab=env.vocab).to(DEV)
            else:
                net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"],
                               vocab=env.vocab).to(DEV)
            net.load_state_dict(c["net"])
            net.eval()
            g = torch.Generator(device=DEV).manual_seed(5)
            st = env.scramble(5000, torch.randint(1, 26, (5000,), device=DEV,
                                                  generator=g), generator=g)
            solved, actions = rollout(env, net, cfg["method"], st, 60,
                                      forbid_repeat=(cfg["method"] == "denoise"))
            ok = verify(env, st, actions) & solved
            assert torch.equal(ok, solved)
            rates.append(solved.float().mean().item())
            d = optimal_lengths(st[:1000].cpu()).float()
            L = (actions[:1000][solved[:1000]] >= 0).sum(1).float().cpu()
            opt_ratio.append((L / d[solved[:1000].cpu()].clamp_min(1)).mean().item())
        res[tag] = {"solve_rate": agg(rates), "len_over_optimal": agg(opt_ratio)}
    return res


# ------------------------------------------------------------ mountaincar
def eval_mc_seeds():
    from domains.mountaincar import Net, rollout
    res = {}
    for tag, pat, out_dim, method in (
            ("denoise", "runs/mc_diff_s*", 3, "denoise"),
            ("value", "runs/mc_value_s*", 1, "value")):
        rates = []
        for ck in seeds_for(pat):
            c = torch.load(ck, map_location=DEV, weights_only=True)
            net = Net(out_dim).to(DEV)
            net.load_state_dict(c["net"])
            net.eval()
            g = torch.Generator(device=DEV).manual_seed(1)
            x0 = -0.5 + torch.randn(4096, device=DEV, generator=g) * 0.05
            v0 = torch.zeros(4096, device=DEV)
            done, _ = rollout(net, method, x0, v0)
            rates.append(done.float().mean().item())
        res[tag] = {"reach_rate": agg(rates)}
    return res


# ------------------------------------------------------------------ pegs
def eval_peg_seeds():
    from domains.pegsolitaire import PegEnv
    from domains.train_misc import rollout, verify
    env = PegEnv(DEV)
    res = {}
    for tag, pat in (("denoise", "runs/peg_diff_s*"),
                     ("davi", "runs/peg_davi_s*")):
        rates = []
        for ck in seeds_for(pat):
            c = torch.load(ck, map_location=DEV, weights_only=True)
            cfg = c["cfg"]
            if cfg["method"] == "denoise":
                net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"],
                                cfg["blocks"], vocab=env.vocab).to(DEV)
            else:
                net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"],
                               vocab=env.vocab).to(DEV)
            net.load_state_dict(c["net"])
            net.eval()
            g = torch.Generator(device=DEV).manual_seed(9)
            st, _ = env.scramble(5000, 25, return_actions=True, generator=g)
            solved, actions = rollout(env, net, cfg["method"], st, 60)
            ok = verify(env, st, actions) & solved
            assert torch.equal(ok, solved)
            rates.append(solved.float().mean().item())
        res[tag] = {"solve_rate_depth25": agg(rates)}
    return res


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    jobs = [("slide3", eval_slide3), ("maze", eval_maze_seeds),
            ("pomdp_maze", lambda: eval_maze_seeds(pomdp=True)),
            ("sokoban", eval_soko_seeds), ("pendulum", eval_pend_seeds),
            ("g2048", eval_2048_seeds), ("hanoi", eval_hanoi_seeds),
            ("lightsout", eval_lo_seeds), ("mountaincar", eval_mc_seeds),
            ("pegs", eval_peg_seeds)]
    for name, fn in jobs:
        try:
            OUT[name] = fn()
            print(name, json.dumps(OUT[name]), flush=True)
        except Exception as e:
            OUT[name] = {"error": str(e)}
            print(name, "ERROR", e, flush=True)
    with open("paper2_data/seed_variance.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print("saved paper2_data/seed_variance.json")


if __name__ == "__main__":
    main()
