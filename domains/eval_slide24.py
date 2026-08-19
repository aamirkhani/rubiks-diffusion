"""24-puzzle head-to-head: greedy + legality-aware beam on long scrambles,
replay-verified. Writes/updates paper2_data/domain1_slide24.json.

  python domains/eval_slide24.py --tag denoise --ckpt runs/slide5_diff/ckpt_latest.pt
  python domains/eval_slide24.py --tag davi    --ckpt runs/slide5_davi/ckpt_latest.pt
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from domains.slide import SlideEnv
from domains.train_slide import rollout, verify
from model import PolicyNet, ValueNet

DEV = "cuda"


@torch.no_grad()
def _fwd(net, x, chunk=16384):
    outs = []
    for i in range(0, x.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(net(x[i:i + chunk]).float())
    return torch.cat(outs)


@torch.no_grad()
def beam(env, net, method, states, width, max_depth, chunk=2000):
    """Legality-aware batched beam. Scores: cumulative logprob (denoise) or
    predicted cost-to-go (davi)."""
    net.eval()
    dev = states.device
    C_all = states.shape[0]
    solved_out = torch.zeros(C_all, dtype=torch.bool, device=dev)
    actions_out = torch.full((C_all, max_depth), -1, dtype=torch.long, device=dev)
    chunk = max(64, min(chunk, chunk * 32 // max(width, 1)))
    for c0 in range(0, C_all, chunk):
        st = states[c0:c0 + chunk]
        C = st.shape[0]
        idx_map = torch.arange(C, device=dev)
        pre = env.is_solved(st)
        solved_out[c0:c0 + chunk] |= pre
        keep = ~pre
        idx_map = idx_map[keep]
        bm = st[keep].unsqueeze(1)
        score = torch.zeros(int(keep.sum()), 1, device=dev)
        seqs = torch.full((int(keep.sum()), 1, 0), -1, dtype=torch.long, device=dev)
        for t in range(max_depth):
            A, W = bm.shape[0], bm.shape[1]
            if A == 0:
                break
            flat = bm.reshape(A * W, env.S)
            legal = env.legal_mask(flat).view(A, W * env.M)
            nb = env.neighbors(flat).reshape(A, W * env.M, env.S)
            if method == "denoise":
                lp = F.log_softmax(_fwd(net, flat), 1).view(A, W * env.M)
                cand = score.unsqueeze(2).expand(A, W, env.M).reshape(A, -1) + lp
                cand = cand.masked_fill(~legal, -1e30)
                better = torch.argsort(cand, dim=1, descending=True)
            else:
                v = _fwd(net, nb.reshape(-1, env.S)).view(A, W * env.M)
                v = torch.where(env.is_solved(nb.reshape(-1, env.S)).view(A, -1),
                                torch.zeros_like(v), v.clamp_min(0))
                cand = v.masked_fill(~legal, 1e30)
                better = torch.argsort(cand, dim=1)
            solved = env.is_solved(nb.reshape(-1, env.S)).view(A, W * env.M) & legal
            hit = solved.any(1)
            if hit.any():
                rows = hit.nonzero(as_tuple=True)[0]
                first = solved[rows].float().argmax(1)
                parent, mv = first // env.M, first % env.M
                out_rows = idx_map[rows] + c0
                if t > 0:
                    actions_out[out_rows, :t] = seqs[rows, parent]
                actions_out[out_rows, t] = mv
                solved_out[out_rows] = True
                km = ~hit
                idx_map, bm, score, seqs = idx_map[km], bm[km], score[km], seqs[km]
                nb, cand, better = nb[km], cand[km], better[km]
                A = bm.shape[0]
                if A == 0:
                    break
            K = min(width, cand.shape[1])
            top = better[:, :K]
            parent = top // env.M
            mv = (top % env.M).unsqueeze(2)
            bm = torch.gather(nb, 1, top.unsqueeze(2).expand(A, K, env.S))
            if method == "denoise":
                score = torch.gather(cand, 1, top)
            else:
                score = torch.zeros(A, K, device=dev)
            seqs = torch.cat([torch.gather(seqs, 1, parent.unsqueeze(2)
                                           .expand(A, K, seqs.shape[2])), mv], 2)
    return solved_out, actions_out


def main(tag, ckpt, n_cubes=500, scramble=1000, max_steps=400, width=128):
    ck = torch.load(ckpt, map_location=DEV, weights_only=True)
    cfg = ck["cfg"]
    env = SlideEnv(cfg["n"], DEV)
    cls = PolicyNet if cfg["method"] == "denoise" else ValueNet
    if cfg["method"] == "denoise":
        net = PolicyNet(env.S, env.M, cfg["h1"], cfg["h2"], cfg["blocks"],
                        vocab=env.vocab).to(DEV)
    else:
        net = ValueNet(env.S, cfg["h1"], cfg["h2"], cfg["blocks"],
                       vocab=env.vocab).to(DEV)
    net.load_state_dict(ck["net"])
    net.eval()
    g = torch.Generator(device=DEV).manual_seed(99)
    st = env.scramble(n_cubes, scramble, generator=g)

    t0 = time.time()
    solved_g, act_g = rollout(env, net, cfg["method"], st, max_steps)
    okg = verify(env, st, act_g) & solved_g
    assert torch.equal(okg, solved_g)
    lens_g = (act_g[solved_g] >= 0).sum(1).float()
    res = {"iter": ck["iter"], "n": n_cubes, "scramble": scramble,
           "greedy_rate": round(solved_g.float().mean().item(), 4),
           "greedy_avg_len": round(lens_g.mean().item(), 1) if solved_g.any() else None,
           "greedy_secs": round(time.time() - t0, 1)}

    need = ~solved_g
    if need.any():
        t0 = time.time()
        sb, ab = beam(env, net, cfg["method"], st[need], width, max_steps)
        okb = verify(env, st[need], ab)
        assert torch.equal(okb, sb)
        res["beam_width"] = width
        res["beam_extra_solved"] = int(sb.sum())
        res["beam_secs"] = round(time.time() - t0, 1)
        res["total_rate"] = round((int(solved_g.sum()) + int(sb.sum())) / n_cubes, 4)
    else:
        res["total_rate"] = 1.0
    print(tag, json.dumps(res), flush=True)

    os.makedirs("paper2_data", exist_ok=True)
    path = "paper2_data/domain1_slide24.json"
    out = json.load(open(path)) if os.path.exists(path) else {
        "domain": "24-puzzle (slide 5x5)", "state_space": "25!/2 ~ 7.7e24"}
    out[tag] = res
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"updated {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--width", type=int, default=128)
    args = ap.parse_args()
    main(args.tag, args.ckpt, n_cubes=args.n, width=args.width)
