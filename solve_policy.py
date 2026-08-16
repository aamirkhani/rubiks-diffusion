"""Solvers for the diffusion-style policy net (reverse-process sampling).

Greedy = argmax move each step (ancestral sampling, deterministic).
Beam = keep top-W partial solutions by cumulative log-probability.
Solutions are replay-verified with solve.verify_solutions.
"""
import torch
import torch.nn.functional as F

from cube_env import CubeEnv


@torch.no_grad()
def policy_logits(net, states, chunk=65536):
    outs = []
    for i in range(0, states.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(net(states[i:i + chunk]).float())
    return torch.cat(outs)


@torch.no_grad()
def policy_greedy(env: CubeEnv, net, states, max_steps, forbid_backtrack=True):
    """Batched reverse-process rollout with argmax moves."""
    net.eval()
    B = states.shape[0]
    dev = states.device
    states = states.clone()
    done = env.is_solved(states)
    lengths = torch.zeros(B, dtype=torch.long, device=dev)
    actions = torch.full((B, max_steps), -1, dtype=torch.long, device=dev)
    prev = torch.full((B,), -1, dtype=torch.long, device=dev)

    for t in range(max_steps):
        if done.all():
            break
        idx = (~done).nonzero(as_tuple=True)[0]
        sub = states[idx]
        logits = policy_logits(net, sub)
        if forbid_backtrack:
            p = prev[idx]
            has = p >= 0
            logits[has, env.inverse_action[p[has]]] = -1e9
        a = logits.argmax(1)
        states[idx] = env.step(sub, a)
        actions[idx, t] = a
        prev[idx] = a
        lengths[idx] += 1
        done = done | env.is_solved(states)
    return env.is_solved(states), lengths, actions


@torch.no_grad()
def policy_beam_solve(env: CubeEnv, net, states, width, max_depth, chunk=20000):
    """Batched beam search maximizing cumulative log-prob of the move sequence.

    Returns (solved [C] bool, actions [C, max_depth] long, -1 padded).
    """
    net.eval()
    dev = states.device
    C_all = states.shape[0]
    solved_out = torch.zeros(C_all, dtype=torch.bool, device=dev)
    actions_out = torch.full((C_all, max_depth), -1, dtype=torch.long, device=dev)
    chunk = max(256, min(chunk, chunk * 32 // max(width, 1)))

    for c0 in range(0, C_all, chunk):
        st = states[c0:c0 + chunk]
        C = st.shape[0]
        idx_map = torch.arange(C, device=dev)
        pre = env.is_solved(st)
        solved_out[c0:c0 + chunk] |= pre
        keep = ~pre
        idx_map = idx_map[keep]
        beam = st[keep].unsqueeze(1)                                # [A, W, S]
        score = torch.zeros(int(keep.sum()), 1, device=dev)         # [A, W]
        seqs = torch.full((int(keep.sum()), 1, 0), -1, dtype=torch.long, device=dev)

        for t in range(max_depth):
            A, W = beam.shape[0], beam.shape[1]
            if A == 0:
                break
            flat = beam.reshape(A * W, env.S)
            logits = policy_logits(net, flat)
            logp = F.log_softmax(logits, dim=1).view(A, W, env.M)
            cand_score = (score.unsqueeze(2) + logp).view(A, W * env.M)

            nb = env.neighbors(flat).reshape(A, W * env.M, env.S)
            solved = env.is_solved(nb.reshape(-1, env.S)).view(A, W * env.M)

            hit = solved.any(1)
            if hit.any():
                rows = hit.nonzero(as_tuple=True)[0]
                # among solved candidates pick the highest-scoring one
                sc = cand_score[rows].masked_fill(~solved[rows], -1e30)
                best = sc.argmax(1)
                parent = best // env.M
                mv = best % env.M
                out_rows = idx_map[rows] + c0
                if t > 0:
                    actions_out[out_rows, :t] = seqs[rows, parent]
                actions_out[out_rows, t] = mv
                solved_out[out_rows] = True
                keepm = ~hit
                idx_map, beam, score, seqs = (
                    idx_map[keepm], beam[keepm], score[keepm], seqs[keepm])
                nb, cand_score = nb[keepm], cand_score[keepm]
                A = beam.shape[0]
                if A == 0:
                    break

            K = min(width, cand_score.shape[1])
            top = cand_score.topk(K, dim=1, largest=True).indices    # [A, K]
            parent = top // env.M
            mv = (top % env.M).unsqueeze(2)
            beam = torch.gather(nb, 1, top.unsqueeze(2).expand(A, K, env.S))
            score = torch.gather(cand_score, 1, top)
            seqs = torch.cat(
                [torch.gather(seqs, 1, parent.unsqueeze(2).expand(A, K, seqs.shape[2])),
                 mv], dim=2)
    return solved_out, actions_out
