"""Solvers driven by a trained value net: batched greedy descent + beam search.

Every returned solution is replay-verified by apply-and-check in the env —
a solve only counts if stepping the recorded moves from the scramble state
lands exactly on the solved state.
"""
import torch
from cube_env import CubeEnv


@torch.no_grad()
def value(net, states, chunk=65536):
    outs = []
    for i in range(0, states.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(net(states[i:i + chunk]).float())
    return torch.cat(outs)


@torch.no_grad()
def greedy_solve(env: CubeEnv, net, states, max_steps, forbid_backtrack=True):
    """Batched greedy descent on V. Returns (solved_mask, lengths, actions [B,max_steps])."""
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
        act_idx = (~done).nonzero(as_tuple=True)[0]
        sub = states[act_idx]
        nb = env.neighbors(sub)                                # [b, M, S]
        b = sub.shape[0]
        flat = nb.reshape(b * env.M, env.S)
        v = value(net, flat).view(b, env.M)
        solved_nb = env.is_solved(flat).view(b, env.M)
        v = torch.where(solved_nb, torch.full_like(v, -1e6), v)
        if forbid_backtrack:
            p = prev[act_idx]
            has_prev = p >= 0
            v[has_prev, env.inverse_action[p[has_prev]]] = 1e6
        a = v.argmin(1)
        states[act_idx] = nb[torch.arange(b, device=dev), a]
        actions[act_idx, t] = a
        prev[act_idx] = a
        lengths[act_idx] += 1
        done = done | env.is_solved(states)

    solved = env.is_solved(states)
    return solved, lengths, actions


@torch.no_grad()
def beam_solve(env: CubeEnv, net, state, width, max_depth, hash_vec=None):
    """Beam search from a single state [S]. Score = V(s). Returns move list or None."""
    net.eval()
    dev = state.device
    if env.is_solved(state.unsqueeze(0))[0]:
        return []
    if hash_vec is None:
        g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
        hash_vec = torch.randint(-2**62, 2**62, (env.S,), generator=g).to(dev)

    def hkeys(s):
        return (s.to(torch.int64) * hash_vec).sum(1)

    beam = state.unsqueeze(0)                      # [W, S]
    seqs = torch.empty(1, 0, dtype=torch.long, device=dev)
    visited = hkeys(beam)

    for depth in range(max_depth):
        W = beam.shape[0]
        nb = env.neighbors(beam).reshape(W * env.M, env.S)
        cand_seqs = torch.cat(
            [seqs.repeat_interleave(env.M, 0),
             torch.arange(env.M, device=dev).repeat(W).unsqueeze(1)], dim=1)

        solved = env.is_solved(nb)
        if solved.any():
            i = int(solved.nonzero(as_tuple=True)[0][0])
            return cand_seqs[i].tolist()

        keys = hkeys(nb)
        # dedupe within candidates and against everything seen before
        order = keys.argsort()
        keys_s = keys[order]
        first = torch.ones_like(keys_s, dtype=torch.bool)
        first[1:] = keys_s[1:] != keys_s[:-1]
        keep = order[first]
        keep = keep[~torch.isin(keys[keep], visited)]
        if keep.numel() == 0:
            return None
        nb, cand_seqs, keys = nb[keep], cand_seqs[keep], keys[keep]

        v = value(net, nb)
        top = v.argsort()[:width]
        beam, seqs = nb[top], cand_seqs[top]
        visited = torch.cat([visited, keys[top]])
    return None


@torch.no_grad()
def batched_beam_solve(env: CubeEnv, net, states, width, max_depth, chunk=20000):
    """Beam search over many start states at once (no dedup — width covers it).

    Returns (solved [C] bool, actions [C, max_depth] long with -1 padding).
    """
    net.eval()
    dev = states.device
    C_all = states.shape[0]
    solved_out = torch.zeros(C_all, dtype=torch.bool, device=dev)
    actions_out = torch.full((C_all, max_depth), -1, dtype=torch.long, device=dev)
    # keep the candidate tensor [chunk, width*M, S] bounded regardless of width
    chunk = max(256, min(chunk, chunk * 32 // max(width, 1)))

    for c0 in range(0, C_all, chunk):
        st = states[c0:c0 + chunk]
        C = st.shape[0]
        idx_map = torch.arange(C, device=dev)          # active row -> chunk row
        pre = env.is_solved(st)
        solved_out[c0:c0 + chunk] |= pre
        keep = ~pre
        idx_map = idx_map[keep]
        beam = st[keep].unsqueeze(1)                   # [A, W, S]
        seqs = torch.full((int(keep.sum()), 1, 0), -1, dtype=torch.long, device=dev)

        for t in range(max_depth):
            A, W = beam.shape[0], beam.shape[1]
            if A == 0:
                break
            flat = beam.reshape(A * W, env.S)
            nb = env.neighbors(flat).reshape(A, W * env.M, env.S)
            cand_move = torch.arange(env.M, device=dev).repeat(W)      # [W*M]
            solved = env.is_solved(nb.reshape(-1, env.S)).view(A, W * env.M)

            hit = solved.any(1)
            if hit.any():
                rows = hit.nonzero(as_tuple=True)[0]
                first = solved[rows].float().argmax(1)
                parent = first // env.M
                mv = first % env.M
                out_rows = idx_map[rows] + c0
                if t > 0:
                    actions_out[out_rows, :t] = seqs[rows, parent]
                actions_out[out_rows, t] = mv
                solved_out[out_rows] = True
                keep = ~hit
                idx_map, beam, seqs, nb, solved = (
                    idx_map[keep], beam[keep], seqs[keep], nb[keep], solved[keep])
                A = beam.shape[0]
                if A == 0:
                    break

            v = value(net, nb.reshape(-1, env.S)).view(A, W * env.M)
            K = min(width, W * env.M)
            top = v.topk(K, dim=1, largest=False).indices              # [A, K]
            parent = top // env.M
            mv = (top % env.M).unsqueeze(2)                            # [A, K, 1]
            beam = torch.gather(nb, 1, top.unsqueeze(2).expand(A, K, env.S))
            seqs = torch.cat(
                [torch.gather(seqs, 1, parent.unsqueeze(2).expand(A, K, seqs.shape[2])),
                 mv], dim=2)
    return solved_out, actions_out


@torch.no_grad()
def verify_solutions(env: CubeEnv, start_states, actions, lengths=None):
    """Replay recorded actions from start states; return mask of true solves.

    actions: [B, T] with -1 padding (ignored).
    """
    s = start_states.clone()
    B, T = actions.shape
    for t in range(T):
        a = actions[:, t]
        valid = a >= 0
        if not valid.any():
            break
        safe_a = torch.where(valid, a, torch.zeros_like(a))
        nxt = env.step(s, safe_a)
        s = torch.where(valid.unsqueeze(1), nxt, s)
    return env.is_solved(s)
