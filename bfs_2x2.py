"""Exact distance oracle for the 2x2 cube: BFS over the entire state space.

With moves restricted to U/R/F quarter turns the DLB corner is fixed, so the
reachable space is exactly 7! * 3^6 = 3,674,160 states and the QTM diameter
is 14. Saves a sorted (key -> distance) table for O(log N) lookups.
"""
import time
import torch
from cube_env import CubeEnv, pack_2x2

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EXPECTED_STATES = 3_674_160
EXPECTED_DIAMETER = 14  # QTM


def run(out_path="oracle_2x2.pt"):
    env = CubeEnv(2, DEV)
    t0 = time.time()

    solved = env.solved.unsqueeze(0)
    frontier = solved
    visited_keys = pack_2x2(solved)                 # sorted tensor of all seen keys
    all_keys = [visited_keys.clone()]
    all_dists = [torch.zeros(1, dtype=torch.int8, device=DEV)]
    counts = [1]

    d = 0
    while frontier.shape[0] > 0:
        nxt = env.neighbors(frontier).reshape(-1, 24)
        keys = pack_2x2(nxt)
        # dedupe within the expansion, keeping one representative state per key
        order = keys.argsort()
        keys_s = keys[order]
        first = torch.ones_like(keys_s, dtype=torch.bool)
        first[1:] = keys_s[1:] != keys_s[:-1]
        keys_u = keys_s[first]
        states_u = nxt[order][first]
        # drop already-visited keys
        new_mask = ~torch.isin(keys_u, visited_keys, assume_unique=True)
        keys_new = keys_u[new_mask]
        frontier = states_u[new_mask]
        d += 1
        if keys_new.numel() == 0:
            break
        counts.append(int(keys_new.numel()))
        all_keys.append(keys_new)
        all_dists.append(torch.full((keys_new.numel(),), d, dtype=torch.int8, device=DEV))
        visited_keys = torch.sort(torch.cat([visited_keys, keys_new])).values
        print(f"depth {d:2d}: {keys_new.numel():>9,} new states "
              f"(total {visited_keys.numel():,}, {time.time()-t0:.1f}s)")

    keys = torch.cat(all_keys)
    dists = torch.cat(all_dists)
    order = keys.argsort()
    keys, dists = keys[order], dists[order]

    total = keys.numel()
    diameter = int(dists.max().item())
    print(f"\ntotal states: {total:,}  diameter (QTM): {diameter}  time: {time.time()-t0:.1f}s")
    assert total == EXPECTED_STATES, f"state count {total} != {EXPECTED_STATES} — move tables wrong!"
    assert diameter == EXPECTED_DIAMETER, f"diameter {diameter} != {EXPECTED_DIAMETER}"
    print("MATCHES known 2x2 group order and QTM diameter — engine is exact.")

    torch.save({"keys": keys.cpu(), "dists": dists.cpu(), "counts": counts}, out_path)
    print(f"saved oracle to {out_path}")


def load_oracle(path="oracle_2x2.pt", device=DEV):
    d = torch.load(path, map_location=device, weights_only=True)
    return d["keys"].to(device), d["dists"].to(device)


def lookup(keys_sorted, dists, query_keys):
    idx = torch.searchsorted(keys_sorted, query_keys)
    assert (keys_sorted[idx] == query_keys).all(), "query key not in oracle (unreachable state?)"
    return dists[idx]


if __name__ == "__main__":
    run()
