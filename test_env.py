"""Correctness tests for the vectorized cube engine.

Group-theory identities pin down the move tables almost completely; the
sticker chirality checks pin down clockwise-vs-counterclockwise; the 2x2 BFS
state count (run separately in bfs_2x2.py) seals it.
"""
import torch
from cube_env import CubeEnv, FACE_NAMES

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def apply_seq(env, state, names, times=1):
    s = state
    for _ in range(times):
        s = env.apply_names(s, names)
    return s


def seq_order(env, names, max_order=2000):
    """Order of a move sequence in the cube group."""
    s0 = env.solved.clone()
    s = s0
    for k in range(1, max_order + 1):
        s = env.apply_names(s, names)
        if torch.equal(s, s0):
            return k
    raise AssertionError(f"order of {names} exceeds {max_order}")


def run():
    for n in (2, 3):
        faces = ["U", "R", "F"] if n == 2 else FACE_NAMES
        env = CubeEnv(n, DEV)
        s0 = env.solved.clone()

        # every quarter turn has order 4
        for f in faces:
            assert seq_order(env, [f]) == 4, f"{n}x{n} {f} order != 4"
            assert seq_order(env, [f + "'"]) == 4

        # move then inverse = identity
        for f in faces:
            s = apply_seq(env, s0, [f, f + "'"])
            assert torch.equal(s, s0), f"{n}x{n} {f} inverse broken"

        # inverse_action mapping is consistent with the tables
        B = 64
        st = env.scramble(B, 20)
        a = torch.randint(env.M, (B,), device=DEV)
        back = env.step(env.step(st, a), env.inverse_action[a])
        assert torch.equal(back, st), f"{n}x{n} inverse_action broken"

        # sexy move (R U R' U') has order 6
        assert seq_order(env, ["R", "U", "R'", "U'"]) == 6, f"{n}x{n} sexy move order != 6"

        # chirality: after U on solved, F top row shows R's color (1);
        # after R on solved, F right column shows D's color (3)
        F0 = 2 * n * n  # start index of F face
        sU = apply_seq(env, s0, ["U"])
        assert (sU[F0:F0 + n] == 1).all(), f"{n}x{n} U chirality wrong"
        sR = apply_seq(env, s0, ["R"])
        right_col = [F0 + r * n + (n - 1) for r in range(n)]
        assert (sR[right_col] == 3).all(), f"{n}x{n} R chirality wrong"

        # commuting opposite faces (3x3): R L == L R
        if n == 3:
            assert torch.equal(apply_seq(env, s0, ["R", "L"]), apply_seq(env, s0, ["L", "R"]))
            # (R U) has order 105 in the 3x3 cube group — strong global check
            assert seq_order(env, ["R", "U"]) == 105, "3x3 (R U) order != 105"
        else:
            # (R U) on 2x2: corners only -> order 15
            o = seq_order(env, ["R", "U"])
            assert o == 15, f"2x2 (R U) order {o} != 15"

        # scramble depths honored: depth-0 rows stay solved, deep rows don't
        depths = torch.tensor([0, 1, 20] * 10, device=DEV)
        st = env.scramble(30, depths)
        assert env.is_solved(st)[::3].all(), "depth-0 scramble not solved"
        assert not env.is_solved(st)[2::3].any(), "depth-20 scrambles all solved (?)"

        # neighbors == step with each action
        st = env.scramble(8, 15)
        nb = env.neighbors(st)
        for m in range(env.M):
            am = torch.full((8,), m, dtype=torch.long, device=DEV)
            assert torch.equal(nb[:, m], env.step(st, am))

        print(f"{n}x{n}: all engine tests passed ({env.M} moves, {env.S} stickers)")

    # 2x2 pack/unpack roundtrip
    from cube_env import pack_2x2, unpack_2x2
    env2 = CubeEnv(2, DEV)
    st = env2.scramble(10000, 14)
    assert torch.equal(unpack_2x2(pack_2x2(st)), st), "2x2 pack/unpack roundtrip failed"
    print("2x2 pack/unpack roundtrip passed")


if __name__ == "__main__":
    run()
