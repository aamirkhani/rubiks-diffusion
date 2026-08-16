"""Throughput benchmark: batched cube moves/sec on GPU."""
import time
import torch
from cube_env import CubeEnv

DEV = "cuda"

for n in (2, 3):
    env = CubeEnv(n, DEV)
    for B in (100_000, 1_000_000):
        states = env.scramble(B, 10)
        actions = torch.randint(env.M, (B,), device=DEV)
        for _ in range(3):
            states = env.step(states, actions)
        torch.cuda.synchronize()
        t0 = time.time()
        REPS = 200
        for _ in range(REPS):
            states = env.step(states, actions)
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"{n}x{n}  batch {B:>9,}: {B*REPS/dt/1e9:6.2f} B moves/sec  "
              f"({dt/REPS*1e6:.0f} us/step)")
