# rubiks-diffusion — cube solving as discrete denoising diffusion

Companion code for the paper *"Scramble Inversion as Discrete Denoising
Diffusion: A Matched-Compute Study on the Rubik's Cube Group"*
([`paper/main.pdf`](paper/main.pdf)).

Two learned solvers trained and compared under matched compute on one RTX 5080
Laptop GPU, with exhaustive exact-oracle validation:

| | DAVI (value iteration) | Denoising (diffusion-style) |
|---|---|---|
| 2x2: all 3,674,160 states solved | 100% (26 min train) | 100% (6.5 min train), 99.989% greedy-only |
| 3x3: 1000 hundred-move scrambles | 1000/1000 (13.9 h train) | 1000/1000 (4.9 h train), 349 with no search |

Every solution is replay-verified in the environment.

## Vectorized GPU cube engine (2x2 + 3x3)

Reinforcement-learning cube solver trained from scratch on a single RTX 5080 Laptop GPU.
Massively parallel cube simulation in pure torch: every face turn is a fixed permutation
of sticker indices, so stepping N cubes is one `gather` kernel (billions of moves/sec).

## Approach

- **Engine** (`cube_env.py`): `[B, S]` int8 sticker states, move tables generated
  geometrically (rotate sticker positions/normals, re-match) — no hand-typed cycles.
  2x2 uses U/R/F quarter turns only (fixes the DLB corner: unique solved state,
  exactly 7!·3^6 = 3,674,160 reachable states). 3x3 uses all 12 quarter turns.
- **Learning** (`davi.py`, `model.py`): DAVI — deep approximate value iteration
  (DeepCubeA-style). Cost-to-go net V(s) trained on reverse scrambles with targets
  `y(s) = 1 + min_a V_target(step(s,a))`; target net syncs when loss < 0.05.
- **Solving** (`solve.py`): batched greedy descent on V + batched beam search.
  **Every solution is replay-verified** by re-applying the recorded moves in the
  environment and requiring exact arrival at the solved state.
- **Exact oracle** (`bfs_2x2.py`): full-state-space GPU BFS for the 2x2
  (3,674,160 states, QTM diameter 14, 0.6 s) — validates both the engine
  (state count matches group order) and the learned value function.

## Files

| file | purpose |
|---|---|
| `cube_env.py` | vectorized env, move-table generation, pack/unpack |
| `test_env.py` | engine tests: move order 4, inverses, sexy-move order 6, (R U) order 105, chirality |
| `bfs_2x2.py` | exact 2x2 distance oracle (full BFS) |
| `model.py` | residual MLP value net (LayerNorm variant of DeepCubeA arch) |
| `davi.py` | DAVI trainer (`--preset 2x2` / `--preset 3x3`) |
| `solve.py` | greedy / beam / batched-beam solvers + replay verification |
| `eval_2x2.py` | full validation vs oracle (all 3,674,160 states) |
| `eval_3x3.py` | solve-rate eval on fully scrambled 3x3 cubes |
| `bench.py` | engine throughput benchmark |
| `viz_traces.py`, `make_viz.py`, `viz_template.html` | interactive visualizer (3D playback + telemetry) |

## Reproduce

```bash
PY=~/miniconda3/envs/pytorch_r8/bin/python
$PY test_env.py                                   # engine tests
$PY bfs_2x2.py                                    # exact oracle (seconds)
$PY davi.py --preset 2x2 --out runs/2x2_v1        # ~13 min on 5080 Laptop
$PY eval_2x2.py --ckpt runs/2x2_v1/ckpt_latest.pt --out runs/2x2_v1/eval_report.json
$PY davi.py --preset 3x3 --out runs/3x3_v1        # hours (checkpoints + resume)
$PY eval_3x3.py --ckpt runs/3x3_v1/ckpt_latest.pt --out runs/3x3_v1/eval_report.json
$PY make_viz.py --traces2 ... --metrics2 ...      # build viz/index.html
```
