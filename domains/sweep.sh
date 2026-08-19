#!/bin/bash
# Seed-variance + new-domain sweep. Sequential; each line logs its own dir.
set -x
PY=/home/akhani/miniconda3/envs/pytorch_r8/bin/python
cd /c/q/RLE_rubiks_cube/RLE_rubiks_cube__r1

# Domain 6: Hanoi, 3 seeds x 2 methods
for s in 0 1 2; do
  $PY -u domains/train_hanoi.py --method denoise --iters 20000 --out runs/hanoi_diff_s$s --seed $s
  $PY -u domains/train_hanoi.py --method davi    --iters 20000 --out runs/hanoi_davi_s$s --seed $s
done

# Domain 7: POMDP maze (radius 3), 3 seeds x 2 methods
for s in 0 1 2; do
  $PY -u domains/train_maze.py --method denoise --iters 60000 --out runs/pomdp_diff_s$s --seed $s --radius 3
  $PY -u domains/train_maze.py --method davi    --iters 60000 --out runs/pomdp_davi_s$s --seed $s --radius 3
done

# Extra seeds for existing domains (seed 0 = original runs)
for s in 1 2; do
  $PY -u domains/train_slide.py --method denoise --n 3 --iters 20000 --h1 1024 --h2 512 --blocks 2 --no-compile --out runs/slide3_diff_s$s --seed $s
  $PY -u domains/train_slide.py --method davi    --n 3 --iters 20000 --h1 1024 --h2 512 --blocks 2 --no-compile --out runs/slide3_davi_s$s --seed $s
  $PY -u domains/pendulum.py --method denoise      --iters 30000 --out runs/pend_diff_s$s --seed $s
  $PY -u domains/pendulum.py --method denoise-disc --iters 30000 --out runs/pend_disc_s$s --seed $s
  $PY -u domains/pendulum.py --method value        --iters 30000 --out runs/pend_value_s$s --seed $s
  $PY -u domains/train_2048.py --iters 40000 --out runs/g2048_diff_s$s --seed $s
  $PY -u domains/train_sokoban.py --method denoise --iters 80000 --out runs/soko_diff_s$s --seed $s
  $PY -u domains/train_sokoban.py --method davi    --iters 80000 --out runs/soko_davi_s$s --seed $s
  $PY -u domains/train_maze.py --method denoise --iters 60000 --out runs/maze_diff_s$s --seed $s
  $PY -u domains/train_maze.py --method davi    --iters 60000 --out runs/maze_davi_s$s --seed $s
done
echo "SWEEP COMPLETE"
