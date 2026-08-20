#!/bin/bash
# Idempotent, restart-proof training queue. Skips runs whose final iter exists.
PY=/home/akhani/miniconda3/envs/pytorch_r8/bin/python
cd /c/q/RLE_rubiks_cube/RLE_rubiks_cube__r1
fin() { grep -q "\"iter\": $2," runs/$1/metrics.jsonl 2>/dev/null; }
res() { [ -f runs/$1/ckpt_latest.pt ] && echo "--resume runs/$1/ckpt_latest.pt"; }

fin pomdp_davi_s2 60000 || $PY -u domains/train_maze.py --method davi --iters 60000 --out runs/pomdp_davi_s2 --seed 2 --radius 3 $(res pomdp_davi_s2)
fin peg_davi_s0 30000 || $PY -u domains/train_misc.py --domain pegs --method davi --iters 30000 --out runs/peg_davi_s0 --seed 0 $(res peg_davi_s0)
fin peg_davi_s1 12000 || $PY -u domains/train_misc.py --domain pegs --method davi --iters 12000 --out runs/peg_davi_s1 --seed 1 $(res peg_davi_s1)
fin peg_davi_s2 12000 || $PY -u domains/train_misc.py --domain pegs --method davi --iters 12000 --out runs/peg_davi_s2 --seed 2 $(res peg_davi_s2)
echo "MUSTHAVES COMPLETE"
bash finalize.sh || true
for s in 1 2; do
  fin slide3_diff_s$s 20000 || $PY -u domains/train_slide.py --method denoise --n 3 --iters 20000 --h1 1024 --h2 512 --blocks 2 --no-compile --out runs/slide3_diff_s$s --seed $s
  fin slide3_davi_s$s 20000 || $PY -u domains/train_slide.py --method davi --n 3 --iters 20000 --h1 1024 --h2 512 --blocks 2 --no-compile --out runs/slide3_davi_s$s --seed $s
  fin pend_diff_s$s 30000  || $PY -u domains/pendulum.py --method denoise --iters 30000 --out runs/pend_diff_s$s --seed $s
  fin pend_disc_s$s 30000  || $PY -u domains/pendulum.py --method denoise-disc --iters 30000 --out runs/pend_disc_s$s --seed $s
  fin pend_value_s$s 30000 || $PY -u domains/pendulum.py --method value --iters 30000 --out runs/pend_value_s$s --seed $s
  fin g2048_diff_s$s 40000 || $PY -u domains/train_2048.py --iters 40000 --out runs/g2048_diff_s$s --seed $s
  fin maze_diff_s$s 60000  || $PY -u domains/train_maze.py --method denoise --iters 60000 --out runs/maze_diff_s$s --seed $s $(res maze_diff_s$s)
  fin maze_davi_s$s 60000  || $PY -u domains/train_maze.py --method davi --iters 60000 --out runs/maze_davi_s$s --seed $s $(res maze_davi_s$s)
  fin soko_diff_s$s 80000  || $PY -u domains/train_sokoban.py --method denoise --iters 80000 --out runs/soko_diff_s$s --seed $s $(res soko_diff_s$s)
  fin soko_davi_s$s 80000  || $PY -u domains/train_sokoban.py --method davi --iters 80000 --out runs/soko_davi_s$s --seed $s $(res soko_davi_s$s)
done
echo "QUEUE4 COMPLETE"
bash finalize.sh || true
echo "ALL DONE"
