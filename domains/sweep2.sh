#!/bin/bash
# New domains 8-10 (seed 0 first, then seeds 1,2) + evals baked into probes.
set -x
PY=/home/akhani/miniconda3/envs/pytorch_r8/bin/python
cd /c/q/RLE_rubiks_cube/RLE_rubiks_cube__r1
for s in 0 1 2; do
  $PY -u domains/train_misc.py --domain lightsout --method denoise --iters 20000 --out runs/lo_diff_s$s --seed $s
  $PY -u domains/train_misc.py --domain lightsout --method davi    --iters 20000 --out runs/lo_davi_s$s --seed $s
  $PY -u domains/mountaincar.py --method denoise --iters 20000 --out runs/mc_diff_s$s --seed $s
  $PY -u domains/mountaincar.py --method value   --iters 20000 --out runs/mc_value_s$s --seed $s
  $PY -u domains/train_misc.py --domain pegs --method denoise --iters 30000 --out runs/peg_diff_s$s --seed $s
  $PY -u domains/train_misc.py --domain pegs --method davi    --iters 30000 --out runs/peg_davi_s$s --seed $s
done
echo "SWEEP2 COMPLETE"
