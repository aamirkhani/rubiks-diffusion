#!/bin/bash
# One-shot finisher: aggregate -> numbers -> figures -> galleries -> compile -> push.
set -e
PY=/home/akhani/miniconda3/envs/pytorch_r8/bin/python
cd /c/q/RLE_rubiks_cube/RLE_rubiks_cube__r1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY -u domains/aggregate_all.py
$PY paper2/gen_numbers.py
$PY -c "import sys; sys.path.insert(0,'paper2'); import figures2 as f; f.fig_summary(); f.fig_slide24_training(); f.fig_sokoban_depth(); f.fig_pendulum_bars(); f.fig_domains_strip(); f.fig_architecture2()"
$PY paper2/fig_process_gallery.py
cp paper2/fig2_gallery_A.png paper2/fig2_gallery_B.png paper2/fig2_architecture.png paper2/png/ 2>/dev/null || true
cd paper2
pdflatex -interaction=nonstopmode -jobname=scramble-inversion-beyond-groups main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode -jobname=scramble-inversion-beyond-groups main.tex 2>&1 | grep -E "^!|Output written" | head -3
cd ..
git add -A
git commit -m "Finalize paper 2: aggregated seeds, final numbers, complete galleries

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true
git push
echo "FINALIZE DONE"
