#!/usr/bin/env bash
# Ising L=5 DAM, rerun with a budget that clears the transient.
#
# The 1200-iteration attempt ended at TV 0.80981 with the loss still negative
# (-5.31) and the ESS still rising (6.27 -> 8.14 / 32), i.e. inside the same
# transient that made rounds 1-3 misread the L=4 runs.  Ising L=4 needed 3000
# iterations at K=16 and plateaued only past 5000 at K=32; L=5 is 404x larger,
# so 5000 is the floor for a conclusive answer.  ~3.1 s/iter, ~4.3 h.
set -u
PY=/root/miniconda3/envs/SML_env/bin/python
STAB="--a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10"

echo "=== [$(date -u '+%F %T UTC')] starting Ising L=5 long run"
CUDA_VISIBLE_DEVICES=1 "$PY" -u -m dam.discrete ising \
    --L 5 --K 32 --iters 5000 --steps 128 --inner 4 --mb 256 --batch 512 \
    --eval-every 250 --n-samples 20000 $STAB \
    --tag fix_isingL5_5000 --out json/results_fix_isingL5_5000.json \
    > dam/logs/fix_isingL5_5000.log 2>&1
echo "=== [$(date -u '+%F %T UTC')] END fix_isingL5_5000 exit=$?"
