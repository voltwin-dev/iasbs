#!/usr/bin/env bash
# Occupation-scale m=1000 with DAM -- the one IASBS benchmark DAM has never been
# run on.  Earlier it was dropped as "a foregone divergence", which is an
# assumption, not a measurement.  This measures it.
#
# K=64 because m=32 already needed K=64 to hold, and m=1000 is far larger.
# Cost warning: measured per-iteration cost is 16.6 s at m=32 K=64 and ~178 s at
# m=128 K=64, i.e. 10.7x for a 4x space.  Extrapolating puts m=1000 near
# 3,500 s/iter, so this is budgeted at 200 iterations and is expected to be read
# early and killed rather than run to completion.  Whatever it reaches is a
# result: either DAM converges here, or the cost of finding out is itself the
# finding.
#
# Waits for the Ising L=5 run to free GPU1.
set -u
PY=/root/miniconda3/envs/SML_env/bin/python
STAB="--a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10"

echo "=== [$(date -u '+%F %T UTC')] waiting for fix_isingL5_5000 to finish"
while pgrep -f "tag fix_isingL5_5000" > /dev/null; do sleep 60; done
echo "=== [$(date -u '+%F %T UTC')] GPU1 clear, starting occupation m=1000"

CUDA_VISIBLE_DEVICES=1 "$PY" -u -m dam.discrete occupation-scale \
    --m 1000 --K 64 --iters 200 --steps 128 --inner 4 --mb 256 --batch 512 \
    --eval-every 10 --n-samples 4000 $STAB \
    --tag fix_occs1000_K64 --out json/results_fix_occs1000_K64.json \
    > dam/logs/fix_occs1000_K64.log 2>&1
echo "=== [$(date -u '+%F %T UTC')] END fix_occs1000_K64 exit=$?"
