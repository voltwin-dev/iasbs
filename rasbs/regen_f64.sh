#!/usr/bin/env bash
cd "$(dirname "$0")/../structured_asbs"   # json/ and ckpt/ live there
# Regenerate the beta = 2 checkpoints in float64 so figure 7 can draw the
# orthogonality residual from disk instead of skipping it.  Same configuration
# and seed as the runs that produced the headline numbers; only the storage
# dtype changes.
set -e
PY=/root/miniconda3/envs/SML_env/bin/python
export CUDA_VISIBLE_DEVICES=1

$PY -u ../rasbs/rasbs_port.py --betas 2 --tag rasbs --ckpt-dir ckpt \
    --out json/results_rasbs_b2_f64.json

$PY -u stiefel.py train --betas 2 \
    --sigma 1.4142135623730951 --steps 199 --nq 64 --iters 1500 --inner 8 \
    --batch 2048 --mb 16384 --hidden 256 --lr 0.001 --ema 0.9995 --nbuf 4 \
    --seeds 1 --eval-every 500 --n-samples 100000 --mc-moment 200000 \
    --mcmc-chains 200000 --mcmc-sweeps 3000 --mcmc-eps 0.35 --seed 0 --verbose \
    --antithetic --refine 2,4 --ckpt-dir ckpt --tag stiefel_grid \
    --out json/results_stiefel_b2_f64.json
