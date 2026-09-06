#!/usr/bin/env bash
# beta=100 leg of the annealing chain, warm-started from the beta=50 control
# that run_anneal.sh just produced (dE = 0.0685 at 398 steps).  Split out
# because run_train exits nonzero on a failed gate, which killed the original
# two-leg script after beta=50.  No `set -e` here for the same reason.
PY=/root/miniconda3/envs/SML_env/bin/python
export CUDA_VISIBLE_DEVICES=1
$PY -u stiefel.py train --betas 100 \
    --sigma 1.4142135623730951 --nq 64 --iters 1500 --inner 8 \
    --batch 2048 --mb 16384 --hidden 256 --lr 0.001 --ema 0.9995 --nbuf 4 \
    --seeds 1 --eval-every 250 --n-samples 100000 --mc-moment 200000 \
    --mcmc-chains 200000 --mcmc-sweeps 3000 --mcmc-eps 0.35 --seed 0 \
    --verbose --antithetic --steps 398 --refine 2,4 --ckpt-dir ckpt \
    --tag stiefel_anneal --init-from ckpt/stiefel_anneal_b50_seed0.pt \
    --out json/results_stiefel_anneal_b100.json
