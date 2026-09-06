#!/usr/bin/env bash
# The two remaining hypotheses for the beta >= 50 failure, tested together.
#
#   (C) time discretisation: the score is O(beta) and the Euler step was 1/199,
#       so h*score was O(1) per step.  Training now runs on 398 steps and is
#       evaluated out to 1592.
#   (B) on-policy collapse: collection simulates the current control, so a
#       control that never reaches the mode never generates labels near it.
#       Each beta is warm-started from the next one down -- 20 -> 50 -> 100 --
#       which is why these are two sequential invocations rather than one
#       --betas 50,100 run.
#
# Everything else matches the stiefel_fill sweep so the comparison is clean.
# NOTE: run_train exits nonzero when a gate fails, and these betas are
# exactly the ones we expect to miss the 0.05 gate, so `set -e` would kill
# the chain after the first one.  Failures are reported, not fatal.
PY=/root/miniconda3/envs/SML_env/bin/python
export CUDA_VISIBLE_DEVICES=1

COMMON="--sigma 1.4142135623730951 --nq 64 --iters 1500 --inner 8 \
    --batch 2048 --mb 16384 --hidden 256 --lr 0.001 --ema 0.9995 --nbuf 4 \
    --seeds 1 --eval-every 250 --n-samples 100000 --mc-moment 200000 \
    --mcmc-chains 200000 --mcmc-sweeps 3000 --mcmc-eps 0.35 --seed 0 \
    --verbose --antithetic --steps 398 --refine 2,4 --ckpt-dir ckpt \
    --tag stiefel_anneal"

echo "########## beta=50, warm start from the beta=20 control ##########"
$PY -u stiefel.py train --betas 50 $COMMON \
    --init-from ckpt/stiefel_fill_b20_seed0.pt \
    --out json/results_stiefel_anneal_b50.json

echo "########## beta=100, warm start from the beta=50 control ##########"
$PY -u stiefel.py train --betas 100 $COMMON \
    --init-from ckpt/stiefel_anneal_b50_seed0.pt \
    --out json/results_stiefel_anneal_b100.json
