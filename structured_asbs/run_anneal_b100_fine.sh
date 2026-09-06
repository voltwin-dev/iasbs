#!/usr/bin/env bash
# beta=100, third attempt.  The 398-step annealed run diverged *away* from a
# good warm start (E climbed 3.1 -> 7.7 by iteration 250), and the measured
# out_scale says why: it settled at 2.27e+02, so h*score = 226/398 = 0.57 per
# Euler step.  The beta=50 run that worked settled at 2.74e+01, i.e.
# h*score = 27.4/398 = 0.069, eight times smaller.  The warm start starts on
# the mode, but as training pushes the output up towards the beta=100 target
# the 398-step integrator stops being able to follow it, collection collapses,
# and the control is dragged off the mode.
#
# Doubling to 796 steps restores roughly the beta=50 margin.  Evaluation
# refines to 1592 and 3184.  If this still fails, the failure is a real
# property of the h-transform at sharp targets and not a budget we forgot to
# spend.  No `set -e`: a failed gate exits nonzero and that is expected here.
PY=/root/miniconda3/envs/SML_env/bin/python
export CUDA_VISIBLE_DEVICES=1
$PY -u stiefel.py train --betas 100 \
    --sigma 1.4142135623730951 --nq 64 --iters 1500 --inner 8 \
    --batch 2048 --mb 16384 --hidden 256 --lr 0.001 --ema 0.9995 --nbuf 4 \
    --seeds 1 --eval-every 250 --n-samples 100000 --mc-moment 200000 \
    --mcmc-chains 200000 --mcmc-sweeps 3000 --mcmc-eps 0.35 --seed 0 \
    --verbose --antithetic --steps 796 --refine 2,4 --ckpt-dir ckpt \
    --tag stiefel_anneal_fine --init-from ckpt/stiefel_anneal_b50_seed0.pt \
    --out json/results_stiefel_anneal_b100_fine.json
