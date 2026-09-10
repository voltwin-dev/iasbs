#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # repo root: json/, ckpt/ and fig/ live here
# Final attempt at a beta=100 control that is usable on a *coarse* grid.
#
# What we know now.  The 796-step run learned a control that is essentially
# correct -- integrated on 3184 steps it gives dE = 0.046, better than R-ASBS's
# 0.234 -- while its own training samples sat off-mode at E = 6.5.  So learning
# was never the problem, and neither was the loss scale.  The problem is a
# transient: out_scale is ~0.55*beta when the sampler is on the mode (beta=50
# settled at 27.4) but ~2.2*beta when it is off (beta=100 sat at 223).  A single
# 50 -> 100 jump forces the control to grow through that off-mode regime, where
# h*score = 223/796 = 0.28 is four times the 0.069 margin that kept beta=50
# stable, so the integrator loses the control and collection collapses.
#
# The fix is to never make that jump: anneal 50 -> 65 -> 80 -> 100 so the
# control grows in increments small enough that the sampler stays on the mode
# and out_scale stays near 0.55*beta.  At beta=100 that predicts
# h*score = 55/796 = 0.069 -- exactly the beta=50 margin.
#
# Intermediate legs are short (600 iters, warm starts have little to learn) and
# evaluate cheaply; only the final leg gets the full sampling budget.  No
# `set -e`: a failed gate exits nonzero and that is expected here.
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=1

BASE="--sigma 1.4142135623730951 --nq 64 --inner 8 --batch 2048 --mb 16384 \
    --hidden 256 --lr 0.001 --ema 0.9995 --nbuf 4 --seeds 1 --eval-every 200 \
    --seed 0 --verbose --antithetic --steps 796 --ckpt-dir ckpt"
CHEAP="--n-samples 20000 --mc-moment 50000 --mcmc-chains 50000 \
    --mcmc-sweeps 1000 --mcmc-eps 0.35 --refine 2"

prev=ckpt/stiefel_anneal_b50_seed0.pt
for b in 65 80; do
  echo "########## leg beta=$b (warm start $prev) ##########"
  $PY -u iasbs/stiefel.py train --betas $b $BASE $CHEAP --iters 600 \
      --tag chain --init-from $prev --out json/results_chain_b$b.json
  prev=ckpt/chain_b${b}_seed0.pt
done

echo "########## final leg beta=100 (warm start $prev) ##########"
$PY -u iasbs/stiefel.py train --betas 100 $BASE --iters 900 \
    --n-samples 100000 --mc-moment 200000 --mcmc-chains 200000 \
    --mcmc-sweeps 3000 --mcmc-eps 0.35 --refine 2,4 \
    --tag chain --init-from $prev --out json/results_chain_b100.json
