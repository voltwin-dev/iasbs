#!/usr/bin/env bash
cd "$(dirname "$0")/../.."   # repo root: json/, ckpt/ and fig/ live here
# Rerun the two betas that diverged (50, 100) with the scale-free label
# parametrisation of ScoreNet.  Every other flag is byte-identical to the
# stiefel_fill run recorded in json/results_stiefel_fill.json, so the before /
# after comparison isolates the fix.
set -e
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=1
$PY -u iasbs/stiefel.py train --betas 50,100 \
    --sigma 1.4142135623730951 --steps 199 --nq 64 --iters 1500 --inner 8 \
    --batch 2048 --mb 16384 --hidden 256 --lr 0.001 --ema 0.9995 --nbuf 4 \
    --seeds 1 --eval-every 500 --n-samples 100000 --mc-moment 200000 \
    --mcmc-chains 200000 --mcmc-sweeps 3000 --mcmc-eps 0.35 --seed 0 --verbose \
    --antithetic --refine 2,4 --ckpt-dir ckpt --tag stiefel_scalefix \
    --out json/results_stiefel_scalefix.json
