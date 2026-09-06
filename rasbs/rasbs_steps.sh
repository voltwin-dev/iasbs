#!/bin/bash
cd "$(dirname "$0")/../structured_asbs"   # json/ and ckpt/ live there
# PLAN 9.3: does R-ASBS show a nonzero geometric-surrogate floor as the
# integration step count grows?  Measured at beta=2, where their error against
# the MCMC reference is largest (+0.547 at N=199).
set -x
PY=/root/miniconda3/envs/SML_env/bin/python
for N in 32 64 128 256 512; do
  $PY -u ../rasbs/rasbs_port.py --betas 2 --steps $N --epochs 1000 \
      --n-samples 100000 --out json/results_rasbs_steps_$N.json
done
