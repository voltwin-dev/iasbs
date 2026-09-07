#!/bin/bash
cd "$(dirname "$0")/.."   # repo root: json/, ckpt/ and fig/ live here
# Step-count sweep: does R-ASBS show a nonzero geometric-surrogate floor as the
# integration step count grows?  Measured at beta=2, where their error against
# the MCMC reference is largest (+0.547 at N=199).
#
# --ckpt-dir/--tag are mandatory here, not optional: the first version of this
# script omitted them, so the five controls were discarded at exit and the
# sweep had to be re-run from scratch to answer a follow-up question.
set -x
PY=${PY:-python}
for N in 32 64 128 256 512; do
  $PY -u rasbs/rasbs_port.py --betas 2 --steps $N --epochs 1000 \
      --n-samples 100000 --ckpt-dir ckpt --tag rasbs_steps$N \
      --out json/results_rasbs_steps_$N.json
done
