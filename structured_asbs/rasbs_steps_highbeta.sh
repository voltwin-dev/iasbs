#!/usr/bin/env bash
# Fairness control for the high-beta write-up.
#
# Our beta = 50 / 100 error shrinks under grid refinement (4.39 -> 0.010 at
# beta = 50, 5.29 -> 0.046 at beta = 100), so it is discretisation error.  We
# want to say that R-ASBS's error at the same betas is a *bias floor* instead,
# which grid refinement does not remove -- but the only evidence we have for
# that is their step sweep at beta = 2 (0.547 at 199 steps -> 0.492 at 512,
# extrapolating to ~0.46).  Inferring the same behaviour at beta = 50 and 100
# without measuring it would be exactly the kind of flattering assumption this
# repository is supposed to avoid.  So: measure it.
#
# 100k samples to match the main run, at 199 / 512 / 1024 steps.
PY=/root/miniconda3/envs/SML_env/bin/python
export CUDA_VISIBLE_DEVICES=0
for N in 199 512 1024; do
  $PY -u rasbs_port.py --betas 50,100 --steps $N --epochs 1000 \
      --n-samples 100000 --out json/results_rasbs_highbeta_steps_$N.json
done
