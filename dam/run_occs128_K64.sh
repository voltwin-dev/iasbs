#!/usr/bin/env bash
# Round 8: rerun the round-7 arm O with enough rollouts.
#
# Round 7 ran occupation-scale m=128 at K=32 and starved the adjoint estimator:
# ESS 1.52/32 at iteration 25, with KS_occ 0.2083 against an uncontrolled
# reference of 0.2053, i.e. the control was actively harmful.  Arm N already
# needed K=64 to hold m=32, so K=32 on a 4x larger space was a misconfiguration
# on our side.  This reruns it at K=64.
#
# Arm N converged by iteration ~425 of 1000, so 500 iterations is budgeted here.
# At m=128 the per-iteration cost is ~5.4x arm N's, and doubling K doubles it
# again, so expect ~178 s/iter and ~25 h.
set -u
PY=/root/miniconda3/envs/SML_env/bin/python
STAB="--a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10"

echo "=== [$(date -u '+%F %T UTC')] starting round 8"
CUDA_VISIBLE_DEVICES=0 "$PY" -u -m dam.discrete occupation-scale \
    --m 128 --K 64 --iters 500 --steps 128 --inner 4 --mb 256 --batch 512 \
    --eval-every 25 --n-samples 10000 $STAB \
    --tag fix_O2_occs128_K64 --out json/results_fix_O2_occs128_K64.json \
    > dam/logs/fix_O2_occs128_K64.log 2>&1
echo "=== [$(date -u '+%F %T UTC')] END fix_O2_occs128_K64 exit=$?"
