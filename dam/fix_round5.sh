#!/usr/bin/env bash
# DAM divergence-stabiliser sweep, round 5 (2026-09-07 19:15 UTC) -- run to the gate.
#
# Round 4 settled it: the stabilised runs converge, and rounds 1-3 were simply
# starved of iterations.  Both arms descend monotonically with a POSITIVE loss
# and a rising ESS for the whole run:
#
#   H  K=16, 3000 it: TV 0.769 (it 100) -> 0.399 (2000) -> 0.203 (3000),
#                     ESS 5.66 -> 11.19 / 16,  6488 s
#   I  K=32, 1500 it: TV 0.685 (it  300) -> 0.216 ( 800) -> 0.129 (1500),
#                     ESS 4.71 -> 26.01 / 32,  3493 s
#
# I beats H to a better TV in HALF the wall clock, so the ESS ceiling was
# binding as well as the budget: at K=16 the damped arm retained 70% of its
# rollouts by the end, at K=32 it retained 81%, and the extra retention pays for
# the doubled per-iteration cost twice over.  Neither run had flattened at its
# cutoff -- I was still falling 0.0110 per 100 iterations over its last 200.
#
# Round 5 runs the K axis out to the gate (TV <= 0.05) at matched wall clock,
# ~3.2 h per arm on one GPU each:
#   J  K=32, 5000 iterations  -- the extrapolation from I says the gate is
#      reachable in roughly this budget.
#   L  K=64, 2500 iterations  -- if retention keeps buying more than it costs,
#      the K axis has not saturated and L beats J at equal cost.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/SML_env/bin/python
STAB="--a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10"
COMMON="ising --L 4 --steps 128 --inner 4 --mb 256 --batch 512 \
        --eval-every 250 --n-samples 20000"

run() {
    local gpu="$1" tag="$2"; shift 2
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m dam.discrete $COMMON $STAB "$@" \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "dam/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END $tag exit=$?"
}

run 0 fix_J_K32_5000 --K 32 --iters 5000 &
run 1 fix_L_K64_2500 --K 64 --iters 2500 &
wait
echo "=== [$(date -u '+%F %T UTC')] ROUND5 DONE"
