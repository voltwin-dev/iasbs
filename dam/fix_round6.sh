#!/usr/bin/env bash
# DAM divergence-stabiliser sweep, round 6 (2026-09-07 23:40 UTC).
#
# Round 5 result: the repaired DAM gets within touching distance of the gate but
# does not cross it inside the budget given.
#
#   J  K=32, 5000 it: TV 0.129 (1500) -> 0.0824 (3500) -> 0.06168,  ESS 30.47/32
#                     168,960,000 f1 evals, 974.8 M jumps, 11,428 s
#   L  K=64, 2500 it: TV 0.125 (1500) -> 0.0798 (2250) -> 0.08103,  ESS 58.62/64
#                     166,400,000 f1 evals, 924.0 M jumps,  6,950 s
#
# Two things follow.  First, the K axis has saturated: L spends the same f1
# budget as J and lands no better (0.081 vs 0.062), so beyond K=32 the extra
# rollouts buy retention that is already at 92-95% and nothing else.  Second, J
# is still descending at its cutoff -- 0.06435 -> 0.06168 over its last 250
# iterations -- so the gate is a budget question, not a wall.  Linear
# extrapolation of that rate puts TV = 0.05 at roughly iteration 6100.
#
# Round 6, ~4.4 h per arm, one GPU each:
#   M  K=32, 7000 iterations.  Runs the extrapolation out with ~15% headroom.
#      If it crosses, the honest claim becomes "DAM does reach the gate on Ising
#      L=4 once repaired, for ~2.4e8 rollout-endpoint evaluations", which is a
#      stronger and fairer statement than "it fails".
#   N  occupation-scale m=32, K=64, 1000 iterations.  The repair has only ever
#      been tested on Ising L=4.  m=32 is the OTHER benchmark that diverged
#      (at it 10 for K=16 and it 15 for K=64, with 153 jumps per f1 eval and
#      ESS p10 pinned at 1.00), so this is the transfer test: does the same
#      control box and truncation fix a different failure, or was it tuned to
#      one problem?  Note the clamp cannot be sized against an exact optimum
#      here -- ScaleOccupation has no ExactControl -- which is precisely the
#      practical difficulty the revised claim 6 asserts.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/SML_env/bin/python
STAB="--a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10"

run() {
    local gpu="$1" tag="$2"; shift 2
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m dam.discrete "$@" $STAB \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "dam/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END $tag exit=$?"
}

run 0 fix_M_K32_7000 ising --L 4 --K 32 --iters 7000 --steps 128 --inner 4 \
    --mb 256 --batch 512 --eval-every 250 --n-samples 20000 &

run 1 fix_N_occs32 occupation-scale --m 32 --K 64 --iters 1000 --steps 128 \
    --inner 4 --mb 256 --batch 512 --eval-every 25 --n-samples 10000 &
wait
echo "=== [$(date -u '+%F %T UTC')] ROUND6 DONE"
