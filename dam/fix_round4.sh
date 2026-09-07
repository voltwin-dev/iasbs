#!/usr/bin/env bash
# DAM divergence-stabiliser sweep, round 4 (2026-09-07 17:20 UTC) -- the budget test.
#
# Round 3 verdict: raising the clamp to 8 (above the exact optimum of 6.31) does
# what it was predicted to do -- the arm with the round-1 damping reaches
# ESS 4.32/16 mean and 5.68 peak, against 2.35 at clamp 3 and 1.04 for the
# diverging baseline -- and drives the energy histogram to 0.157.  The full TV
# still does not descend: 0.817 (G) and ~0.77 (F) at iteration 600.
#
# Every arm so far has been given 600 iterations.  That is very likely just
# short.  The one DAM leg that converges, occupation m=4, needed 1200
# iterations at K=16 to reach TV 0.0149 and was still at 0.39 at iteration 200
# on a state space of |X| = 35.  Ising L=4 has |Omega| = 12,870, i.e. 368x
# larger, and has been given half the budget of the small problem.  Round 4
# removes that objection.
#
# Two arms at matched wall clock (~110 min each, one GPU each):
#   H  the round-3 G configuration, K=16, 3000 iterations  -- 5x the budget.
#   I  the same configuration at K=32, 1500 iterations     -- tests whether the
#      ESS ceiling is the binding constraint instead of the budget.  Round 3's
#      best arm still only retained 4.32 of 16 rollouts (27%); if that is what
#      blocks the descent then doubling K should beat H at equal wall clock,
#      and if the budget is what blocks it then H should win.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/SML_env/bin/python
STAB="--a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10"
COMMON="ising --L 4 --steps 128 --inner 4 --mb 256 --batch 512 \
        --eval-every 100 --n-samples 20000"

# round 3's arm F may still be writing its artifacts on GPU0
while pgrep -f "tag fix_F_clamp8" > /dev/null; do sleep 10; done

run() {
    local gpu="$1" tag="$2"; shift 2
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m dam.discrete $COMMON $STAB "$@" \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "dam/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END $tag exit=$?"
}

run 0 fix_H_long3000 --K 16 --iters 3000 &
run 1 fix_I_K32      --K 32 --iters 1500 &
wait
echo "=== [$(date -u '+%F %T UTC')] ROUND4 DONE"
