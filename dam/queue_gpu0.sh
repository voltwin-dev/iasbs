#!/usr/bin/env bash
# GPU0 serial queue, v2 (2026-09-07 14:10 UTC).
#
# v1 sized dam_ising_L4_K16 at --eval-every 50 from the 10-iteration diagnostic
# (2.13 s/iter, with two evals already INCLUDED in that 21 s).  Running alone on
# GPU0, v1 then produced ZERO eval lines in 905 s -- it never reached iteration
# 50, which at 2.13 s/iter should have cost ~110 s.  Per-iteration cost
# therefore GROWS after iteration 10: the same jump-count blow-up seen on
# occupation-scale m=32, only with a slower onset.  v1 was killed at 15:05
# elapsed; no artifacts are written before the loop ends, so nothing was lost.
#
# v2 makes the growth observable.  --eval-every 5 prints cumulative wall clock,
# TV, loss and ESS every 5 iterations, so the s/iter curve and the loss
# magnitude can be read straight off the log.  iters is cut 1200 -> 200: if the
# cost really is growing then 1200 is unreachable, and if it is not, 200
# iterations is enough to see the trend and re-size.
#
# occupation-scale m=32 is retried at K=64 (v1 rationale unchanged): at K=16 it
# diverged with 153 jumps per f1 eval and ESS p10 = 1.00, i.e. the adjoint
# denominator was carried by a single rollout.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/SML_env/bin/python

run() {
    local tag="$1"; shift
    echo "=== [$(date -u '+%F %T UTC')] START $tag"
    "$PY" -u -m dam.discrete "$@" --tag "$tag" --out "json/results_${tag}.json" \
        > "dam/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
}

run dam_ising_L4_K16 ising --L 4 --K 16 --steps 128 --iters 200 \
    --inner 4 --mb 256 --batch 512 --eval-every 5 --n-samples 20000

run dam_occs32_K64 occupation-scale --m 32 --K 64 --steps 128 --iters 400 \
    --inner 4 --mb 256 --batch 512 --eval-every 25 --n-samples 10000

echo "=== [$(date -u '+%F %T UTC')] QUEUE DONE"
