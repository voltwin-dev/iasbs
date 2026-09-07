#!/usr/bin/env bash
# DAM divergence-stabiliser sweep, round 7 (queued behind round 6).
#
# Completes the discrete coverage.  Rounds 1-6 repaired Ising L=4 and tested the
# transfer to occupation-scale m=32; the two benchmarks never tried with the
# repair are m=128 (diverged unmodified at K=16: >74 s/iter, killed at 3695 s
# short of iteration 50) and Ising L=5 (never run with DAM at all, because
# L=4 diverged).
#
# Same stabiliser set as rounds 3-6, unchanged, so this is a transfer test and
# not another tuning pass:  --a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10
# at K=32, the value round 5 showed to be the optimum on Ising L=4.
#
# Budgets are sized from measured per-iteration costs, not guessed:
#   m=128    m=32 at K=64 runs at ~20 s/iter, and the unmodified m=128 leg at
#            K=16 exceeded 74 s/iter WHILE DIVERGING.  A stabilised K=32 run
#            should sit between; 600 iterations is budgeted at ~5 h.
#   Ising L=5  |Omega| = 5,200,300, and every eval calls propagate_exact over
#            all of them, so --eval-every is kept at 250.  1200 iterations is a
#            first sizing run; if it is still descending at the cutoff that is
#            reported as such rather than extrapolated.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/SML_env/bin/python
STAB="--a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10"

# wait for round 6 to clear both GPUs
while pgrep -f "tag fix_M_K32_7000|tag fix_N_occs32" > /dev/null; do sleep 60; done
echo "=== [$(date -u '+%F %T UTC')] round 6 clear, starting round 7"

run() {
    local gpu="$1" tag="$2"; shift 2
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m dam.discrete "$@" $STAB \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "dam/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END $tag exit=$?"
}

run 0 fix_O_occs128 occupation-scale --m 128 --K 32 --iters 600 --steps 128 \
    --inner 4 --mb 256 --batch 512 --eval-every 25 --n-samples 10000 &

run 1 fix_P_isingL5 ising --L 5 --K 32 --iters 1200 --steps 128 --inner 4 \
    --mb 256 --batch 512 --eval-every 250 --n-samples 20000 &
wait
echo "=== [$(date -u '+%F %T UTC')] ROUND7 DONE"
