#!/usr/bin/env bash
# IASBS legs of the fair St(4,2) comparison against R-ASBS.
#
# Three legs, all with the IASBS construction untouched (Dirac source at
# E0 = [e1, e2], exact geodesic step, intertwined kernel, 199 training steps):
#
#   frame     frame-sensitive target E = tr(X^T H X) - lam tr(C^T X) at
#             beta = 1, lam = 1, H = diag(1,2,5,8), C the fixed 4x2 matrix in
#             stiefel.py.  Native budget (2048 x 2500 = 5,120,000 oracle
#             calls), 5 seeds, refined to 398 and 796 steps.  --antithetic is
#             refused by the script here: the frame term breaks the 16-fold
#             sign symmetry, which is the whole point of this target.
#
#   frame600  the same target at R-ASBS's own budget, 400 x 1500 = 600,000
#             oracle calls, 5 seeds.  This is the oracle-matched row.
#
#   trace600  the trace target at beta = 1.3, 2, 5, 600,000 oracle calls,
#             3 seeds, refined to 398.  Extends the single-seed
#             json/results_stiefel_matched.json to a seed spread.
#
#   bash iasbs/scripts/run_fair_stiefel.sh frame
#   bash iasbs/scripts/run_fair_stiefel.sh frame600
#   bash iasbs/scripts/run_fair_stiefel.sh trace600
#   bash iasbs/scripts/run_fair_stiefel.sh all
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/root/miniconda3/envs/SML_env/bin/python}
mkdir -p iasbs/logs
WHICH=${1:-all}

REF="--mcmc-chains 200000 --mcmc-sweeps 3000 --mcmc-eps 0.35 --n-samples 100000"

leg() {                              # leg <tag> <extra flags...>
    local tag="$1"; shift
    echo "=== [$(date -u '+%F %T UTC')] START $tag"
    "$PY" -u iasbs/stiefel.py train --steps 199 --nq 64 \
        --inner 8 --hidden 256 --lr 1e-3 --ema 0.9995 --nbuf 4 \
        $REF --verbose "$@" \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "iasbs/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
}

if [ "$WHICH" = all ] || [ "$WHICH" = frame ]; then
    leg stiefel_frame_s5 --frame --betas 1.0 --iters 2500 --batch 2048 \
        --mb 16384 --seeds 5 --eval-every 500 --refine 2,4
fi

if [ "$WHICH" = all ] || [ "$WHICH" = frame600 ]; then
    leg stiefel_frame600 --frame --betas 1.0 --iters 1500 --batch 400 \
        --mb 8192 --seeds 5 --eval-every 750 --refine 2,4
fi

if [ "$WHICH" = all ] || [ "$WHICH" = trace600 ]; then
    leg stiefel_matched_s3 --betas 1.3,2,5 --iters 1500 --batch 400 \
        --mb 8192 --seeds 3 --eval-every 750 --refine 2 --antithetic
fi

echo "=== [$(date -u '+%F %T UTC')] DONE ($WHICH)"
