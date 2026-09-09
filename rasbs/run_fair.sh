#!/usr/bin/env bash
# R-ASBS legs of the fair St(4,2) comparison.
#
# Two legs, both with the port's native construction untouched (Haar source,
# ambient Euler + QR retraction, netU/netH, sigma = 1, B = 600, 1000 epochs =
# 600,000 terminal oracle calls, lr 1e-3, 199 training steps):
#
#   frame   the frame-sensitive target  E = tr(X^T H X) - lam tr(C^T X)  at
#           beta = 1, lam = 1, C = U C_iasbs.  This is the key experiment: the
#           frame term breaks the 16-fold sign symmetry of the trace target,
#           removing the symmetry advantage IASBS has there.  5 seeds,
#           evaluated at 199 / 398 / 796 inference steps -- the same three
#           resolutions IASBS is reported on -- and scored against IASBS's own
#           MCMC reference with 100,000 terminal samples on both sides.
#
#   trace   the matched-600k-oracle trace target at beta = 1.3, 2, 5, 3 seeds,
#           199 / 398 steps, same reference and sample count.
#
#   bash rasbs/run_fair.sh frame
#   bash rasbs/run_fair.sh trace
#   bash rasbs/run_fair.sh both
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/root/miniconda3/envs/SML_env/bin/python}
mkdir -p rasbs/logs
WHICH=${1:-both}

if [ "$WHICH" = both ] || [ "$WHICH" = frame ]; then
    tag=rasbs_frame
    echo "=== [$(date -u '+%F %T UTC')] START $tag"
    "$PY" -u rasbs/rasbs_port.py --frame --lam 1.0 --betas 1.0 \
        --seeds 5 --refine 2,4 --n-samples 100000 \
        --ref-stem stiefel_frame --verbose \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "rasbs/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
fi

if [ "$WHICH" = both ] || [ "$WHICH" = trace ]; then
    tag=rasbs_m600
    echo "=== [$(date -u '+%F %T UTC')] START $tag"
    "$PY" -u rasbs/rasbs_port.py --betas 1.3,2,5 \
        --seeds 3 --refine 2 --n-samples 100000 \
        --ref-stem stiefel_matched --verbose \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "rasbs/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
fi

echo "=== [$(date -u '+%F %T UTC')] DONE ($WHICH)"
