#!/usr/bin/env bash
# DAM on occupation-scale m=128, K=64 -- the section 6.2.1 leg.
#
# Same control box as the m=32 leg of section 6.2 (a-clamp 8, m-clip 5,
# ess-min 3, coef-cap 10, 128 steps), scaled up one state-space step.  K=64 is
# the value the m=32 leg needed; K=32 on this space starves the adjoint
# estimator outright (ESS 1.52/32, KS_occ above the uncontrolled reference).
# Even at K=64 the estimator collapses here -- ESS ends at 1.69 / 64 with
# 451,665 clipped labels -- which is the point of the section.
#
# Budget: 500 iterations, ~135 s/iter, ~18.8 h on one A100.
#
#   bash dam/run_occs128_K64.sh
#
# Writes json/results_dam_occs128_K64_500.json and ckpt/dam_occs128_K64_500.pt.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/root/miniconda3/envs/SML_env/bin/python}
mkdir -p dam/logs
STAB="--a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10"
TAG=dam_occs128_K64_500

echo "=== [$(date -u '+%F %T UTC')] START $TAG"
"$PY" -u -m dam.discrete occupation-scale \
    --m 128 --K 64 --iters 500 --steps 128 --inner 4 --mb 256 --batch 512 \
    --eval-every 25 --n-samples 10000 $STAB \
    --tag "$TAG" --out "json/results_${TAG}.json" \
    > "dam/logs/${TAG}.log" 2>&1
echo "=== [$(date -u '+%F %T UTC')] END   $TAG  exit=$?"
