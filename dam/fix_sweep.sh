#!/usr/bin/env bash
# DAM divergence-stabiliser sweep, round 1 (2026-09-07).
#
# Testbed: Ising L=4, K=16.  It is the cheapest leg that diverges (2.1 s/iter
# while healthy) and its failure is sharply dated: loss turns negative at
# iteration 15 and the ESS is 1.04/16 by iteration 40.  60 iterations at
# --eval-every 5 therefore separates "fixed" from "delayed" without spending an
# hour per arm.  Reference trajectory to beat (dam_ising_L4_K16_DIVERGED.log):
#
#   it  5  loss  10.93  TV 0.76046  ESS 3.82
#   it 15  loss -16.30  TV 0.76199  ESS 3.02   <- sign flip
#   it 25  loss -3.8e12 TV 0.85085  ESS 1.76
#   it 40  loss -4.2e12 TV 0.97347  ESS 1.04
#
# Four arms, each attacking a different link in the feedback loop
# (large log m_hat -> large gradient on a -> large escape rate -> more jumps ->
#  worse ESS -> larger log m_hat):
#
#   A  bound both ends of that loop hard:  |a| <= 3, |log m| <= 5.
#   B  leave the loop alone, remove the outliers that drive it: drop labels
#      with ESS < 4 / 16, winsorise the r/q prefactor at 10x its median.
#   C  same loop, slower: lr 1e-3 -> 2e-4 with moderate bounds.
#   D  everything at once, as the upper bound on what these knobs can buy.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/SML_env/bin/python
COMMON="ising --L 4 --K 16 --steps 128 --iters 60 --inner 4 --mb 256 \
        --batch 512 --eval-every 5 --n-samples 2000"

run() {
    local gpu="$1" tag="$2"; shift 2
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m dam.discrete $COMMON "$@" \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "dam/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END $tag exit=$?"
}

# GPU0: arms A then C
(
  run 0 fix_A_clampboth --a-clamp 3 --m-clip 5
  run 0 fix_C_slowlr    --a-clamp 5 --m-clip 10 --lr 2e-4
) &

# GPU1: arms B then D
(
  run 1 fix_B_essfilter --ess-min 4 --coef-cap 10
  run 1 fix_D_all       --a-clamp 3 --m-clip 5 --ess-min 4 --coef-cap 10 --lr 3e-4
) &

wait
echo "=== [$(date -u '+%F %T UTC')] SWEEP DONE"
