#!/usr/bin/env bash
# DAM divergence-stabiliser sweep, round 2 (2026-09-07 16:15 UTC).
#
# Round 1 verdict: all four arms KILLED the divergence.  No arm reproduced the
# baseline's 1e12 loss or its ESS collapse to 1.04/16, and all four ran 60
# iterations in ~130 s where the baseline needed 286 s to reach iteration 40
# (the cost explosion is gone).  None of them learned: TV sat at 0.767-0.785
# for all 60 iterations.
#
# That flat TV is NOT yet evidence of a stall.  The one DAM leg that converges
# (occupation m=4, K=16) also sat flat for a long time -- TV 0.41674 at
# iteration 50, still 0.39471 at 200, and only broke down to 0.06995 by 300.
# Sixty iterations is inside the flat phase for that leg too, so round 1 simply
# cannot distinguish "stabilised but stalled" from "stabilised and about to
# descend".  Round 2 buys the iterations needed to tell them apart: 600, i.e.
# past the point where the m=4 leg had already crossed the gate.
#
# Two arms, chosen on round-1 evidence:
#   A2  the round-1 arm that moved the energy histogram furthest
#       (E-hist TV 0.75153 -> 0.70286) at the original lr.
#   E   the same bounds at lr 3e-4, because round-1 arm C (lr 2e-4) held the
#       best ESS (3.37/16 mean, p10 1.34) and the fewest jumps (7.5 M vs 9.6 M),
#       suggesting the bounds and a gentler step are complementary.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/SML_env/bin/python
COMMON="ising --L 4 --K 16 --steps 128 --iters 600 --inner 4 --mb 256 \
        --batch 512 --eval-every 25 --n-samples 20000"

run() {
    local gpu="$1" tag="$2"; shift 2
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m dam.discrete $COMMON "$@" \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "dam/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END $tag exit=$?"
}

run 0 fix_A2_long --a-clamp 3 --m-clip 5 &
run 1 fix_E_lr3e4 --a-clamp 3 --m-clip 5 --lr 3e-4 &
wait
echo "=== [$(date -u '+%F %T UTC')] ROUND2 DONE"
