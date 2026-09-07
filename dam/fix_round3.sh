#!/usr/bin/env bash
# DAM divergence-stabiliser sweep, round 3 (2026-09-07 16:45 UTC).
#
# Round 2 verdict: |a| <= 3 is BELOW THE TARGET.  The exact Ising L=4 control
# multiplier, read straight off FI.ExactControl, grows towards t = 1:
#
#     t      0.000  0.250  0.500  0.750  0.938  0.992
#     max|a| 0.185  0.576  1.431  2.629  4.205  6.308
#
# so a clamp of 3 makes the optimal controller literally unrepresentable over
# the last ~20% of the time axis, and the run cannot converge no matter how
# stable it is.  This is visible in the round-2 artifacts: the learned-vs-exact
# multiplier error is 0.69-0.84 RMSE for t <= 0.75 but 2.09 RMSE / 8.10 max at
# t = 0.992, i.e. the whole error budget sits exactly where the clamp binds.
# It also explains the shape of round 2 -- A2 drove the energy histogram from
# 0.632 to 0.158 and then bounced back to 0.303, which is what a run does when
# it approaches a boundary it cannot cross.
#
# Round 3 sets the clamp ABOVE the target (8 > 6.31, ~27% headroom) and keeps
# the truncation that actually stopped the divergence (|log m_hat| <= 5).  The
# baseline's clamp of 20 permits an escape-rate multiplier of 4.9e8; 8 permits
# 3.0e3, which covers the optimum with room to spare and still bounds the jump
# explosion.  Two arms so the extra damping is measured, not assumed:
#
#   F  clamp 8, m-clip 5                      -- minimal change from round 2
#   G  clamp 8, m-clip 5, ess-min 3, cap 10   -- plus round-1 arm B's outlier
#                                                removal, in case the looser
#                                                clamp reopens the feedback loop
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

run 0 fix_F_clamp8 --a-clamp 8 --m-clip 5 &
run 1 fix_G_clamp8damp --a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10 &
wait
echo "=== [$(date -u '+%F %T UTC')] ROUND3 DONE"
