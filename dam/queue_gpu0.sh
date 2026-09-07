#!/usr/bin/env bash
# GPU0 serial queue, sized from the 10-iteration diagnostics of 2026-09-07.
#
# diag_isingL4: loss 10.56 -> 7.75, ESS 3.68/16, 5.4 jumps per f1 eval,
#   2.1 s/iter on an UNCONTENDED device.  Healthy -- the earlier >104 s/iter was
#   pure co-scheduling contention, not divergence.  1200 iters is ~45 min.
# diag_occs32:  loss -2.4e5 -> -2.9e13, ESS 1.93/16 with p10 1.00, 153 jumps per
#   f1 eval.  Diverging at K=16, i.e. the stability cliff of section 5.2 MOVES
#   with m: K=16 is sufficient at m=4 and insufficient at m=32.  So m=32 is
#   retried at K=64 rather than repeated at K=16.
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
run dam_ising_L4_K16 ising --L 4 --K 16 --steps 128 --iters 1200 \
    --inner 4 --mb 256 --batch 512 --eval-every 50 --n-samples 20000
run dam_occs32_K64 occupation-scale --m 32 --K 64 --steps 128 --iters 400 \
    --inner 4 --mb 256 --batch 512 --eval-every 25 --n-samples 10000
echo "=== [$(date -u '+%F %T UTC')] QUEUE DONE"
