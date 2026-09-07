#!/usr/bin/env bash
# Sequential queue for the remaining DAM baseline legs.
#
# These run one at a time on purpose. Every DAM rollout is a Gillespie loop whose
# wall-clock cost is set by the number of `while` iterations, not by batch size, so
# the runs are latency-bound rather than throughput-bound: putting two of them on
# one device roughly doubles both their wall times instead of overlapping. Running
# them serially therefore gives the same total time AND a clean per-run wall-clock
# number, which is the quantity the cost comparison in additional_result.md needs.
#
# Usage:  CUDA_VISIBLE_DEVICES=1 nohup bash dam/queue_remaining.sh > dam/logs/queue.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/SML_env/bin/python

run() {
    local tag="$1"; shift
    echo "=== [$(date -u '+%F %T UTC')] START $tag"
    "$PY" -u -m dam.discrete "$@" \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "dam/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
}

# Occupation scaling, matched against IASBS section 2.3. The iteration budget is
# cut as m grows because the per-rollout jump count grows with m; if a leg cannot
# finish in a comparable budget that is itself the reportable result.
run dam_occs128_K16  occupation-scale --m 128  --K 16 --steps 128 --iters 600 \
    --inner 4 --mb 256 --batch 512 --eval-every 50 --n-samples 10000

run dam_occs1000_K16 occupation-scale --m 1000 --K 16 --steps 256 --iters 300 \
    --inner 4 --mb 256 --batch 512 --eval-every 25 --n-samples 4000

# Ising L=5, matched against IASBS section 2.1.
run dam_ising_L5_K16 ising --L 5 --K 16 --steps 128 --iters 600 \
    --inner 4 --mb 256 --batch 512 --eval-every 50 --n-samples 20000

echo "=== [$(date -u '+%F %T UTC')] QUEUE DONE"
