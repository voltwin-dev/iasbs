#!/usr/bin/env bash
# Reproduces every DAM row of README section 6.2, in the order the tables list
# them.  One leg at a time on purpose: each Gillespie rollout is a Python
# `while` loop, so DAM is latency-bound and co-scheduling two legs on one device
# roughly doubles both wall times instead of overlapping them.
#
#   bash dam/run_dam.sh            # all seven legs, ~19 h total on one A100
#   bash dam/run_dam.sh occ4       # only the occupation m=4 legs   (~ 6 h)
#   bash dam/run_dam.sh ising      # only the Ising L=4 legs        (~10 h)
#   bash dam/run_dam.sh occs32     # only occupation-scale m=32     (~4.6 h)
#
# Writes json/results_<tag>.json and ckpt/<tag>.pt for each leg.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p dam/logs
WHICH=${1:-all}

# The section 6.2 control box.  Off by default in the code, so the section 5.1
# K-sweep below reproduces the unstabilised numbers bit-for-bit.
STAB="--a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10"

run() {
    local tag="$1"; shift
    echo "=== [$(date -u '+%F %T UTC')] START $tag"
    "$PY" -u -m dam.discrete "$@" \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "dam/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
}

# ---------------------------------------------------------------- occupation m=4
# Sections 5.1, 5.2, 5.3, 6.2 and the head-to-head of 6.3.  No stabilisers:
# |X| = 35 is small enough that stock DAM converges.
if [ "$WHICH" = all ] || [ "$WHICH" = occ4 ]; then
run dam_occupation_m4_K16 occupation --m 4 --K 16 --steps 128 --iters 400 \
    --inner 4 --mb 256 --batch 512 --hidden 256 --lr 1e-3 \
    --eval-every 25 --n-samples 20000 --seed 0

run dam_occ4_K16_long     occupation --m 4 --K 16 --steps 128 --iters 1200 \
    --inner 4 --mb 256 --batch 512 --hidden 256 --lr 1e-3 \
    --eval-every 50 --n-samples 20000 --seed 0

run dam_occ4_K64          occupation --m 4 --K 64 --steps 128 --iters 400 \
    --inner 4 --mb 256 --batch 512 --hidden 256 --lr 1e-3 \
    --eval-every 50 --n-samples 20000 --seed 0
fi

# ---------------------------------------------------------------- Ising L=4
# Section 6.2.  K=32 at 5000 iterations is the best of the three; K=64 at 2500
# is the matched-budget K-axis check; K=32 at 7000 is the plateau check.
if [ "$WHICH" = all ] || [ "$WHICH" = ising ]; then
run dam_ising_L4_K32_5000 ising --L 4 --K 32 --steps 128 --iters 5000 \
    --inner 4 --mb 256 --batch 512 --hidden 512 --lr 1e-3 $STAB \
    --eval-every 250 --n-samples 20000 --seed 0

run dam_ising_L4_K64_2500 ising --L 4 --K 64 --steps 128 --iters 2500 \
    --inner 4 --mb 256 --batch 512 --hidden 512 --lr 1e-3 $STAB \
    --eval-every 250 --n-samples 20000 --seed 0

run dam_ising_L4_K32_7000 ising --L 4 --K 32 --steps 128 --iters 7000 \
    --inner 4 --mb 256 --batch 512 --hidden 512 --lr 1e-3 $STAB \
    --eval-every 250 --n-samples 20000 --seed 0
fi

# ---------------------------------------------------------------- occupation-scale m=32
# Section 6.2.  K=64 is what m=32 needs; the clamp is carried over from Ising
# L=4 unchanged, since ScaleOccupation has no ExactControl to size it against.
if [ "$WHICH" = all ] || [ "$WHICH" = occs32 ]; then
run dam_occs32_K64_1000 occupation-scale --m 32 --K 64 --steps 128 --iters 1000 \
    --inner 4 --mb 256 --batch 512 --hidden 128 --lr 1e-3 $STAB \
    --eval-every 25 --n-samples 10000 --seed 0
fi

echo "=== [$(date -u '+%F %T UTC')] DONE ($WHICH)"
