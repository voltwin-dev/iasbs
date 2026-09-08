#!/usr/bin/env bash
# DAM on the Appendix A.2 fixed-support spaces: the toy Potts ring at tau = 1
# and GB1 k=3 on the same temperature-annealed schedule that is the IASBS
# default for that target (400 iterations each at tau = 2.0, 1.4, 1.0, warm
# started, total 1200 = the flat toy budget).  Giving DAM the identical schedule
# is what keeps the section 6.2-style head-to-head matched.
#
#   bash dam/run_dam_a2.sh toy     # 1200 it, tau = 1
#   bash dam/run_dam_a2.sh gb1     # 3 x 400 it, tau 2 -> 1.4 -> 1
#   bash dam/run_dam_a2.sh both
#
# Writes json/results_<tag>.json and ckpt/<tag>.pt per leg.  DAM is
# latency-bound (each rollout is a Python while loop), so co-scheduling the two
# legs on one device roughly doubles both wall times rather than overlapping.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p dam/logs
WHICH=${1:-both}

GB1="--gb1-measured data/gb1/elife-16965-supp1-v4.xlsx \
 --gb1-imputed  data/gb1/elife-16965-supp2-v4.xlsx"
COMMON="--K 16 --K-num 1 --steps 256 --inner 4 --batch 512 --mb 256 \
 --buffer 8 --hidden 512 --lr 1e-3 --eval-every 50 --seed 0"

if [ "$WHICH" = both ] || [ "$WHICH" = toy ]; then
    tag=dam_fs_toy_v2_K16
    echo "=== [$(date -u '+%F %T UTC')] START $tag"
    "$PY" -u -m dam.discrete fixed-support --target toy $COMMON \
        --iters 1200 --n-samples 20000 \
        --tag "$tag" --out "json/results_${tag}.json" \
        > "dam/logs/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
fi

if [ "$WHICH" = both ] || [ "$WHICH" = gb1 ]; then
    prev=""
    stage=A
    for tau in 2.0 1.4 1.0; do
        tag=dam_fs_gb1_k3_K16_${stage}
        ns=2000; [ "$stage" = C ] && ns=20000
        init=""; [ -n "$prev" ] && init="--init-from ckpt/${prev}.pt"
        echo "=== [$(date -u '+%F %T UTC')] START $tag  tau=$tau  ${init:-cold}"
        "$PY" -u -m dam.discrete fixed-support --target gb1 $GB1 $COMMON \
            --tau "$tau" --iters 400 --n-samples "$ns" $init \
            --tag "$tag" --out "json/results_${tag}.json" \
            > "dam/logs/${tag}.log" 2>&1
        echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
        prev="$tag"
        case $stage in A) stage=B ;; B) stage=C ;; esac
    done
fi

echo "=== [$(date -u '+%F %T UTC')] DONE ($WHICH)"
