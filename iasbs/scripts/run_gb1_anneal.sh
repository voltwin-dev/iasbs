#!/usr/bin/env bash
# GB1 k=3 (Appendix A.2) with the temperature-annealed schedule that is now the
# default for this target.  The budget is the same 3000 iterations as the flat
# tau=1 run of README section 2.5; it is only spent differently: 1000 iterations
# at tau=2, 1000 at tau=1.4, 1000 at tau=1, each stage warm-starting the network
# weights of the previous one via --init-from.
#
# The reported number is the stage-C exact TV, evaluated at tau=1 against the
# unmodified Wu et al. Boltzmann law, so the benchmark itself is untouched --
# only the optimiser's path through it changes.  The same schedule applies to
# DAM (dam/run_dam.sh) so the head-to-head stays matched.
#
#   bash iasbs/scripts/run_gb1_anneal.sh dirac
#   bash iasbs/scripts/run_gb1_anneal.sh nd
#   bash iasbs/scripts/run_gb1_anneal.sh both
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
mkdir -p iasbs/logs
WHICH=${1:-both}

GB1="--target gb1 \
 --gb1-measured data/gb1/elife-16965-supp1-v4.xlsx \
 --gb1-imputed  data/gb1/elife-16965-supp2-v4.xlsx"
COMMON="--steps 256 --batch 2048 --buffer 8 --inner 40 --mb 1024 \
 --hidden 512 --lr 3e-4 --loss poisson --seed 0 --eval-every 250"

# Stage iters and temperatures.  1000 + 1000 + 1000 = the 3000 of section 2.5.
IT=1000
TAUS="2.0 1.4 1.0"

# Only the final stage needs the full 20k-sample constraint audit; the two
# annealed stages are intermediate and get a cheap eval.
anneal() {                       # anneal <cmd> <stem> [extra flags...]
    local cmd="$1" stem="$2"; shift 2
    local prev="" stage=A
    for tau in $TAUS; do
        local tag="fs_gb1_k3_${stem}_${stage}"
        local ns=2000
        [ "$stage" = C ] && ns=20000
        local init=""
        [ -n "$prev" ] && init="--init-from ckpt/${prev}.pt"
        echo "=== [$(date -u '+%F %T UTC')] START $tag  tau=$tau  ${init:-cold}"
        "$PY" -u iasbs/fixed_support.py "$cmd" $GB1 $COMMON "$@" \
            --tau "$tau" --iters "$IT" --n-samples "$ns" $init \
            --tag "$tag" --out "json/results_${tag}.json" \
            > "iasbs/logs/${tag}.log" 2>&1
        echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
        prev="$tag"
        case $stage in A) stage=B ;; B) stage=C ;; esac
    done
}

if [ "$WHICH" = both ] || [ "$WHICH" = dirac ]; then
    anneal train dirac
fi

if [ "$WHICH" = both ] || [ "$WHICH" = nd ]; then
    anneal train-nondirac nd \
        --inner-h 40 --lr-h 3e-4 --corrector-hidden 256 --corrector-actions 32
fi

echo "=== [$(date -u '+%F %T UTC')] DONE ($WHICH)"
