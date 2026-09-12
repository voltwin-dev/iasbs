#!/usr/bin/env bash
# Seed replication for every single-seed headline number in MULTISEED_RESULTS.md (appendix).
#
# Nothing in the four discrete mains (fixed_ising.py, occupation.py,
# fixed_support.py, dam/discrete.py) loops over seeds: they take a single
# --seed, and common.py's save_ckpt writes <ckpt_dir>/<tag>.pt with no seed
# suffix.  Re-running a recorded config with --seed 1 would therefore overwrite
# the seed-0 checkpoint and its results JSON.  This script closes that hole
# without touching the mains: every row below re-issues the exact flags taken
# from the seed-0 results JSON's "config" block, and injects
#
#     --seed S  --tag <base>_s<S>  --out json/results_<base>_s<S>.json
#
# so each seed lands in its own artifact.  Seed 0's original files are never
# written to; the s0 copy is a fresh run under the seed-suffixed name, which
# also serves as a reproducibility check against the number already in README.
#
#   bash iasbs/scripts/multiseed.sh list
#   bash iasbs/scripts/multiseed.sh <row> [seeds] [drop|keep]
#   bash iasbs/scripts/multiseed.sh group:ising  [seeds] [drop|keep]
#   bash iasbs/scripts/multiseed.sh group:main   [seeds] [drop|keep]
#   bash iasbs/scripts/multiseed.sh group:last   [seeds] [drop|keep]
#   bash iasbs/scripts/multiseed.sh all          [seeds] [drop|keep]
#
# group:last is the five long DAM rows (63 h of the 100 h serial estimate);
# group:main is everything else (37 h).  Run main first, last last.
#
# Third argument -- mid-run evaluations, default "drop".
#
#   drop  append --eval-every 1000000000 to every command.  All sixteen
#         evaluation sites in the four mains are guarded by
#         "it % args.eval_every == 0 or it == args.iters", so the final
#         evaluation still runs and every headline number is unchanged: the
#         reported figure is always the last eval, never a min over history
#         (checked -- e.g. dam_occs32_K64_1000 reports its final 0.01284, not
#         its best 0.00272).  What is lost is the TV-vs-iteration trajectory
#         and the progress lines in the log.  Seed 0's curves already exist.
#         On Ising 5x5 non-Dirac this removes 120 exact propagations over
#         2^25 states and is worth roughly 9 h per seed.
#
#   keep  use each row's recorded --eval-every, i.e. reproduce the seed-0 run
#         including its evaluation cadence.  Slower; needed only if the
#         trajectory itself is wanted for the added seeds.
#
# Aggregate afterwards with
#   python iasbs/scripts/aggregate_seeds.py <base> [<base> ...]
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/root/miniconda3/envs/SML_env/bin/python}
mkdir -p iasbs/logs dam/logs json

GB1DATA="--gb1-measured data/gb1/elife-16965-supp1-v4.xlsx \
 --gb1-imputed data/gb1/elife-16965-supp2-v4.xlsx"

# ---------------------------------------------------------------- row table --
# name | entry point | subcommand + flags (verbatim from the seed-0 config) | base tag
rows() {
cat <<'ROWS'
ising_L4_dirac|iasbs/fixed_ising.py|train --L 4 --tau 2.0 --gamma 10.0 --steps 256 --iters 3000 --batch 2048 --buffer 8 --inner 40 --mb 1024 --hidden 512 --lr 3e-4 --loss poisson --eval-every 200 --n-samples 20000|ising_poisson256
ising_L4_nd|iasbs/fixed_ising.py|train-nondirac --L 4 --tau 2.0 --gamma 10.0 --steps 256 --iters 3000 --batch 2048 --buffer 8 --inner 40 --mb 1024 --hidden 512 --lr 3e-4 --loss poisson --inner-h 40 --lr-h 3e-4 --eval-every 25 --n-samples 200000|ising_nd_L4
ising_L4_dam|dam/discrete.py|ising --L 4 --J 1.0 --tau 2.0 --gamma 10.0 --K 32 --steps 128 --iters 5000 --inner 4 --batch 512 --mb 256 --buffer 8 --hidden 512 --lr 1e-3 --a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10 --eval-every 250 --n-samples 20000|dam_ising_L4_K32_5000
ising_L5_dirac|iasbs/fixed_ising.py|train --L 5 --tau 2.0 --gamma 10.0 --steps 512 --iters 6000 --batch 2048 --buffer 8 --inner 40 --mb 1024 --hidden 512 --lr 3e-4 --loss poisson --eval-every 500 --n-samples 200000|ising_t1_L5_s512
ising_L5_nd|iasbs/fixed_ising.py|train-nondirac --L 5 --tau 2.0 --gamma 10.0 --steps 256 --iters 3000 --batch 2048 --buffer 8 --inner 40 --mb 1024 --hidden 512 --lr 3e-4 --loss poisson --inner-h 40 --lr-h 3e-4 --eval-every 25 --n-samples 200000|ising_nd_L5
ising_L5_dam|dam/discrete.py|ising --L 5 --J 1.0 --tau 2.0 --gamma 10.0 --K 32 --steps 128 --iters 5000 --inner 4 --batch 512 --mb 256 --buffer 8 --hidden 512 --lr 1e-3 --a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10 --eval-every 250 --n-samples 20000|dam_ising_L5_K32_5000
occ4_dirac|iasbs/occupation.py|train --m 4 --N 4 --d 0.5 --tau 1.0 --gamma 4.0 --steps 128 --iters 1500 --inner 4 --batch 512 --mb 1024 --buffer 8 --hidden 256 --lr 1e-3 --loss bregman --estimator full --eval-every 250 --n-samples 20000 --reps 2000|occ4_full
occ4_nd|iasbs/occupation.py|train-nondirac --m 4 --N 4 --d 0.5 --tau 1.0 --gamma 4.0 --steps 128 --iters 1500 --inner 4 --batch 512 --mb 1024 --buffer 8 --hidden 256 --lr 1e-3 --loss bregman --estimator full --nu-skew 1.0 --inner-h 4 --lr-h 1e-2 --eval-every 100 --n-samples 20000 --reps 2000|occ_nd_m4
occ4_dam|dam/discrete.py|occupation --m 4 --N 4 --d 0.5 --tau 1.0 --gamma 4.0 --K 16 --steps 128 --iters 1200 --inner 4 --batch 512 --mb 256 --buffer 8 --hidden 256 --lr 1e-3 --eval-every 50 --n-samples 20000|dam_occ4_K16_long
occs32_dirac|iasbs/occupation.py|scale --m 32 --d 0.5 --tau 1.0 --gamma 4.0 --steps 128 --iters 3000 --inner 4 --batch 512 --mb 1024 --buffer 8 --hidden 256 --lr 1e-3 --loss bregman --estimator full --eval-every 500 --n-samples 20000 --reps 2000|occ_s32
occs32_nd|iasbs/occupation.py|scale-nondirac --m 32 --d 0.5 --tau 1.0 --gamma 4.0 --steps 128 --iters 3000 --inner 4 --batch 512 --mb 1024 --buffer 8 --hidden 256 --lr 1e-3 --loss bregman --estimator full --nu-skew 1.0 --inner-h 4 --lr-h 1e-3 --eval-every 100 --n-samples 10000 --reps 2000|occ_nd_s32
occs32_dam|dam/discrete.py|occupation-scale --m 32 --d 0.5 --tau 1.0 --gamma 4.0 --K 64 --steps 128 --iters 1000 --inner 4 --batch 512 --mb 256 --buffer 8 --hidden 128 --lr 1e-3 --a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10 --eval-every 25 --n-samples 10000|dam_occs32_K64_1000
occs128_dirac|iasbs/occupation.py|scale --m 128 --d 0.5 --tau 1.0 --gamma 4.0 --steps 128 --iters 3000 --inner 4 --batch 512 --mb 1024 --buffer 8 --hidden 256 --lr 1e-3 --loss bregman --estimator full --eval-every 500 --n-samples 10000 --reps 2000|occ_s128
occs128_nd|iasbs/occupation.py|scale-nondirac --m 128 --d 0.5 --tau 1.0 --gamma 4.0 --steps 128 --iters 3000 --inner 4 --batch 512 --mb 1024 --buffer 8 --hidden 256 --lr 1e-3 --loss bregman --estimator full --nu-skew 1.0 --inner-h 4 --lr-h 1e-2 --eval-every 100 --n-samples 10000 --reps 2000|occ_nd_s128
occs128_dam|dam/discrete.py|occupation-scale --m 128 --d 0.5 --tau 1.0 --gamma 4.0 --K 64 --steps 128 --iters 500 --inner 4 --batch 512 --mb 256 --buffer 8 --hidden 128 --lr 1e-3 --a-clamp 8 --m-clip 5 --ess-min 3 --coef-cap 10 --eval-every 25 --n-samples 10000|dam_occs128_K64_500
occs1000_dirac|iasbs/occupation.py|scale --m 1000 --d 0.5 --tau 1.0 --gamma 4.0 --steps 256 --iters 1500 --inner 4 --batch 128 --mb 512 --buffer 8 --hidden 256 --lr 1e-3 --loss bregman --estimator full --eval-every 250 --n-samples 4000 --reps 2000|occ_s1000
occs1000_nd|iasbs/occupation.py|scale-nondirac --m 1000 --d 0.5 --tau 1.0 --gamma 4.0 --steps 256 --iters 1500 --inner 4 --batch 512 --mb 512 --buffer 8 --hidden 256 --lr 1e-3 --loss bregman --estimator full --nu-skew 1.0 --inner-h 4 --lr-h 1e-2 --eval-every 100 --n-samples 4000 --reps 2000|occ_nd_s1000
toy_dirac|iasbs/fixed_support.py|train --target toy --steps 256 --iters 3000 --batch 2048 --buffer 8 --inner 40 --mb 1024 --hidden 512 --lr 3e-4 --loss poisson --eval-every 500 --n-samples 20000|fs_toy_v2_dirac
toy_nd|iasbs/fixed_support.py|train-nondirac --target toy --steps 256 --iters 3000 --batch 2048 --buffer 8 --inner 40 --mb 1024 --hidden 512 --lr 3e-4 --loss poisson --inner-h 40 --lr-h 3e-4 --corrector-hidden 256 --corrector-actions 32 --eval-every 500 --n-samples 20000|fs_toy_v2_nd
toy_dam|dam/discrete.py|fixed-support --target toy --tau 1.0 --gamma 10.0 --K 16 --steps 256 --iters 1200 --inner 4 --batch 512 --mb 256 --buffer 8 --hidden 512 --lr 1e-3 --m-clip 30 --eval-every 50 --n-samples 20000|dam_fs_toy_v2_K16
ROWS
}

# The five DAM rows that dominate the budget: 63 h of the 100 h serial estimate,
# occupation m=128 alone 36.5 h.  Marked LAST in MULTISEED_RESULTS.md (appendix) and
# selectable as group:last, with everything else as group:main.
LAST_ROWS="ising_L4_dam ising_L5_dam occ4_dam occs32_dam occs128_dam"

is_last() { case " $LAST_ROWS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Seeds a row already holds on disk, which must therefore not be re-run.
# Occupation m=32 non-Dirac was trained twice before this script existed; its
# runs live in json/results_occ_nd_s32_seed{0,1}.json and carry seed 0 and
# seed 1.  verify_multiseed.py confirms both were run under exactly the
# configuration this row replays, so only seed 2 is still missing and a
# request for "1 2" must run seed 2 alone.  Every other row has seed 0 only.
have_seeds() {                    # have_seeds <row name>
    case "$1" in
        occs32_nd) echo "0 1" ;;
        *)         echo "0" ;;
    esac
}

# Requested seeds minus the ones already on disk.
wanted_seeds() {                  # wanted_seeds <row name> <seeds...>
    local name="$1"; shift
    local have out=""
    have=" $(have_seeds "$name") "
    for s in "$@"; do
        case "$have" in
            *" $s "*) echo "--- skip $name seed $s (already on disk)" >&2 ;;
            *)        out="$out $s" ;;
        esac
    done
    echo "$out"
}

group_of() {                      # group_of <row name>
    case "$1" in
        ising_*)      echo ising ;;
        occ4_*)       echo occ4 ;;
        occs32_*)     echo occs32 ;;
        occs128_*)    echo occs128 ;;
        occs1000_*)   echo occs1000 ;;
        toy_*)        echo toy ;;
        gb1_*)        echo gb1 ;;
        *)            echo other ;;
    esac
}

# ------------------------------------------------------------------ runners --
run_plain() {                     # run_plain <script> <args> <base tag> <seed>
    local script="$1" args="$2" base="$3" s="$4"
    local tag="${base}_s${s}"
    local logdir=iasbs/logs
    [ "${script#dam/}" != "$script" ] && logdir=dam/logs
    local extra=""
    case "$script" in *fixed_support.py|dam/discrete.py) extra="$GB1DATA" ;; esac
    echo "=== [$(date -u '+%F %T UTC')] START $tag"
    # shellcheck disable=SC2086
    # MIDEVAL comes last so its --eval-every overrides the row's recorded one.
    "$PY" -u "$script" $args $extra $MIDEVAL \
        --seed "$s" --tag "$tag" --out "json/results_${tag}.json" \
        > "${logdir}/${tag}.log" 2>&1
    echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
}

# GB1 k=3 is a three-stage annealed chain (tau 2.0 -> 1.4 -> 1.0) in which each
# stage warm-starts from the previous stage's checkpoint via --init-from.  The
# stage tags must carry the seed too, otherwise stage B of seed 1 would resume
# from seed 0's stage-A weights and the chain would be silently crossed.
run_gb1_chain() {                 # run_gb1_chain <dirac|nd|dam> <seed>
    local kind="$1" s="$2" prev="" stage=A ns
    local common script args base logdir
    case "$kind" in
        dirac) script=iasbs/fixed_support.py; base="fs_gb1_k3_dirac"
               args="train --target gb1 --steps 256 --batch 2048 --buffer 8 --inner 40 --mb 1024 --hidden 512 --lr 3e-4 --loss poisson --eval-every 250"
               logdir=iasbs/logs; local it=1000 ;;
        nd)    script=iasbs/fixed_support.py; base="fs_gb1_k3_nd"
               args="train-nondirac --target gb1 --steps 256 --batch 2048 --buffer 8 --inner 40 --mb 1024 --hidden 512 --lr 3e-4 --loss poisson --inner-h 40 --lr-h 3e-4 --corrector-hidden 256 --corrector-actions 32 --eval-every 250"
               logdir=iasbs/logs; local it=1000 ;;
        dam)   script=dam/discrete.py; base="dam_fs_gb1_k3_K16"
               args="fixed-support --target gb1 --gamma 10.0 --K 16 --steps 256 --inner 4 --batch 512 --mb 256 --buffer 8 --hidden 512 --lr 1e-3 --m-clip 30 --eval-every 50"
               logdir=dam/logs; local it=400 ;;
        *) echo "unknown gb1 kind: $kind" >&2; return 1 ;;
    esac
    for tau in 2.0 1.4 1.0; do
        local tag="${base}_s${s}_${stage}"
        ns=2000; [ "$stage" = C ] && ns=20000
        local init=""
        [ -n "$prev" ] && init="--init-from ckpt/${prev}.pt"
        echo "=== [$(date -u '+%F %T UTC')] START $tag  tau=$tau  ${init:-cold}"
        # shellcheck disable=SC2086
        "$PY" -u "$script" $args $GB1DATA $MIDEVAL \
            --tau "$tau" --iters "$it" --n-samples "$ns" $init \
            --seed "$s" --tag "$tag" --out "json/results_${tag}.json" \
            > "${logdir}/${tag}.log" 2>&1
        echo "=== [$(date -u '+%F %T UTC')] END   $tag  exit=$?"
        prev="$tag"
        case $stage in A) stage=B ;; B) stage=C ;; esac
    done
}

run_row() {                       # run_row <name> <seed>
    local name="$1" s="$2"
    case "$name" in
        gb1_dirac) run_gb1_chain dirac "$s"; return ;;
        gb1_nd)    run_gb1_chain nd    "$s"; return ;;
        gb1_dam)   run_gb1_chain dam   "$s"; return ;;
    esac
    local line; line=$(rows | awk -F'|' -v n="$name" '$1==n')
    if [ -z "$line" ]; then echo "unknown row: $name" >&2; return 1; fi
    run_plain "$(echo "$line" | cut -d'|' -f2)" \
              "$(echo "$line" | cut -d'|' -f3)" \
              "$(echo "$line" | cut -d'|' -f4)" "$s"
}

all_rows() { rows | cut -d'|' -f1; echo gb1_dirac; echo gb1_nd; echo gb1_dam; }

# run_row over the requested seeds, dropping any the row already has on disk.
run_seeds() {                     # run_seeds <row name> <seeds...>
    local n="$1"; shift
    # shellcheck disable=SC2046
    for s in $(wanted_seeds "$n" "$@"); do run_row "$n" "$s"; done
}

# --------------------------------------------------------------------- main --
WHICH=${1:-list}
SEEDS=${2:-"1 2"}
EVALS=${3:-drop}

case "$EVALS" in
    drop) MIDEVAL="--eval-every 1000000000" ;;
    keep) MIDEVAL="" ;;
    *) echo "third argument must be drop or keep, got: $EVALS" >&2; exit 2 ;;
esac

case "$WHICH" in
    list)
        printf '%-16s %-28s %s\n' ROW BASE-TAG ENTRY
        rows | while IFS='|' read -r n script args base; do
            printf '%-16s %-28s %s\n' "$n" "$base" "$script"
        done
        printf '%-16s %-28s %s\n' gb1_dirac 'fs_gb1_k3_dirac_s<S>_{A,B,C}' iasbs/fixed_support.py
        printf '%-16s %-28s %s\n' gb1_nd    'fs_gb1_k3_nd_s<S>_{A,B,C}'    iasbs/fixed_support.py
        printf '%-16s %-28s %s\n' gb1_dam   'dam_fs_gb1_k3_K16_s<S>_{A,B,C}' dam/discrete.py
        ;;
    all)
        for n in $(all_rows); do run_seeds "$n" $SEEDS; done ;;
    group:last)
        for n in $LAST_ROWS; do run_seeds "$n" $SEEDS; done ;;
    group:main)
        for n in $(all_rows); do
            is_last "$n" && continue
            run_seeds "$n" $SEEDS
        done ;;
    group:*)
        g="${WHICH#group:}"
        for n in $(all_rows); do
            [ "$(group_of "$n")" = "$g" ] || continue
            run_seeds "$n" $SEEDS
        done ;;
    *)
        run_seeds "$WHICH" $SEEDS ;;
esac

echo "=== [$(date -u '+%F %T UTC')] DONE ($WHICH, seeds: $SEEDS, mid-evals: $EVALS)"
