#!/usr/bin/env bash
# Shared work pool for the multi-seed replication, one worker per GPU.
#
# Two hand-written per-GPU queues went wrong twice: they drifted out of balance
# as the real per-row costs came in, and a row ended up in both lists with two
# different mid-eval modes.  A pool fixes both -- workers pull the next
# unclaimed row, so there is no imbalance to rebalance and no row can be
# claimed twice.
#
# Claiming uses mkdir, which is atomic on a local filesystem: exactly one
# worker can create a given directory, and the loser moves to the next row.
#
#   bash iasbs/scripts/multiseed_pool.sh init <row> [<row> ...]
#   CUDA_VISIBLE_DEVICES=0 bash iasbs/scripts/multiseed_pool.sh work w0
#   bash iasbs/scripts/multiseed_pool.sh status
#
# Mid-eval mode is per row, not global.  occupation.py's scale-nondirac
# evaluation block draws samples (line 1558, torch.randint + simulate_scale),
# so it advances the training RNG stream; those rows must reproduce seed 0's
# cadence exactly and run with "keep".  Every other evaluation site in the
# four discrete mains is deterministic and under no_grad, so "drop" there
# leaves the RNG stream identical to seed 0's.
set -u
cd "$(dirname "$0")/../.."
POOL=${POOL:-.multiseed_pool}
SEEDS=${SEEDS:-"1 2"}

# Rows whose mid-run evaluation consumes RNG and must keep seed 0's cadence.
KEEP_ROWS="occs32_nd occs128_nd occs1000_nd"
mode_of() {
    case " $KEEP_ROWS " in *" $1 "*) echo keep ;; *) echo drop ;; esac
}

case "${1:-status}" in
init)
    shift
    mkdir -p "$POOL"
    : > "$POOL/rows"
    for r in "$@"; do echo "$r" >> "$POOL/rows"; done
    echo "pool $POOL initialised with $# row(s)"
    while read -r r; do echo "  $r  ($(mode_of "$r"))"; done < "$POOL/rows"
    ;;
work)
    who=${2:-w}
    echo "### worker $who up on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
    while read -r row; do
        [ -n "$row" ] || continue
        # atomic claim: only one worker can create this directory
        if mkdir "$POOL/claim_$row" 2>/dev/null; then
            m=$(mode_of "$row")
            echo "##### [$(date -u '+%F %T UTC')] $who CLAIMED $row ($m)"
            bash iasbs/scripts/multiseed.sh "$row" "$SEEDS" "$m"
            echo "$who $(date -u '+%F %T UTC')" > "$POOL/claim_$row/done"
            echo "##### [$(date -u '+%F %T UTC')] $who FINISHED $row"
        fi
    done < "$POOL/rows"
    echo "### worker $who: pool drained"
    ;;
status)
    tot=$(wc -l < "$POOL/rows" 2>/dev/null || echo 0)
    echo "pool $POOL: $tot row(s)"
    while read -r r; do
        if [ -f "$POOL/claim_$r/done" ]; then st="done  ($(cat "$POOL/claim_$r/done"))"
        elif [ -d "$POOL/claim_$r" ];   then st="running"
        else                                 st="queued"
        fi
        printf '  %-16s %-6s %s\n' "$r" "$(mode_of "$r")" "$st"
    done < "$POOL/rows" 2>/dev/null
    ;;
*)
    echo "usage: $0 {init <rows...>|work <name>|status}" >&2; exit 2 ;;
esac
