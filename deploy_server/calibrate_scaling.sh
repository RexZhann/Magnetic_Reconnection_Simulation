#!/usr/bin/env bash
# Step 3 (revised): cost calibration + thread scaling scan.
#   - FULL-SIZE 1024x512, t=2 short test -> extrapolation is a pure x200
#     in time, no cell-ratio / cache-penalty guesswork;
#   - threads 8/16/24/32 (48 dropped: hyperthread lanes already falsified);
#   - uptime printed before every timing (record neighbour load);
#   - every setting run TWICE, faster value kept (neighbour jitter);
#   - extra pinned runs OMP_PROC_BIND=close OMP_PLACES=cores at 16/24
#     (thread-core binding: often 5-15% free gain for bandwidth-bound code,
#     prevents cross-socket hopping on dual-socket machines).
# Writes the winner to OMP_RECOMMENDED; if a pinned variant wins, also
# writes OMP_BIND_RECOMMENDED (run_tier.sh exports the bind env when present).
[ "$(hostname)" = "cerberus3" ] || { echo "WRONG HOST: $(hostname)"; exit 1; }
echo "host=$(hostname) nproc=$(nproc) load=$(uptime)" | tee -a run.log
df -h $HOME | tee -a run.log && quota -s | tee -a run.log
set -u
cd "$(dirname "$0")/.."

run_one () {   # $1=threads $2=bind(0/1) -> echoes best wall seconds of 2 reps
    local NT=$1 BIND=$2 BEST="" W
    for rep in 1 2; do
        rm -rf output/test29_campaign/calib
        echo "  [pre-run uptime] $(uptime)" >&2
        if [ "$BIND" = 1 ]; then
            W=$(OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$NT \
                ./build/mhd2d 31 1024 512 2 1 -1 2 test29_campaign/calib 2>&1 \
                | grep -a "Total wall time" | awk '{print $5}')
        else
            W=$(OMP_NUM_THREADS=$NT \
                ./build/mhd2d 31 1024 512 2 1 -1 2 test29_campaign/calib 2>&1 \
                | grep -a "Total wall time" | awk '{print $5}')
        fi
        [ -n "$W" ] || { echo "FAIL" ; return; }
        echo "    rep$rep: ${W}s" >&2
        if [ -z "$BEST" ] || awk "BEGIN{exit !($W < $BEST)}"; then BEST=$W; fi
    done
    echo "$BEST"
}

echo "config          | wall(t=2) best-of-2 | t=400 estimate"
declare -A RES
for CFG in "8 0" "16 0" "24 0" "32 0" "16 1" "24 1"; do
    read -r NT BIND <<< "$CFG"
    TAG="${NT}$([ "$BIND" = 1 ] && echo '+bind')"
    echo "-- testing $TAG threads --" >&2
    W=$(run_one $NT $BIND)
    [ "$W" = "FAIL" ] && { echo "$TAG | RUN FAILED"; continue; }
    RES[$TAG]=$W
    printf "%-15s | %10ss | %.1f h\n" "$TAG" "$W" \
        "$(awk "BEGIN{printf \"%.1f\", $W*200/3600}")"
done

# knee rule on unbound ladder: stop when marginal gain < 10%
BEST_NT=8; PREV=${RES[8]:-}
for NT in 16 24 32; do
    W=${RES[$NT]:-}; [ -n "$W" ] && [ -n "$PREV" ] || continue
    if awk "BEGIN{exit !( ($PREV-$W)/$PREV >= 0.10 )}"; then BEST_NT=$NT; PREV=$W
    else break; fi
done
# pinned variants compete for the crown
WINNER=$BEST_NT; WBEST=${RES[$BEST_NT]}; BINDWIN=0
for TAG in "16+bind" "24+bind"; do
    W=${RES[$TAG]:-}; [ -n "$W" ] || continue
    if awk "BEGIN{exit !($W < $WBEST)}"; then
        WINNER=${TAG%+bind}; WBEST=$W; BINDWIN=1
    fi
done
echo "$WINNER" > OMP_RECOMMENDED
rm -f OMP_BIND_RECOMMENDED
[ "$BINDWIN" = 1 ] && { touch OMP_BIND_RECOMMENDED; }
echo "winner: ${WINNER}$([ $BINDWIN = 1 ] && echo ' (with OMP_PROC_BIND=close OMP_PLACES=cores)') " \
     "-> OMP_RECOMMENDED$([ $BINDWIN = 1 ] && echo ' + OMP_BIND_RECOMMENDED')"
echo "t=400 estimate at winner: $(awk "BEGIN{printf \"%.1f\", $WBEST*200/3600}") h"
