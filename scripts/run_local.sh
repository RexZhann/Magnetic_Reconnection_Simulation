#!/usr/bin/env bash
# Build and run on a local machine (laptop / workstation).
#
#   ./scripts/run_local.sh                       # uses config/ctrl.txt
#   ./scripts/run_local.sh config/ctrl_smoke.txt # any other deck
#   OMP_NUM_THREADS=4 ./scripts/run_local.sh     # choose thread count
#
# Output and a log (run.log) go to <output_dir>/<label>/ as set in the deck.
set -euo pipefail
cd "$(dirname "$0")/.."

CTRL="${1:-config/ctrl.txt}"
[ -f "$CTRL" ] || { echo "control deck not found: $CTRL"; exit 1; }

# Default thread count: all cores
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    OMP_NUM_THREADS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi
export OMP_NUM_THREADS

make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# Resolved output folder (expands ~ and $VARS exactly as the code does)
OUT="$(./build/mhd2d --print-outdir "$CTRL")"
mkdir -p "$OUT"

echo "deck=$CTRL  threads=$OMP_NUM_THREADS  output=$OUT  host=$(hostname)" | tee -a "$OUT/run.log"
./build/mhd2d "$CTRL" 2>&1 | tee -a "$OUT/run.log"
