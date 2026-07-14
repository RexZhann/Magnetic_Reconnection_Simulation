#!/usr/bin/env bash
# Step 2: scaled smoke + kill -9 RESUME test ON THIS MACHINE (do not reuse
# the local-laptop conclusion).  64x32 scaled Sym, t_end=20, ckpt every t=10;
# kill -9 around t~14, RESUME=1, check L1 seam.
[ "$(hostname)" = "cerberus3" ] || { echo "WRONG HOST: $(hostname)"; exit 1; }
echo "host=$(hostname) nproc=$(nproc) load=$(uptime)" | tee -a run.log
df -h $HOME | tee -a run.log && quota -s | tee -a run.log
set -u
cd "$(dirname "$0")/.."
DIR=output/test29_campaign/smoke64
rm -rf "$DIR"
mkdir -p output/test29_campaign

OMP_NUM_THREADS=4 ./build/mhd2d 31 64 32 2 1 -1 20 test29_campaign/smoke64 1 \
    > "$DIR.log" 2>&1 &
PID=$!
echo "smoke running (pid $PID), waiting for t>=14 to kill -9 ..."
while sleep 2; do
    kill -0 $PID 2>/dev/null || { echo "finished before kill -- lower the wait"; break; }
    T=$(grep -v '^#' "$DIR/l1.csv" 2>/dev/null | tail -1 | cut -d, -f1)
    if [ -n "$T" ] && awk "BEGIN{exit !($T>=14)}"; then
        kill -9 $PID; echo "killed -9 at t=$T"; break
    fi
done
wait $PID 2>/dev/null

echo "--- RESUME=1 ---"
RESUME=1 OMP_NUM_THREADS=4 ./build/mhd2d 31 64 32 2 1 -1 20 test29_campaign/smoke64 1 \
    >> "$DIR.log" 2>&1
grep -a "RESUMED\|floor activations" "$DIR.log"

# seam check: t strictly increasing, no duplicates, reaches 20
awk -F, '!/^#/ && !/^t,/ {
    if ($1+0 <= prev+0 && NR>1 && prev!="") { bad=1 }
    prev=$1; last=$1 }
  END {
    if (bad) print "FAIL: L1 time not strictly increasing (duplicate/overlap rows)";
    else if (last+0 < 19.99) print "FAIL: did not reach t_end (last t=" last ")";
    else print "PASS: L1 seamless, " NR " rows to t=" last }' "$DIR/l1.csv"
