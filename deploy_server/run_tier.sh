#!/usr/bin/env bash
# test29 campaign, cerberus3: launch one tier (1024x512, t_end=400).
#   OMP_NUM_THREADS=$(cat OMP_RECOMMENDED) nohup bash deploy_server/run_tier.sh AB1 > ab1.log 2>&1 &
#   RESUME=1 ... to continue from output/test29_campaign/<tier>/ckpt.bin
[ "$(hostname)" = "cerberus3" ] || { echo "WRONG HOST: $(hostname)"; exit 1; }
echo "host=$(hostname) nproc=$(nproc) load=$(uptime)" | tee -a run.log
df -h $HOME | tee -a run.log && quota -s | tee -a run.log
set -euo pipefail
cd "$(dirname "$0")/.."

TIER="${1:?usage: run_tier.sh <tier>}"

# CS2008 Table 1 (paper-verified 2026-07; note ABN2 rho02 = 4, the earlier
# working table had 2 -- fixed after checking T2=0.594=2.375/4 in the PDF):
#        B01  B02  r01  r02
declare -A TB=(
  [Sym]="1 1 1 1"     [AB1]="1 2 1 1"    [AB2]="1 3 1 1"   [AB3]="1 0.5 1 1"
  [AN1]="1 1 1 2"     [AN2]="1 1 1 3"    [AN3]="1 1 1 0.5"
  [ABN1]="2 1 1 2"    [ABN2]="1 0.5 1 4"
)
[ -n "${TB[$TIER]:-}" ] || { echo "unknown tier '$TIER' (have: ${!TB[*]})"; exit 1; }
read -r B01 B02 R01 R02 <<< "${TB[$TIER]}"

# quota hard gate: abort if remaining < 500 MB (a tier writes ~0.3 GB;
# running out of quota mid-run wastes ~15 h)
REM_KB=$(quota -w 2>/dev/null | awk 'END{gsub(/\*/,"",$2); if ($3+0>0) print ($3-$2)}')
if [ -n "${REM_KB:-}" ] && [ "$REM_KB" -lt 512000 ]; then
    echo "ABORT: quota remaining ${REM_KB} KB < 500 MB -- retrieve+delete finished tiers first"
    exit 1
fi
[ -z "${REM_KB:-}" ] && echo "WARN: could not parse quota, proceeding on df numbers above"

NX=1024; NY=512; TEND=400
OUT="output/test29_campaign/$TIER"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(cat OMP_RECOMMENDED 2>/dev/null || echo 16)}"
# calibrate_scaling.sh 若判定绑核更快，会留下 OMP_BIND_RECOMMENDED 标记
if [ -f OMP_BIND_RECOMMENDED ]; then
    export OMP_PROC_BIND=close OMP_PLACES=cores
    echo "thread binding ON (OMP_PROC_BIND=close OMP_PLACES=cores)"
fi
mkdir -p "$OUT"
echo "tier=$TIER B01=$B01 B02=$B02 rho01=$R01 rho02=$R02 threads=$OMP_NUM_THREADS resume=${RESUME:-0}"

# CLI: test nx ny divb solver etaH t_end label psi0 E0 b2 rho1 B01 B02 rho01 rho02
./build/mhd2d 31 $NX $NY 2 1 -1 $TEND "test29_campaign/$TIER" 1 0 0 0 \
    "$B01" "$B02" "$R01" "$R02" >> "$OUT/run.log" 2>&1
RC=$?

md5sum "$OUT"/* > "output/test29_campaign/${TIER}.md5" 2>/dev/null || true
echo "=============================================================="
echo "TIER DONE (exit=$RC) -- retrieve with:"
echo "  scp -r yz2019@cerberus3:$(pwd)/output/test29_campaign/$TIER <本地路径>"
echo "  scp yz2019@cerberus3:$(pwd)/output/test29_campaign/${TIER}.md5 <本地路径>"
echo "verify locally (md5sum -c ${TIER}.md5), THEN delete manually on server:"
echo "  rm -rf $(pwd)/output/test29_campaign/$TIER   # <- NOT automated, by design"
echo "=============================================================="
exit $RC
