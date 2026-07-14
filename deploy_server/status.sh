#!/usr/bin/env bash
# progress check on cerberus3:  bash deploy_server/status.sh [tier]
[ "$(hostname)" = "cerberus3" ] || { echo "WRONG HOST: $(hostname)"; exit 1; }
echo "host=$(hostname) nproc=$(nproc) load=$(uptime)"
df -h $HOME | tail -1 && quota -s | tail -2
cd "$(dirname "$0")/../output/test29_campaign" 2>/dev/null || { echo "no campaign dir yet"; exit 0; }
TIER="${1:-$(ls -td */ 2>/dev/null | head -1 | tr -d /)}"
L1="$TIER/l1.csv"
[ -f "$L1" ] || { echo "no $L1 yet"; exit 0; }
LAST=$(grep -v '^#' "$L1" | tail -1)
echo "== $TIER ==  t = $(echo "$LAST" | cut -d, -f1) / 400  ($(grep -vc '^#' "$L1") L1 rows)"
echo "counters (floor_rho,floor_p,fallback): $(echo "$LAST" | cut -d, -f10-12)"
echo "L2 frames:"; ls "$TIER"/l2_*.f32 2>/dev/null | sed 's/^/  /'
echo "disk: $(du -sh "$TIER" | cut -f1)   ckpt: $(ls -lh "$TIER/ckpt.bin" 2>/dev/null | awk '{print $5, $6, $7, $8}')"
grep -v '^#' "$L1" | tail -42 | awk -F, '
  NR>1 { ea=0.5*((($13>p13)?$13-p13:p13-$13)+(($19>p19)?$19-p19:p19-$19))/($1-p1); s+=ea; n++ }
  { p1=$1; p13=$13; p19=$19 }
  END { if (n>0) printf "recent Method-A rate (last %d rows): %.4f\n", n, s/n }'
