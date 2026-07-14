"""
test29 campaign: single-tier summary from the frozen L1 table alone.

Usage:  python output/test29_campaign/analyze_tier.py [Sym]
Writes: output/test29_campaign/<tier>/tier_<tier>_summary.txt

Top line is the Stage 0 verdict (PASS/FAIL + numbers, no prose), per the
pre-registered criterion: PASS = the plateau window lies entirely inside the
island-width < Ly/2 segment (plateau precedes mutual inductance).
Pre-registered plateau: longest window with len >= 20, all smoothed Method-A
rates within +/-20% of the window median, median > 1e-3, searched at t > 10.
Island Ly/2-crossing uses the frozen arming rule (armed after wisl < Ly/4).
"""

import os, re, sys
import numpy as np

TIER = sys.argv[1] if len(sys.argv) > 1 else "Sym"
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), TIER)
L1 = os.path.join(DIR, "l1.csv")

hdr_meta = {}
with open(L1) as f:
    for line in f:
        if line.startswith("#"):
            for k, v in re.findall(r"(\w+)\s*=\s*([0-9.eE+-]+)", line):
                try:
                    hdr_meta[k] = float(v)
                except ValueError:
                    pass
        if line.startswith("t,"):
            cols = line.strip().split(",")
            break
# plain parse (robust to the header line)
rows = []
with open(L1) as f:
    for line in f:
        if line.startswith("#") or line.startswith("t,") or not line.strip():
            continue
        rows.append([float(v) for v in line.strip().split(",")])
A = np.array(rows)
C = {name: A[:, k] for k, name in enumerate(cols)}
T = C["t"]
LX, LY = hdr_meta.get("Lx", 102.4), hdr_meta.get("Ly", 51.2)
PT_RMS0 = hdr_meta.get("ptot_rms0", float("nan"))


def smooth(v, n=5):
    return np.convolve(v, np.ones(n) / n, mode="same") if len(v) >= n else v


tm = 0.5 * (T[:-1] + T[1:])
EA = {s: np.abs(np.diff(C[f"{s}_psi"])) / np.diff(T) for s in ("up", "lo")}
EAc = smooth(0.5 * (EA["up"] + EA["lo"]))

# --- island crossing / saturation (armed rule + SUSTAINED rule) ---
# 2026-07-14: added the pre-registered sustained criterion (pitfall #2 in the
# handover ruling: crossing must HOLD >= 5 time units).  AB1 has early X/O-swap
# frames where wisl flips between the degenerate full-box value (Ly) and the
# true small width; the instantaneous test fired a false crossing at t=3.0
# (visual truth ~176).  Sym is unaffected (re-run verified identical verdict).
def island_times(wI):
    armed = np.zeros(len(wI), bool)
    seen = False
    for k, v in enumerate(wI):
        if v < 0.25 * LY:
            seen = True
        armed[k] = seen
    ok = armed & (wI >= 0.5 * LY)
    kc = None
    for k in np.where(ok)[0]:
        hold = (T >= T[k]) & (T <= T[k] + 5.0)
        if np.all(ok[hold]):          # sustained: holds for 5 time units
            kc = int(k)
            break
    if kc is None:
        return None, None
    t_cross = float(T[kc])
    wmax = wI[kc:].max()
    sat = (wI >= 0.98 * wmax) & (np.arange(len(wI)) >= kc)
    t_sat = float(T[np.argmax(sat)]) if sat.any() else None
    return t_cross, t_sat

cross = {s: island_times(C[f"{s}_wisl"]) for s in ("up", "lo")}
t_cross_min = min([v[0] for v in cross.values() if v[0] is not None],
                  default=None)

# --- plateau (pre-registered: len>=20, +/-20%, searched WITHIN the
#     island-width < Ly/2 segment, i.e. tm in (10, t_cross)).
#     RULING (nine-tier uniform): report TWO windows -- the primary
#     "first qualifying window" and the sensitivity "last qualifying
#     window before crossing"; the unrestricted longest window is kept
#     as a transparency line. ---
def window_stats(i, j):
    seg = EAc[i:j + 1]
    return dict(t0=tm[i], t1=tm[j], med=float(np.median(seg)),
                hiqr=float(0.5 * (np.percentile(seg, 75)
                                  - np.percentile(seg, 25))), i=i, j=j)


def all_windows(t_hi):
    """Maximal qualifying windows (one per admissible start index).

    Onset guard (relative, no absolute tuning): a window only qualifies if
    its median rate >= 50% of the peak smoothed rate in the searched
    segment -- otherwise the pre-onset slow-growth creep produces a
    formally-flat but physically meaningless "first window"."""
    idx = np.where((tm > 10) & (tm < t_hi))[0]
    if len(idx) == 0:
        return []
    floor = 0.5 * float(EAc[idx].max())
    out = []
    for a in range(len(idx)):
        i = idx[a]
        jbest = None
        for b in range(a + 10, len(idx)):
            j = idx[b]
            seg = EAc[i:j + 1]
            med = np.median(seg)
            if med <= 1e-3 or not np.all(np.abs(seg - med) < 0.2 * med):
                break
            if tm[j] - tm[i] >= 20 and med >= floor:
                jbest = j
        if jbest is not None:
            out.append(window_stats(i, jbest))
    return out

t_hi = t_cross_min if t_cross_min is not None else float(tm[-1]) + 1
wins = all_windows(t_hi)
plateau = wins[0] if wins else None                       # first qualifying
plateau_last = max(wins, key=lambda w: w["t1"]) if wins else None  # sensitivity
wins_any = all_windows(float(tm[-1]) + 1)
plateau_any = (max(wins_any, key=lambda w: w["t1"] - w["t0"])
               if wins_any else None)

# --- verdict (Stage-0-style gate: PASS = a qualifying plateau exists
#     inside the pre-crossing segment; primary = first window) ---
if plateau is not None:
    tail = (f"Ly/2 crossing t={t_cross_min:.1f} > t1" if t_cross_min is not None
            else f"island never reached Ly/2 "
                 f"(w_max={max(C['up_wisl'].max(), C['lo_wisl'].max()):.1f})")
    verdict = (f"PASS: first plateau=[{plateau['t0']:.1f},{plateau['t1']:.1f}] "
               f"E={plateau['med']:.4f}+/-{plateau['hiqr']:.4f}; " + tail)
elif plateau_any is not None:
    verdict = (f"FAIL: only plateau=[{plateau_any['t0']:.1f},"
               f"{plateau_any['t1']:.1f}] E={plateau_any['med']:.4f} "
               f"lies (partly) in the mutual-inductance segment "
               f"(Ly/2 crossing t={t_cross_min:.1f})")
else:
    verdict = "FAIL: no pre-registered plateau anywhere (>=20-long +/-20% window, t>10)"

# --- per-sheet plateau values + A/B cross in the window ---
lines = [verdict, "", f"tier {TIER}  ({int(C['step'][-1])} steps to t={T[-1]:.1f}; "
         f"L1 rows {len(T)})", ""]
if plateau_last and plateau and (plateau_last["t0"] != plateau["t0"]
                                 or plateau_last["t1"] != plateau["t1"]):
    lines.append(f"sensitivity window (last before crossing): "
                 f"[{plateau_last['t0']:.1f},{plateau_last['t1']:.1f}] "
                 f"E={plateau_last['med']:.4f}+/-{plateau_last['hiqr']:.4f}")
    lines.append("")
if plateau:
    w0, w1 = plateau["t0"], plateau["t1"]
    mwin = (tm >= w0) & (tm <= w1)
    lines.append("plateau window details (primary = first qualifying):")
    for s in ("up", "lo"):
        eA = EA[s][mwin]
        eB = np.abs(0.5 * (C[f"{s}_ezB"][:-1] + C[f"{s}_ezB"][1:]))[mwin]
        dev = abs(np.median(eA) - np.median(eB)) / np.median(eA) * 100
        lines.append(f"  [{s}] E_A={np.median(eA):.4f}+/-"
                     f"{0.5*(np.percentile(eA,75)-np.percentile(eA,25)):.4f}"
                     f"  E_B={np.median(eB):.4f}  A/B diff={dev:.1f}%")
    ab_max = max(
        abs(np.median(EA[s][mwin])
            - np.median(np.abs(0.5*(C[f"{s}_ezB"][:-1]+C[f"{s}_ezB"][1:]))[mwin]))
        / np.median(EA[s][mwin]) for s in ("up", "lo")) * 100
else:
    ab_max = float("nan")
    lines.append("plateau window details: n/a")

lines.append("")
lines.append("island evolution:")
for s in ("up", "lo"):
    tc, ts = cross[s]
    lines.append(f"  [{s}] Ly/2({0.5*LY:.1f}) crossing t="
                 f"{tc if tc is not None else 'none'};  saturation t="
                 f"{ts if ts is not None else 'none'};  "
                 f"w_final={C[f'{s}_wisl'][-1]:.1f}")
if plateau_any and (not plateau or plateau_any["t0"] != plateau["t0"]
                    or plateau_any["t1"] != plateau["t1"]):
    lines.append(f"  (transparency: unrestricted longest window = "
                 f"[{plateau_any['t0']:.1f},{plateau_any['t1']:.1f}] "
                 f"E={plateau_any['med']:.4f}, not used for the verdict "
                 f"per the pre-registered in-segment search)")

# --- hard checks x5 ---
psi_u, psi_l = C["up_psi"], C["lo_psi"]
m10 = T > 10
sym_dev = float(np.max(np.abs(psi_u[m10] - psi_l[m10])
                       / np.maximum(np.maximum(psi_u[m10], psi_l[m10]), 1e-12))) * 100
if t_cross_min is not None:
    mseg = m10 & (T < t_cross_min)
    sym_dev_seg = float(np.max(np.abs(psi_u[mseg] - psi_l[mseg])
                        / np.maximum(np.maximum(psi_u[mseg], psi_l[mseg]), 1e-12))) * 100
else:
    sym_dev_seg = sym_dev
floors = (C["floor_rho"][-1], C["floor_p"][-1], C["fallback"][-1])
hard = [
    # threshold 0.5%: the CS2008 double-sheet perturbation is deliberately
    # NOT pressure-balanced (unlike the test27 kick, whose target was 0.1%);
    # pilot run250 measured 0.176% and was accepted in review.
    ("t=0 pressure-balance RMS <= 0.5% (double-sheet convention)",
     f"{PT_RMS0*100:.4f}%", PT_RMS0 <= 5e-3),
    ("div-B (max over run) <= 1e-10", f"{C['max_divb'].max():.2e}",
     C["max_divb"].max() <= 1e-10),
    ("min_rho far from floor + zero floor/fallback counters",
     f"min_rho={C['min_rho'].min():.3f}, counters={tuple(int(v) for v in floors)}",
     C["min_rho"].min() > 0.05 and all(v == 0 for v in floors)),
    ("Method A/B cross-check in plateau < 10%",
     f"{ab_max:.1f}%" if np.isfinite(ab_max) else "n/a (no plateau)",
     bool(np.isfinite(ab_max) and ab_max < 10.0)),
]
lines.append("")
lines.append("hard checks (4 judged + symmetry report):")
npass = 0
for name, val, ok in hard:
    npass += bool(ok)
    lines.append(f"  [{'OK' if ok else 'FAIL'}] {name}: {val}")
lines.append(f"  -> {npass}/4 passed")
# ruling: symmetry is REPORTED per sheet, no 5% threshold (asym tiers
# legitimately diverge; number stays visible for judgement)
lines.append(f"  [REPORT] double-sheet psi divergence: {sym_dev:.1f}% full run, "
             f"{sym_dev_seg:.1f}% pre-crossing segment (no threshold per ruling)")

# --- X-line drift: along-sheet (x) self-check + inflow-direction (y)
#     physics signal (AB/ABN tiers; CS2008 [22]: sheet drifts along the
#     inflow toward the STRONGER-field plasma) ---
lines.append("")
lines.append("X-line drift:")
for s in ("up", "lo"):
    m = T > 50
    if m.sum() > 10:
        xu = np.unwrap(C[f"{s}_xX"][m], period=LX)
        vX = float(np.polyfit(T[m], xu, 1)[0])
        lines.append(f"  [{s}] along-sheet v_X,x = {vX:+.4f}  (fit t>50)")
B01, B02 = hdr_meta.get("B01"), hdr_meta.get("B02")
vy_sheet = {}
if "up_yX" in C:
    mseg2 = (T > 10) & ((T < t_cross_min) if t_cross_min else np.ones(len(T), bool))
    for s, sgn_out in (("up", +1.0), ("lo", -1.0)):
        yfit = np.polyfit(T[mseg2], C[f"{s}_yX"][mseg2], 1)
        vy_sheet[s] = float(yfit[0])
        direction = "outward (toward B01 side)" if vy_sheet[s] * sgn_out > 0 \
                    else "inward (toward B02 side)"
        lines.append(f"  [{s}] inflow-direction v_X,y = {vy_sheet[s]:+.5f} "
                     f"-> {direction}")
    if B01 is not None and B02 is not None and abs(B01 - B02) > 1e-12:
        strong = "B02 (inner)" if B02 > B01 else "B01 (outer)"
        lines.append(f"  expectation (CS2008 [22]): drift toward stronger field "
                     f"side = {strong}; geometry: B02>B01 means both sheets "
                     f"drift toward y=0, B01>B02 means away from y=0")

# --- inflow diagnostics in the co-moving frame (v - v_X), ruling ---
if "up_vinp" in C and plateau:
    lines.append("")
    lines.append("inflow (signed vy at X column, +/-3.0 from sheet; "
                 "co-moving correction = value - v_X,y):")
    mv = (T >= plateau["t0"]) & (T <= plateau["t1"])
    for s in ("up", "lo"):
        vp = float(np.median(C[f"{s}_vinp"][mv]))
        vm = float(np.median(C[f"{s}_vinm"][mv]))
        vs = vy_sheet.get(s, 0.0)
        lines.append(f"  [{s}] raw: vin(+3)={vp:+.4f} vin(-3)={vm:+.4f}; "
                     f"co-moving: {vp - vs:+.4f} / {vm - vs:+.4f} "
                     f"(v_X,y={vs:+.5f})")

# --- outflow ---
lines.append("")
lines.append("outflow (sheet-band max |vx|):")
for s in ("up", "lo"):
    q = ""
    if plateau:
        mv = (T >= plateau["t0"]) & (T <= plateau["t1"])
        q = f", plateau median={np.median(C[f'{s}_vout'][mv]):.3f}"
    lines.append(f"  [{s}] run max={C[f'{s}_vout'].max():.3f}{q}")

# --- cost ---
lines.append("")
wall = "n/a"
runlog = os.path.join(DIR, "run.log")
if os.path.exists(runlog):
    txt = open(runlog, errors="ignore").read()
    mm = re.findall(r"Total wall time\s*:\s*([0-9.]+)", txt)
    if mm:
        wall = f"{sum(float(v) for v in mm)/3600:.2f} h (sum over segments)"
size = sum(os.path.getsize(os.path.join(DIR, f)) for f in os.listdir(DIR))
lines.append(f"cost: wall={wall}; disk={size/1e9:.2f} GB in {DIR}")

out = os.path.join(DIR, f"tier_{TIER}_summary.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n-> {out}")
