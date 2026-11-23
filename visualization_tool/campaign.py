"""Test-31 campaign analysis from the frozen L1 table.

Pre-registered criteria (unchanged from output/test29_campaign/analyze_tier.py):

- Rate (Method A): E_A = |d psi/dt| per sheet, combined = mean of both sheets,
  5-point running mean.
- Island Ly/2 crossing: armed after w_isl < Ly/4; must hold >= 5 time units
  over valid frames (pitfall #2, 2026-07-14). Degenerate frames w_isl == Ly
  are failed measurements and are masked (ruling 2026-07-18).
- Plateau: window length >= 20, all smoothed rates within +/-20% of the
  window median, median > 1e-3, searched at t > 10 and before the crossing;
  onset guard: median >= 0.5 x peak smoothed rate in the searched segment.
  Report the first qualifying window (primary) and the last one before the
  crossing (sensitivity).
"""

from __future__ import annotations

import os
import re

import numpy as np

from .io import L1Table, load_l1

SHEETS = ("up", "lo")


def smooth(v, n: int = 5) -> np.ndarray:
    return np.convolve(v, np.ones(n) / n, mode="same") if len(v) >= n else v


def method_a_rates(l1: L1Table):
    """Returns ``(tm, EA, EAc)``: midpoint times, per-sheet |dpsi/dt| dict,
    and the smoothed sheet-averaged rate."""
    T = l1.T
    tm = 0.5 * (T[:-1] + T[1:])
    EA = {s: np.abs(np.diff(l1[f"{s}_psi"])) / np.diff(T) for s in SHEETS}
    EAc = smooth(0.5 * (EA["up"] + EA["lo"]))
    return tm, EA, EAc


def island_times(T, wI, Ly, hold: float = 5.0):
    """Sustained Ly/2 crossing time and saturation time ``(t_cross, t_sat)``
    (armed + sustained + degenerate-masked; either may be None)."""
    degen = wI >= Ly - 1e-6            # failed-measurement sentinel frames
    valid = ~degen
    armed = np.zeros(len(wI), bool)
    seen = False
    for k, v in enumerate(wI):
        if valid[k] and v < 0.25 * Ly:
            seen = True
        armed[k] = seen
    ok = armed & valid & (wI >= 0.5 * Ly)
    kc = None
    for k in np.where(ok)[0]:
        m = (T >= T[k]) & (T <= T[k] + hold) & valid
        if m.sum() >= 3 and np.all(wI[m] >= 0.5 * Ly):
            kc = int(k)                # sustained over VALID frames only
            break
    if kc is None:
        return None, None
    t_cross = float(T[kc])
    tail_valid = valid & (np.arange(len(wI)) >= kc)
    wmax = wI[tail_valid].max() if tail_valid.any() else wI[kc]
    sat = tail_valid & (wI >= 0.98 * wmax)
    t_sat = float(T[np.argmax(sat)]) if sat.any() else None
    return t_cross, t_sat


def crossings(l1: L1Table):
    """Per-sheet ``(t_cross, t_sat)`` and the earliest crossing over sheets."""
    cross = {s: island_times(l1.T, l1[f"{s}_wisl"], l1.Ly) for s in SHEETS}
    t_min = min([v[0] for v in cross.values() if v[0] is not None], default=None)
    return cross, t_min


def _window_stats(tm, EAc, i, j):
    seg = EAc[i:j + 1]
    return dict(t0=tm[i], t1=tm[j], med=float(np.median(seg)),
                hiqr=float(0.5 * (np.percentile(seg, 75) - np.percentile(seg, 25))),
                i=i, j=j)


def all_windows(tm, EAc, t_hi):
    """Maximal qualifying plateau windows (one per admissible start index),
    searched in 10 < tm < t_hi, with the relative onset guard."""
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
            out.append(_window_stats(tm, EAc, i, jbest))
    return out


def plateau_windows(l1: L1Table, t_cut: float | None = None):
    """``(primary, sensitivity, unrestricted_longest)`` plateau windows.

    Upper search bound = earliest island crossing, further capped by
    ``t_cut`` if given (plot_tier validity cut)."""
    tm, _, EAc = method_a_rates(l1)
    _, t_cross = crossings(l1)
    bounds = [v for v in (t_cross, t_cut) if v is not None]
    t_hi = min(bounds) if bounds else float(tm[-1]) + 1
    wins = all_windows(tm, EAc, t_hi)
    first = wins[0] if wins else None
    last = max(wins, key=lambda w: w["t1"]) if wins else None
    wins_any = all_windows(tm, EAc, float(tm[-1]) + 1)
    longest = max(wins_any, key=lambda w: w["t1"] - w["t0"]) if wins_any else None
    return first, last, longest


# --------------------------------------------------------------------------
# Tier summary (Stage-0 verdict + hard checks); was analyze_tier.py
# --------------------------------------------------------------------------

def summarize_tier(run_dir: str, tier: str | None = None, write: bool = True) -> str:
    """Build the single-tier summary text; writes ``tier_<tier>_summary.txt``
    into ``run_dir`` when ``write`` is True. Top line is the verdict."""
    tier = tier or os.path.basename(os.path.normpath(run_dir))
    l1 = load_l1(run_dir)
    C, T, LX, LY = l1.C, l1.T, l1.Lx, l1.Ly
    PT_RMS0 = l1.meta.get("ptot_rms0", float("nan"))
    tm, EA, EAc = method_a_rates(l1)
    cross, t_cross_min = crossings(l1)
    plateau, plateau_last, plateau_any = plateau_windows(l1)

    # --- verdict: PASS = a qualifying plateau inside the pre-crossing segment
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

    lines = [verdict, "", f"tier {tier}  ({int(C['step'][-1])} steps to t={T[-1]:.1f}; "
             f"L1 rows {len(T)})", ""]
    if plateau_last and plateau and (plateau_last["t0"] != plateau["t0"]
                                     or plateau_last["t1"] != plateau["t1"]):
        lines.append(f"sensitivity window (last before crossing): "
                     f"[{plateau_last['t0']:.1f},{plateau_last['t1']:.1f}] "
                     f"E={plateau_last['med']:.4f}+/-{plateau_last['hiqr']:.4f}")
        lines.append("")

    def ezB_mid(s):
        return np.abs(0.5 * (C[f"{s}_ezB"][:-1] + C[f"{s}_ezB"][1:]))

    if plateau:
        mwin = (tm >= plateau["t0"]) & (tm <= plateau["t1"])
        lines.append("plateau window details (primary = first qualifying):")
        for s in SHEETS:
            eA, eB = EA[s][mwin], ezB_mid(s)[mwin]
            dev = abs(np.median(eA) - np.median(eB)) / np.median(eA) * 100
            lines.append(f"  [{s}] E_A={np.median(eA):.4f}+/-"
                         f"{0.5*(np.percentile(eA,75)-np.percentile(eA,25)):.4f}"
                         f"  E_B={np.median(eB):.4f}  A/B diff={dev:.1f}%")
        ab_max = max(abs(np.median(EA[s][mwin]) - np.median(ezB_mid(s)[mwin]))
                     / np.median(EA[s][mwin]) for s in SHEETS) * 100
    else:
        ab_max = float("nan")
        lines.append("plateau window details: n/a")

    lines += ["", "island evolution:"]
    for s in SHEETS:
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

    # --- hard checks ---
    psi_u, psi_l = C["up_psi"], C["lo_psi"]
    m10 = T > 10

    def sym(m):
        return float(np.max(np.abs(psi_u[m] - psi_l[m])
                            / np.maximum(np.maximum(psi_u[m], psi_l[m]), 1e-12))) * 100

    sym_dev = sym(m10)
    sym_dev_seg = sym(m10 & (T < t_cross_min)) if t_cross_min is not None else sym_dev
    floors = (C["floor_rho"][-1], C["floor_p"][-1], C["fallback"][-1])
    hard = [
        # 0.5%: the CS2008 double-sheet perturbation is deliberately NOT
        # pressure-balanced; pilot run250 measured 0.176% and was accepted.
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
    lines += ["", "hard checks (4 judged + symmetry report):"]
    npass = 0
    for name, val, ok in hard:
        npass += bool(ok)
        lines.append(f"  [{'OK' if ok else 'FAIL'}] {name}: {val}")
    lines.append(f"  -> {npass}/4 passed")
    # ruling: symmetry REPORTED per sheet, no threshold
    lines.append(f"  [REPORT] double-sheet psi divergence: {sym_dev:.1f}% full run, "
                 f"{sym_dev_seg:.1f}% pre-crossing segment (no threshold per ruling)")

    # --- X-line drift (x: self-check; y: CS2008 [22] physics signal) ---
    lines += ["", "X-line drift:"]
    for s in SHEETS:
        m = T > 50
        if m.sum() > 10:
            xu = np.unwrap(C[f"{s}_xX"][m], period=LX)
            vX = float(np.polyfit(T[m], xu, 1)[0])
            lines.append(f"  [{s}] along-sheet v_X,x = {vX:+.4f}  (fit t>50)")
    B01, B02 = l1.meta.get("B01"), l1.meta.get("B02")
    vy_sheet = {}
    if "up_yX" in C:
        mseg2 = (T > 10) & ((T < t_cross_min) if t_cross_min else np.ones(len(T), bool))
        for s, sgn_out in (("up", +1.0), ("lo", -1.0)):
            vy_sheet[s] = float(np.polyfit(T[mseg2], C[f"{s}_yX"][mseg2], 1)[0])
            direction = ("outward (toward B01 side)" if vy_sheet[s] * sgn_out > 0
                         else "inward (toward B02 side)")
            lines.append(f"  [{s}] inflow-direction v_X,y = {vy_sheet[s]:+.5f} "
                         f"-> {direction}")
        if B01 is not None and B02 is not None and abs(B01 - B02) > 1e-12:
            strong = "B02 (inner)" if B02 > B01 else "B01 (outer)"
            lines.append(f"  expectation (CS2008 [22]): drift toward stronger field "
                         f"side = {strong}; geometry: B02>B01 means both sheets "
                         f"drift toward y=0, B01>B02 means away from y=0")

    # --- inflow in the co-moving frame (v - v_X) ---
    if "up_vinp" in C and plateau:
        lines += ["", "inflow (signed vy at X column, +/-3.0 from sheet; "
                      "co-moving correction = value - v_X,y):"]
        mv = (T >= plateau["t0"]) & (T <= plateau["t1"])
        for s in SHEETS:
            vp = float(np.median(C[f"{s}_vinp"][mv]))
            vm = float(np.median(C[f"{s}_vinm"][mv]))
            vs = vy_sheet.get(s, 0.0)
            lines.append(f"  [{s}] raw: vin(+3)={vp:+.4f} vin(-3)={vm:+.4f}; "
                         f"co-moving: {vp - vs:+.4f} / {vm - vs:+.4f} "
                         f"(v_X,y={vs:+.5f})")

    # --- outflow ---
    lines += ["", "outflow (sheet-band max |vx|):"]
    for s in SHEETS:
        q = ""
        if plateau:
            mv = (T >= plateau["t0"]) & (T <= plateau["t1"])
            q = f", plateau median={np.median(C[f'{s}_vout'][mv]):.3f}"
        lines.append(f"  [{s}] run max={C[f'{s}_vout'].max():.3f}{q}")

    # --- cost ---
    lines.append("")
    wall = "n/a"
    runlog = os.path.join(run_dir, "run.log")
    if os.path.exists(runlog):
        mm = re.findall(r"Total wall time\s*:\s*([0-9.]+)",
                        open(runlog, errors="ignore").read())
        if mm:
            wall = f"{sum(float(v) for v in mm)/3600:.2f} h (sum over segments)"
    size = sum(os.path.getsize(os.path.join(run_dir, f)) for f in os.listdir(run_dir))
    lines.append(f"cost: wall={wall}; disk={size/1e9:.2f} GB in {run_dir}")

    text = "\n".join(lines) + "\n"
    if write:
        with open(os.path.join(run_dir, f"tier_{tier}_summary.txt"), "w") as f:
            f.write(text)
    return text
