"""
Server-side quick-look figures for one campaign tier (self-contained: only
numpy + matplotlib, reads L1 csv + L2 .f32 frames, no local-repo scripts).

Usage (repo root on cerberus3):
    python3 deploy_server/plot_tier.py AB1

Writes into output/test29_campaign/<tier>/:
    <tier>_rate.png           flux + Method A/B rate, windows, Ly/2 crossing
    <tier>_island.png         island width (armed rule), crossing/saturation
    <tier>_keyframes_jz.png   Jz + field lines, up to 4 L2 frames
    <tier>_keyframes_rho.png  density, shared colour scale
    <tier>_drift_inflow.png   v_out / x_X / y_X drift / co-moving inflow

Window logic mirrors analyze_tier.py (pre-registered: len>=20, +/-20%,
median > 1e-3, onset guard med >= 0.5 x segment peak, searched inside the
island<Ly/2 segment; primary = first, sensitivity = last before crossing).
The authoritative summary is still the local analyze_tier.py run.
"""

import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5,
                     "axes.labelsize": 12, "legend.fontsize": 10})

TIER = sys.argv[1] if len(sys.argv) > 1 else "AB1"
DIR = os.path.join("output", "test29_campaign", TIER)
L1 = os.path.join(DIR, "l1.csv")
COL = {"up": "C0", "lo": "C3"}
LABEL = {"up": "upper sheet", "lo": "lower sheet"}

# Analysis validity cut: after t~176 (AB1) the island width reaches Ly/2,
# the two sheets couple and the single-X-line diagnostic frame breaks down.
# Time-series *_tcut figures and ALL statistics are restricted to t <= T_CUT;
# the island-width figure and the 2D snapshots keep the full run.
T_CUT = 170.0
TCUT_XLIM = (0.0, 180.0)

# ---------- L1 ----------
meta, cols, rows = {}, None, []
for line in open(L1):
    if line.startswith("#"):
        for k, v in re.findall(r"(\w+)\s*=\s*([0-9.eE+-]+)", line):
            try: meta[k] = float(v)
            except ValueError: pass
    elif line.startswith("t,"):
        cols = line.strip().split(",")
    elif line.strip():
        rows.append([float(v) for v in line.split(",")])
A = np.array(rows)
C = {n: A[:, k] for k, n in enumerate(cols)}
T = C["t"]
LX, LY = meta["Lx"], meta["Ly"]
B01, B02 = meta.get("B01", 1.0), meta.get("B02", 1.0)
tm = 0.5 * (T[:-1] + T[1:])


def smooth(v, n=5):
    return np.convolve(v, np.ones(n) / n, mode="same") if len(v) >= n else v


EA = {s: np.abs(np.diff(C[f"{s}_psi"])) / np.diff(T) for s in ("up", "lo")}
EAc = smooth(0.5 * (EA["up"] + EA["lo"]))

# island crossing (armed rule + SUSTAINED >=5 time units) ------------------
# 2026-07-14: the window cap previously used the instantaneous test; AB1's
# early X/O-swap frames fired a false crossing at t=3.0 and emptied the
# window search.  Now identical to the fig-2 star and analyze_tier.py.
def island_times(wI):
    seen, ok = False, np.zeros(len(wI), bool)
    for k, v in enumerate(wI):
        if v < 0.25 * LY: seen = True
        ok[k] = seen and v >= 0.5 * LY
    for k in np.where(ok)[0]:
        hold = (T >= T[k]) & (T <= T[k] + 5.0)
        if np.all(ok[hold]):
            return float(T[k])
    return None

cross = {s: island_times(C[f"{s}_wisl"]) for s in ("up", "lo")}
t_cross_min = min([v for v in cross.values() if v is not None], default=None)

# plateau windows (mirror of analyze_tier.py) ------------------------------
def all_windows(t_hi):
    idx = np.where((tm > 10) & (tm < t_hi))[0]
    if len(idx) == 0: return []
    floor = 0.5 * float(EAc[idx].max())
    out = []
    for a in range(len(idx)):
        i = idx[a]; jbest = None
        for b in range(a + 10, len(idx)):
            j = idx[b]
            seg = EAc[i:j + 1]; med = np.median(seg)
            if med <= 1e-3 or not np.all(np.abs(seg - med) < 0.2 * med): break
            if tm[j] - tm[i] >= 20 and med >= floor: jbest = j
        if jbest is not None:
            seg = EAc[i:jbest + 1]
            out.append(dict(t0=tm[i], t1=tm[jbest], med=float(np.median(seg))))
    return out

# window search upper bound: island crossing AND the validity cut
# (onset-based start criterion unchanged; only the upper bound is capped)
t_hi_eff = min([v for v in (t_cross_min, T_CUT) if v is not None])
wins = all_windows(t_hi_eff)
W1 = wins[0] if wins else None
W2 = max(wins, key=lambda w: w["t1"]) if wins else None


def tcut_line(ax):
    ax.axvline(T_CUT, color="gray", ls="--", lw=1.1)
    ax.text(T_CUT - 2, 0.02, "sheets interact ($w_{\\rm isl}\\to L_y/2$)",
            transform=ax.get_xaxis_transform(), fontsize=8, color="gray",
            rotation=90, ha="right", va="bottom")


def shade(ax):
    if W1:
        ax.axvspan(W1["t0"], W1["t1"], color="gold", alpha=0.20,
                   label=f"primary window [{W1['t0']:.0f},{W1['t1']:.0f}]")
    if W2 and W1 and (W2["t0"], W2["t1"]) != (W1["t0"], W1["t1"]):
        ax.axvspan(W2["t0"], W2["t1"], color="tab:green", alpha=0.12,
                   label=f"sensitivity window [{W2['t0']:.0f},{W2['t1']:.0f}]")
    if t_cross_min:
        ax.axvline(t_cross_min, color="k", ls=":", lw=1.2)


# ---------- fig 1: flux + rate ----------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
ax = axes[0]
for s in ("up", "lo"):
    ax.plot(T, C[f"{s}_psi"], color=COL[s], lw=1.7, label=LABEL[s])
shade(ax)
ax.set_xlabel("$t$"); ax.set_ylabel(r"reconnected flux $\psi$")
ax.set_title(f"{TIER}: reconnected flux, both sheets")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
ax = axes[1]
for s in ("up", "lo"):
    ax.plot(tm, smooth(EA[s]), color=COL[s], lw=1.7, label=f"$E_A$ {s}")
    ax.plot(T, np.abs(C[f"{s}_ezB"]), color=COL[s], lw=0.8, alpha=0.45)
shade(ax)
ax.set_xlabel("$t$"); ax.set_ylabel(r"rate $|d\psi/dt|$")
ax.set_title("Method A (thick) and Method B $|E_z^X|$ (thin)")
ymax = float(np.percentile(np.concatenate([smooth(EA[s]) for s in ("up", "lo")]), 99))
ax.set_ylim(0, 1.6 * max(ymax, 0.05))
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{DIR}/{TIER}_rate.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- fig 1b: flux + rate, validity-cut version ----------
mT = T <= T_CUT + 1e-9
mtm = tm <= T_CUT + 1e-9
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
ax = axes[0]
for s in ("up", "lo"):
    ax.plot(T[mT], C[f"{s}_psi"][mT], color=COL[s], lw=1.7, label=LABEL[s])
shade(ax); tcut_line(ax); ax.set_xlim(*TCUT_XLIM)
ax.set_xlabel("$t$"); ax.set_ylabel(r"reconnected flux $\psi$")
ax.set_title(f"{TIER}: reconnected flux ($t \\leq {T_CUT:.0f}$)")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
ax = axes[1]
for s in ("up", "lo"):
    ax.plot(tm[mtm], smooth(EA[s])[mtm], color=COL[s], lw=1.7, label=f"$E_A$ {s}")
    ax.plot(T[mT], np.abs(C[f"{s}_ezB"])[mT], color=COL[s], lw=0.8, alpha=0.45)
shade(ax); tcut_line(ax); ax.set_xlim(*TCUT_XLIM)
ax.set_xlabel("$t$"); ax.set_ylabel(r"rate $|d\psi/dt|$")
ax.set_title("Method A (thick) and Method B $|E_z^X|$ (thin)")
ycap = float(np.percentile(np.concatenate(
    [smooth(EA[s])[mtm] for s in ("up", "lo")]), 99))
ax.set_ylim(0, 1.6 * max(ycap, 0.05))
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{DIR}/{TIER}_rate_tcut.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- fig 2: island width (FULL run; sustained-crossing marker) ----------
def sustained_crossing(wI, hold=5.0):
    """First armed t with w >= Ly/2 kept for > `hold` time units.
    Armed rule (frozen): detection only active after w < Ly/4 has been seen
    once -- the no-island Az-level scan degenerates to the full box height,
    so early rows would otherwise false-trigger (spike-proof via `hold`)."""
    above = wI >= 0.5 * LY
    armed = False
    for k in range(len(T)):
        if wI[k] < 0.25 * LY:
            armed = True
        if not (armed and above[k]):
            continue
        j = k
        while j + 1 < len(T) and above[j + 1]:
            j += 1
        if T[j] - T[k] > hold:
            return float(T[k])
    return None

t_sus = {s: sustained_crossing(C[f"{s}_wisl"]) for s in ("up", "lo")}
fig, ax = plt.subplots(figsize=(9.5, 4.8))
for s in ("up", "lo"):
    w = C[f"{s}_wisl"].copy(); armed = False
    for k in range(len(w)):
        if w[k] < 0.25 * LY: armed = True
        if not armed: w[k] = np.nan
    ax.plot(T, w, color=COL[s], lw=1.7, label=LABEL[s])
    if t_sus[s] is not None:
        ax.plot(t_sus[s], 0.5 * LY, "*", ms=13, color=COL[s], mec="k", mew=0.5)
tsus_min = min([v for v in t_sus.values() if v is not None], default=None)
if tsus_min is not None:
    ax.annotate("first sustained crossing", xy=(tsus_min, 0.5 * LY),
                xytext=(tsus_min + 12, 0.36 * LY), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
ax.axhline(0.5 * LY, color="k", ls="--", lw=1.2, label=f"$L_y/2 = {0.5*LY:.1f}$")
shade(ax); tcut_line(ax)
ax.set_xlabel("$t$"); ax.set_ylabel(r"island width $w_{\rm isl}$")
ax.set_title(f"{TIER}: island width, full run (stars: sustained $L_y/2$ crossing)")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{DIR}/{TIER}_island.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- L2 keyframes ----------
def read_l2(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, "<f8", 6)
        nxo, nyo, stride, t = int(hdr[1]), int(hdr[2]), int(hdr[3]), hdr[4]
        d = np.fromfile(f, "<f4").reshape(9, nyo, nxo)
    return d, stride, t


def flux_function(Bx, By, dx, dy):   # (nx, ny) arrays
    ny = By.shape[1]
    a0 = np.cumsum(Bx[0, :]) * dy
    a = a0[np.newaxis, :] - np.cumsum(
        np.vstack([np.zeros(ny), 0.5 * (By[:-1, :] + By[1:, :])]), axis=0) * dx
    return a - a.mean()


allframes = sorted(glob.glob(os.path.join(DIR, "l2_*.f32")))
# prefer the physics-tagged frames; pad with evenly spaced others up to 4
pref = [p for p in allframes if any(k in p for k in
        ("plateau0", "islandLy2", "burst", "final"))]
rest = [p for p in allframes if p not in pref]
picks = (pref + rest[:: max(1, len(rest) // max(1, 4 - len(pref)))])[:4]
picks = sorted(set(picks))
frames = [read_l2(p) for p in picks]

if frames:
    for qty, cmap, fname in ((None, "RdBu_r", "keyframes_jz"),
                             (0, "viridis", "keyframes_rho")):
        n = len(frames)
        nrow = 2 if n > 2 else 1
        ncol = int(np.ceil(n / nrow))
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.3 * ncol, 4.1 * nrow),
                                 squeeze=False)
        if qty == 0:
            vmin = min(float(d[0].min()) for d, _, _ in frames)
            vmax = max(float(d[0].max()) for d, _, _ in frames)
        for ax, (d, stride, t) in zip(axes.ravel(), frames):
            h = (LX / d.shape[2])
            Bx, By = d[5].T, d[6].T
            az = flux_function(Bx, By, h, h)
            ext = [-LX/2, LX/2, -LY/2, LY/2]
            if qty is None:
                Jz = np.gradient(By, h, axis=0) - np.gradient(Bx, h, axis=1)
                vl = np.percentile(np.abs(Jz), 99.5)
                im = ax.imshow(Jz.T, origin="lower", extent=ext, cmap=cmap,
                               vmin=-vl, vmax=vl, aspect="equal")
                cc = "k"
                ax.set_title(f"$J_z$, $t={t:.1f}$"
                             + (" (stride 2)" if stride == 2 else ""))
            else:
                im = ax.imshow(d[0], origin="lower", extent=ext, cmap=cmap,
                               vmin=vmin, vmax=vmax, aspect="equal")
                cc = "w"
                ax.set_title(f"$\\rho$, $t={t:.1f}$"
                             + (" (stride 2)" if stride == 2 else ""))
            ax.contour(np.linspace(ext[0], ext[1], az.shape[0]),
                       np.linspace(ext[2], ext[3], az.shape[1]),
                       az.T, levels=18, colors=cc, linewidths=0.45, alpha=0.55)
            ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
            cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.08)
            fig.colorbar(im, cax=cax)
        for ax in axes.ravel()[len(frames):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{DIR}/{TIER}_{fname}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

# ---------- fig 5: outflow / x_X / y_X drift / co-moving inflow ----------
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
ax = axes[0, 0]
for s in ("up", "lo"):
    ax.plot(T, C[f"{s}_vout"], color=COL[s], lw=1.5, label=LABEL[s])
shade(ax)
ax.set_xlabel("$t$"); ax.set_ylabel(r"sheet-band max $|v_x|$")
ax.set_title("Outflow speed"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)

ax = axes[0, 1]
for s in ("up", "lo"):
    ax.plot(T, C[f"{s}_xX"], color=COL[s], lw=1.2, label=f"$x_X$ {s}")
shade(ax)
ax.set_xlabel("$t$"); ax.set_ylabel("$x_X$")
ax.set_title("X-line position along the sheet")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)

vy_sheet = {}
ax = axes[1, 0]
if "up_yX" in C:
    mfit = (T > 10) & ((T < t_cross_min) if t_cross_min else np.ones(len(T), bool))
    if mfit.sum() < 5:                     # short/smoke runs: fit whatever exists
        mfit = T > 0.2 * T[-1]
    for s, y0 in (("up", 0.25 * LY), ("lo", -0.25 * LY)):
        ax.plot(T, C[f"{s}_yX"] - y0, color=COL[s], lw=1.5,
                label=f"{LABEL[s]} $y_X - y_0$")
        pf = np.polyfit(T[mfit], C[f"{s}_yX"][mfit], 1)
        vy_sheet[s] = float(pf[0])
        ax.plot(T[mfit], np.polyval(pf, T[mfit]) - y0, color=COL[s], ls="--",
                lw=1.0, label=f"fit $v_{{X,y}}={pf[0]:+.4f}$")
    ttl = "Sheet drift along inflow"
    if abs(B01 - B02) > 1e-12:
        strong = "B02 (inner)" if B02 > B01 else "B01 (outer)"
        ttl += f" (CS2008: expect toward {strong})"
    ax.set_title(ttl)
    ax.axhline(0, color="gray", lw=0.8)
    shade(ax)
    ax.set_xlabel("$t$"); ax.set_ylabel("$y_X - y_{X,0}$")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "no yX columns (schema v1 tier)", ha="center")

ax = axes[1, 1]
if "up_vinp" in C:
    for s in ("up", "lo"):
        vs = vy_sheet.get(s, 0.0)
        ax.plot(T, smooth(C[f"{s}_vinp"] - vs, 9), color=COL[s], lw=1.3,
                label=f"{s} vin(+3), co-moving")
        ax.plot(T, smooth(C[f"{s}_vinm"] - vs, 9), color=COL[s], lw=1.3,
                ls="--", label=f"{s} vin(-3), co-moving")
    shade(ax)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"signed $v_y$ at $x_X$, $\pm 3$ from sheet $- v_{X,y}$")
    ax.set_title("Inflow (co-moving frame), smoothed")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "no vin columns (schema v1 tier)", ha="center")
fig.tight_layout()
fig.savefig(f"{DIR}/{TIER}_drift_inflow.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- fig 5b: drift/inflow, validity-cut version ----------
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
ax = axes[0, 0]
for s in ("up", "lo"):
    ax.plot(T[mT], C[f"{s}_vout"][mT], color=COL[s], lw=1.5, label=LABEL[s])
shade(ax); tcut_line(ax); ax.set_xlim(*TCUT_XLIM)
ax.set_xlabel("$t$"); ax.set_ylabel(r"sheet-band max $|v_x|$")
ax.set_title(f"Outflow speed ($t \\leq {T_CUT:.0f}$)")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)

ax = axes[0, 1]
for s in ("up", "lo"):
    ax.plot(T[mT], C[f"{s}_xX"][mT], color=COL[s], lw=1.2, label=f"$x_X$ {s}")
shade(ax); tcut_line(ax); ax.set_xlim(*TCUT_XLIM)
ax.set_xlabel("$t$"); ax.set_ylabel("$x_X$")
ax.set_title("X-line position along the sheet")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)

vy_cut = {}
ax = axes[1, 0]
if "up_yX" in C:
    mfit2 = (T > 10) & (T <= T_CUT)
    if mfit2.sum() < 5:
        mfit2 = mT
    for s, y0 in (("up", 0.25 * LY), ("lo", -0.25 * LY)):
        ax.plot(T[mT], C[f"{s}_yX"][mT] - y0, color=COL[s], lw=1.5,
                label=f"{LABEL[s]} $y_X - y_0$")
        pf = np.polyfit(T[mfit2], C[f"{s}_yX"][mfit2], 1)
        vy_cut[s] = float(pf[0])
        ax.plot(T[mfit2], np.polyval(pf, T[mfit2]) - y0, color=COL[s], ls="--",
                lw=1.0, label=f"fit $v_{{X,y}}={pf[0]:+.4f}$")
    ttl = "Sheet drift along inflow"
    if abs(B01 - B02) > 1e-12:
        ttl += " (CS2008: expect toward %s)" % \
               ("B02 (inner)" if B02 > B01 else "B01 (outer)")
    ax.set_title(ttl); ax.axhline(0, color="gray", lw=0.8)
    shade(ax); tcut_line(ax); ax.set_xlim(*TCUT_XLIM)
    ax.set_xlabel("$t$"); ax.set_ylabel("$y_X - y_{X,0}$")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "no yX columns (schema v1 tier)", ha="center")

ax = axes[1, 1]
if "up_vinp" in C:
    for s in ("up", "lo"):
        vs = vy_cut.get(s, 0.0)
        ax.plot(T[mT], smooth(C[f"{s}_vinp"] - vs, 9)[mT], color=COL[s], lw=1.3,
                label=f"{s} vin(+3), co-moving")
        ax.plot(T[mT], smooth(C[f"{s}_vinm"] - vs, 9)[mT], color=COL[s], lw=1.3,
                ls="--", label=f"{s} vin(-3), co-moving")
    shade(ax); tcut_line(ax); ax.set_xlim(*TCUT_XLIM)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"signed $v_y$ at $x_X$, $\pm 3$ from sheet $- v_{X,y}$")
    ax.set_title("Inflow (co-moving frame), smoothed")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "no vin columns (schema v1 tier)", ha="center")
fig.tight_layout()
fig.savefig(f"{DIR}/{TIER}_drift_inflow_tcut.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- statistics with the validity cut (stdout + file) ----------
stats = [f"tier {TIER}: statistics with analysis validity cut T_CUT = {T_CUT}",
         f"(window upper bound capped at min(island crossing, T_CUT) = "
         f"{t_hi_eff:.1f}; onset criterion for the window start unchanged)", ""]
if W1:
    mwin = (tm >= W1["t0"]) & (tm <= W1["t1"])
    seg = EAc[mwin]
    stats += [f"plateau window (primary, first qualifying): "
              f"[{W1['t0']:.2f}, {W1['t1']:.2f}]  n={int(mwin.sum())}",
              f"  combined E: mean={seg.mean():.4f} +/- std={seg.std(ddof=1):.4f}"
              f"   median={np.median(seg):.4f} +/- "
              f"hIQR={0.5*(np.percentile(seg,75)-np.percentile(seg,25)):.4f}"]
    for s in ("up", "lo"):
        es = EA[s][mwin]
        stats.append(f"  [{s}] E_A mean={es.mean():.4f}+/-{es.std(ddof=1):.4f} "
                     f"median={np.median(es):.4f}")
    if W2 and (W2["t0"], W2["t1"]) != (W1["t0"], W1["t1"]):
        mw2 = (tm >= W2["t0"]) & (tm <= W2["t1"])
        s2 = EAc[mw2]
        stats.append(f"sensitivity window (last before cap): "
                     f"[{W2['t0']:.2f}, {W2['t1']:.2f}]  n={int(mw2.sum())}  "
                     f"mean={s2.mean():.4f}+/-{s2.std(ddof=1):.4f} "
                     f"median={np.median(s2):.4f}")
    stats.append("")
    stats.append("quasi-steady outflow (plateau window, t <= T_CUT):")
    mv = (T >= W1["t0"]) & (T <= min(W1["t1"], T_CUT))
    for s in ("up", "lo"):
        vo = C[f"{s}_vout"][mv]
        stats.append(f"  [{s}] median={np.median(vo):.3f}  "
                     f"mean={vo.mean():.3f}+/-{vo.std(ddof=1):.3f}  n={int(mv.sum())}")
else:
    stats.append("NO qualifying plateau window inside t <= cap "
                 "(report as-is, do not lower the onset guard silently)")
stats.append("")
for s in ("up", "lo"):
    stats.append(f"island sustained Ly/2 crossing [{s}]: "
                 f"t={t_sus[s] if t_sus[s] is not None else 'none'}")
if vy_cut:
    stats.append("sheet drift v_X,y (fit 10<t<=T_CUT): "
                 + ", ".join(f"{s}={v:+.5f}" for s, v in vy_cut.items()))
with open(f"{DIR}/stats_summary_tcut.txt", "w") as f:
    f.write("\n".join(stats) + "\n")
print("\n".join(stats))

msg = [f"wrote figures to {DIR}/ (incl. _tcut set + stats_summary_tcut.txt)"]
if t_cross_min: msg.append(f"Ly/2 armed crossing t={t_cross_min:.1f}")
print("; ".join(msg))
