"""Test-31 campaign figures (former deploy_server/plot_tier.py and
output/test29_campaign/plot_rate_comparison.py).

Window / crossing logic comes from visualization_tool.campaign, i.e. exactly
the pre-registered rules used for the tier summary.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .. import campaign as cp
from ..io import l2_files, l2_time, load_l1, read_l2
from ..physics import flux_function
from ._common import save

COL = {"up": "C0", "lo": "C3"}
LABEL = {"up": "upper sheet", "lo": "lower sheet"}
OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
             "#56B4E9", "#F0E442", "#000000")
_RC = {"font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12,
       "legend.fontsize": 10}


def _pick_frames(run_dir, frame_times=None, nmax=4):
    files = l2_files(run_dir)
    if frame_times:
        picks = [min(files, key=lambda p: abs(l2_time(p) - float(t))) for t in frame_times]
    else:
        # physics-tagged frames first, padded with evenly spaced others
        pref = [p for p in files if any(k in p for k in
                ("plateau0", "islandLy2", "burst", "final"))]
        rest = [p for p in files if p not in pref]
        picks = (pref + rest[:: max(1, len(rest) // max(1, nmax - len(pref)))])[:nmax]
    return sorted(set(picks))


def plot_l2_keyframes(run_dir, frame_times=None, out_prefix=None, Lx=None, Ly=None):
    """Jz and rho keyframes (with Az contours) from up to 4 L2 frames."""
    l1 = load_l1(run_dir)
    LX, LY = Lx or l1.Lx, Ly or l1.Ly
    frames = [read_l2(p) for p in _pick_frames(run_dir, frame_times)]
    if not frames:
        print(f"  no L2 frames in {run_dir}")
        return []
    out = []
    for qty, cmap, fname in ((None, "RdBu_r", "keyframes_jz"), (0, "viridis", "keyframes_rho")):
        n = len(frames)
        nrow = 2 if n > 2 else 1
        ncol = int(np.ceil(n / nrow))
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.3 * ncol, 4.1 * nrow), squeeze=False)
        if qty == 0:
            vmin = min(float(d[0].min()) for d, _, _ in frames)
            vmax = max(float(d[0].max()) for d, _, _ in frames)
        for ax, (d, stride, t) in zip(axes.ravel(), frames):
            h = LX / d.shape[2]
            Bx, By = d[5].T, d[6].T
            az = flux_function(Bx, By, h, h)
            ext = [-LX / 2, LX / 2, -LY / 2, LY / 2]
            sfx = f" (stride {stride})" if stride > 1 else ""
            if qty is None:
                Jz = np.gradient(By, h, axis=0) - np.gradient(Bx, h, axis=1)
                vl = np.percentile(np.abs(Jz), 99.5)
                im = ax.imshow(Jz.T, origin="lower", extent=ext, cmap=cmap,
                               vmin=-vl, vmax=vl, aspect="equal")
                cc = "k"
                ax.set_title(f"$J_z$, $t={t:.1f}$" + sfx)
            else:
                im = ax.imshow(d[0], origin="lower", extent=ext, cmap=cmap,
                               vmin=vmin, vmax=vmax, aspect="equal")
                cc = "w"
                ax.set_title(f"$\\rho$, $t={t:.1f}$" + sfx)
            ax.contour(np.linspace(ext[0], ext[1], az.shape[0]),
                       np.linspace(ext[2], ext[3], az.shape[1]),
                       az.T, levels=18, colors=cc, linewidths=0.45, alpha=0.55)
            ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
            cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.08)
            fig.colorbar(im, cax=cax)
        for ax in axes.ravel()[len(frames):]:
            ax.set_visible(False)
        fig.tight_layout()
        out.append(save(fig, f"{out_prefix}_{fname}.png", dpi=140))
    return out


def plot_tier(run_dir, tier=None, t_cut=170.0, frame_times=None):
    """All quick-look figures + ``stats_summary_tcut.txt`` for one tier.

    ``t_cut``: analysis validity cut (sheets interact once w_isl -> Ly/2);
    *_tcut figures and all statistics use t <= t_cut, the island-width figure
    and the 2D keyframes keep the full run. ``frame_times``: L2 frames
    nearest these times (default: physics-tagged frames).
    """
    tier = tier or os.path.basename(os.path.normpath(run_dir))
    prefix = os.path.join(run_dir, tier)
    with plt.rc_context(_RC):
        return _plot_tier(run_dir, tier, prefix, t_cut, frame_times)


def _plot_tier(run_dir, tier, prefix, T_CUT, frame_times):
    l1 = load_l1(run_dir)
    C, T, LY = l1.C, l1.T, l1.Ly
    B01, B02 = l1.meta.get("B01", 1.0), l1.meta.get("B02", 1.0)
    tm, EA, _ = cp.method_a_rates(l1)
    cross, t_cross_min = cp.crossings(l1)
    t_sus = {s: cross[s][0] for s in cp.SHEETS}
    W1, W2, _ = cp.plateau_windows(l1, t_cut=T_CUT)
    t_hi_eff = min([v for v in (t_cross_min, T_CUT) if v is not None])
    TCUT_XLIM = (0.0, T_CUT + 10.0)
    mT, mtm = T <= T_CUT + 1e-9, tm <= T_CUT + 1e-9
    smooth = cp.smooth

    def shade(ax):
        if W1:
            ax.axvspan(W1["t0"], W1["t1"], color="gold", alpha=0.20,
                       label=f"primary window [{W1['t0']:.0f},{W1['t1']:.0f}]")
        if W2 and W1 and (W2["t0"], W2["t1"]) != (W1["t0"], W1["t1"]):
            ax.axvspan(W2["t0"], W2["t1"], color="tab:green", alpha=0.12,
                       label=f"sensitivity window [{W2['t0']:.0f},{W2['t1']:.0f}]")
        if t_cross_min:
            ax.axvline(t_cross_min, color="k", ls=":", lw=1.2)

    def tcut_line(ax):
        ax.axvline(T_CUT, color="gray", ls="--", lw=1.1)
        ax.text(T_CUT - 2, 0.02, "sheets interact ($w_{\\rm isl}\\to L_y/2$)",
                transform=ax.get_xaxis_transform(), fontsize=8, color="gray",
                rotation=90, ha="right", va="bottom")

    out = []

    # --- flux + rate (full run, and t <= T_CUT) ---
    for cut in (False, True):
        mt, mm = (mT, mtm) if cut else (slice(None), slice(None))
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
        ax = axes[0]
        for s in cp.SHEETS:
            ax.plot(T[mt], C[f"{s}_psi"][mt], color=COL[s], lw=1.7, label=LABEL[s])
        shade(ax)
        if cut:
            tcut_line(ax); ax.set_xlim(*TCUT_XLIM)
        ax.set_xlabel("$t$"); ax.set_ylabel(r"reconnected flux $\psi$")
        ax.set_title(f"{tier}: reconnected flux" +
                     (f" ($t \\leq {T_CUT:.0f}$)" if cut else ", both sheets"))
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        ax = axes[1]
        for s in cp.SHEETS:
            ax.plot(tm[mm], smooth(EA[s])[mm], color=COL[s], lw=1.7, label=f"$E_A$ {s}")
            ax.plot(T[mt], np.abs(C[f"{s}_ezB"])[mt], color=COL[s], lw=0.8, alpha=0.45)
        shade(ax)
        if cut:
            tcut_line(ax); ax.set_xlim(*TCUT_XLIM)
        ax.set_xlabel("$t$"); ax.set_ylabel(r"rate $|d\psi/dt|$")
        ax.set_title("Method A (thick) and Method B $|E_z^X|$ (thin)")
        ymax = float(np.percentile(np.concatenate(
            [smooth(EA[s])[mm] for s in cp.SHEETS]), 99))
        ax.set_ylim(0, 1.6 * max(ymax, 0.05))
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        fig.tight_layout()
        out.append(save(fig, f"{prefix}_rate{'_tcut' if cut else ''}.png"))

    # --- island width, full run ---
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for s in cp.SHEETS:
        w = C[f"{s}_wisl"].copy()
        armed = False
        for k in range(len(w)):
            if w[k] < 0.25 * LY:
                armed = True
            if not armed:
                w[k] = np.nan
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
    ax.set_title(f"{tier}: island width, full run (stars: sustained $L_y/2$ crossing)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out.append(save(fig, f"{prefix}_island.png"))

    # --- L2 keyframes ---
    out += plot_l2_keyframes(run_dir, frame_times, prefix)

    # --- outflow / x_X / y_X drift / co-moving inflow (full run, and t <= T_CUT) ---
    vy_fit = {}
    for cut in (False, True):
        mt = mT if cut else np.ones(len(T), bool)
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

        def finish(ax):
            shade(ax)
            if cut:
                tcut_line(ax); ax.set_xlim(*TCUT_XLIM)

        ax = axes[0, 0]
        for s in cp.SHEETS:
            ax.plot(T[mt], C[f"{s}_vout"][mt], color=COL[s], lw=1.5, label=LABEL[s])
        finish(ax)
        ax.set_xlabel("$t$"); ax.set_ylabel(r"sheet-band max $|v_x|$")
        ax.set_title("Outflow speed" + (f" ($t \\leq {T_CUT:.0f}$)" if cut else ""))
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        for s in cp.SHEETS:
            ax.plot(T[mt], C[f"{s}_xX"][mt], color=COL[s], lw=1.2, label=f"$x_X$ {s}")
        finish(ax)
        ax.set_xlabel("$t$"); ax.set_ylabel("$x_X$")
        ax.set_title("X-line position along the sheet")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)

        vy = {}
        ax = axes[1, 0]
        if "up_yX" in C:
            if cut:
                mfit = (T > 10) & (T <= T_CUT)
                if mfit.sum() < 5:
                    mfit = mT
            else:
                mfit = (T > 10) & ((T < t_cross_min) if t_cross_min
                                   else np.ones(len(T), bool))
                if mfit.sum() < 5:          # short/smoke runs: fit what exists
                    mfit = T > 0.2 * T[-1]
            for s, y0 in (("up", 0.25 * LY), ("lo", -0.25 * LY)):
                ax.plot(T[mt], C[f"{s}_yX"][mt] - y0, color=COL[s], lw=1.5,
                        label=f"{LABEL[s]} $y_X - y_0$")
                pf = np.polyfit(T[mfit], C[f"{s}_yX"][mfit], 1)
                vy[s] = float(pf[0])
                ax.plot(T[mfit], np.polyval(pf, T[mfit]) - y0, color=COL[s], ls="--",
                        lw=1.0, label=f"fit $v_{{X,y}}={pf[0]:+.4f}$")
            ttl = "Sheet drift along inflow"
            if abs(B01 - B02) > 1e-12:
                ttl += " (CS2008: expect toward %s)" % (
                    "B02 (inner)" if B02 > B01 else "B01 (outer)")
            ax.set_title(ttl); ax.axhline(0, color="gray", lw=0.8)
            finish(ax)
            ax.set_xlabel("$t$"); ax.set_ylabel("$y_X - y_{X,0}$")
            ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "no yX columns (schema v1 tier)", ha="center")

        ax = axes[1, 1]
        if "up_vinp" in C:
            for s in cp.SHEETS:
                vs = vy.get(s, 0.0)
                ax.plot(T[mt], smooth(C[f"{s}_vinp"] - vs, 9)[mt], color=COL[s], lw=1.3,
                        label=f"{s} vin(+3), co-moving")
                ax.plot(T[mt], smooth(C[f"{s}_vinm"] - vs, 9)[mt], color=COL[s], lw=1.3,
                        ls="--", label=f"{s} vin(-3), co-moving")
            finish(ax)
            ax.set_xlabel("$t$")
            ax.set_ylabel(r"signed $v_y$ at $x_X$, $\pm 3$ from sheet $- v_{X,y}$")
            ax.set_title("Inflow (co-moving frame), smoothed")
            ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "no vin columns (schema v1 tier)", ha="center")
        fig.tight_layout()
        out.append(save(fig, f"{prefix}_drift_inflow{'_tcut' if cut else ''}.png"))
        if cut:
            vy_fit = vy

    # --- statistics with the validity cut ---
    EAc = cp.method_a_rates(l1)[2]
    stats = [f"tier {tier}: statistics with analysis validity cut T_CUT = {T_CUT}",
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
        for s in cp.SHEETS:
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
        stats += ["", "quasi-steady outflow (plateau window, t <= T_CUT):"]
        mv = (T >= W1["t0"]) & (T <= min(W1["t1"], T_CUT))
        for s in cp.SHEETS:
            vo = C[f"{s}_vout"][mv]
            stats.append(f"  [{s}] median={np.median(vo):.3f}  "
                         f"mean={vo.mean():.3f}+/-{vo.std(ddof=1):.3f}  n={int(mv.sum())}")
    else:
        stats.append("NO qualifying plateau window inside t <= cap "
                     "(report as-is, do not lower the onset guard silently)")
    stats.append("")
    for s in cp.SHEETS:
        stats.append(f"island sustained Ly/2 crossing [{s}]: "
                     f"t={t_sus[s] if t_sus[s] is not None else 'none'}")
    if vy_fit:
        stats.append("sheet drift v_X,y (fit 10<t<=T_CUT): "
                     + ", ".join(f"{s}={v:+.5f}" for s, v in vy_fit.items()))
    text = "\n".join(stats) + "\n"
    with open(os.path.join(run_dir, "stats_summary_tcut.txt"), "w") as f:
        f.write(text)
    print(text, end="")
    return out


def plot_rate_comparison(tiers, save_path=None, Ly=None, colors=OKABE_ITO,
                         mark_peaks=True, extra=None, xlim=None, title=None):
    """Smoothed Method-A rate (top, log) and island width (bottom) of several tiers.

    ``tiers``: list of ``(label, run_dir)``. Dashed verticals = sustained Ly/2
    crossing. ``mark_peaks``: True/False, or the labels whose pre-crossing
    peak gets a marker. ``extra(ax_rate, ax_island, info)`` may add annotations;
    ``info[label] = dict(peak, tpk, tc, color)``.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True,
                                   gridspec_kw=dict(hspace=0.08))
    info = {}
    LY = Ly
    for (label, run_dir), col in zip(tiers, colors):
        l1 = load_l1(run_dir)
        LY = Ly or l1.Ly
        T = l1.T
        tm, _, EAc = cp.method_a_rates(l1)
        _, tc = cp.crossings(l1)
        wmax = np.maximum(l1["up_wisl"], l1["lo_wisl"])
        mpre = (tm > 10) & ((tm < tc) if tc is not None else True)
        peak = tpk = None
        if mpre.any():
            peak, tpk = EAc[mpre].max(), tm[mpre][EAc[mpre].argmax()]
        info[label] = dict(peak=peak, tpk=tpk, tc=tc, color=col)
        ax1.plot(tm, EAc, color=col, lw=2, label=label)
        ax2.plot(T, wmax, color=col, lw=2)
        if tc is not None:
            ax1.axvline(tc, color=col, ls="--", lw=1.2, alpha=0.7)
            ax2.axvline(tc, color=col, ls="--", lw=1.2, alpha=0.7)
        mark = mark_peaks if isinstance(mark_peaks, bool) else label in mark_peaks
        if mark and peak is not None:
            ax1.plot(tpk, peak, "o", ms=8, color=col, mec="white", mew=1.2, zorder=5)

    ax1.set_yscale("log")
    ax1.set_ylabel(r"reconnection rate $E_A = \langle|\dot\psi|\rangle_{\rm sheets}$"
                   " (5-pt smoothed)")
    ax1.set_title(title or "reconnection rate and island width "
                  "(dashed = sustained $L_y/2$ crossing)")
    ax1.legend(loc="lower right", frameon=False)
    ax1.grid(alpha=0.25, lw=0.5)
    ax2.axhline(0.5 * LY, color="k", ls=":", lw=1.2)
    ax2.text(0.99, 0.5 * LY + 0.8, r"$L_y/2$", ha="right", fontsize=9,
             transform=ax2.get_yaxis_transform())
    ax2.set_ylabel(r"island width $\max(w_{\rm isl}^{\rm up}, w_{\rm isl}^{\rm lo})$")
    ax2.set_xlabel(r"$t\;[d_i/c_{A0}]$")
    ax2.set_ylim(0, LY * 1.04)
    ax2.grid(alpha=0.25, lw=0.5)
    if xlim:
        ax2.set_xlim(*xlim)
    if extra:
        extra(ax1, ax2, info)
    print("peaks:", {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                         for kk, vv in v.items() if kk != "color"} for k, v in info.items()})
    return save(fig, save_path, dpi=160)
