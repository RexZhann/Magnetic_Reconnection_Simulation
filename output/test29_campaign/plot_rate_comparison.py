"""Sym vs AB1 vs AB2: reconnection-rate history and island width.

Question answered by the figure: do the asymmetric tiers ever settle on a
plateau before the islands reach Ly/2 (mutual-inductance cutoff)?

Output: output/test29_campaign/rate_comparison_Sym_AB1_AB2.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Okabe-Ito (CVD-safe, validated); fixed tier->color assignment
TIERS = [("Sym", "#0072B2"), ("AB1", "#E69F00"), ("AB2", "#009E73")]
LY = 51.2


def load(tier):
    rows, cols = [], None
    for line in open(os.path.join(HERE, tier, "l1.csv")):
        if line.startswith("#"):
            continue
        if line.startswith("t,"):
            cols = line.strip().split(",")
            continue
        if line.strip():
            rows.append([float(v) for v in line.split(",")])
    A = np.array(rows)
    return {n: A[:, k] for k, n in enumerate(cols)}


def sustained_crossing(T, wI, hold=5.0):
    # armed + sustained + DEGENERATE-MASKED (wisl==Ly frames are failed
    # measurements, not crossings; ruling approved 2026-07-18 — neutral for
    # Sym/AB1, moves AB2 from the artifact t=79 to t=184/185).
    degen = wI >= LY - 1e-6
    valid = ~degen
    armed = False
    for k in range(len(wI)):
        if valid[k] and wI[k] < 0.25 * LY:
            armed = True
        if not armed or not valid[k] or wI[k] < 0.5 * LY:
            continue
        m = (T >= T[k]) & (T <= T[k] + hold) & valid
        if m.sum() >= 3 and np.all(wI[m] >= 0.5 * LY):
            return float(T[k])
    return None


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True,
                               gridspec_kw=dict(hspace=0.08))
info = {}
for tier, col in TIERS:
    C = load(tier)
    T = C["t"]
    tm = 0.5 * (T[:-1] + T[1:])
    EA = 0.5 * (np.abs(np.diff(C["up_psi"])) + np.abs(np.diff(C["lo_psi"]))) \
        / np.diff(T)
    EAc = np.convolve(EA, np.ones(5) / 5, mode="same")
    wmax = np.maximum(C["up_wisl"], C["lo_wisl"])
    tc = min(x for x in (sustained_crossing(T, C["up_wisl"]),
                         sustained_crossing(T, C["lo_wisl"])) if x is not None)
    mpre = (tm > 10) & (tm < tc)
    info[tier] = dict(peak=EAc[mpre].max(), tpk=tm[mpre][EAc[mpre].argmax()], tc=tc)

    ax1.plot(tm, EAc, color=col, lw=2, label=tier)
    ax1.axvline(tc, color=col, ls="--", lw=1.2, alpha=0.7)
    if tier != "AB2":   # AB2 "peak" is psi-diagnostic contamination, not a rate
        ax1.plot(info[tier]["tpk"], info[tier]["peak"], "o", ms=8, color=col,
                 mec="white", mew=1.2, zorder=5)
    ax2.plot(T, wmax, color=col, lw=2)
    ax2.axvline(tc, color=col, ls="--", lw=1.2, alpha=0.7)

# Sym plateau band (Stage 0 verdict) for reference
ax1.axhspan(0.0373 - 0.0021, 0.0373 + 0.0021, xmin=218.3 / 400, xmax=263.3 / 400,
            color="#0072B2", alpha=0.18, lw=0)
ax1.annotate("Sym plateau\nE=0.0373", xy=(240, 0.043), color="#0072B2",
             fontsize=9, ha="center")
for tier, col in TIERS:
    if tier == "AB2":
        continue
    d = info[tier]
    ax1.annotate(f"{tier} peak {d['peak']:.3f}\n@ t={d['tpk']:.0f}",
                 xy=(d["tpk"], d["peak"]), xytext=(d["tpk"] - 52, d["peak"] * 1.7),
                 color=col, fontsize=9,
                 arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
ax1.annotate("AB2 spikes $\\gtrsim 1$: $\\psi$ O/X identity jumps\n"
             "during merger phase (not physical rates);\n"
             "A/B cross-check fails 117-166% there",
             xy=(6, 0.55), fontsize=8, color="#009E73", ha="left", va="top")

ax1.set_yscale("log")
ax1.set_ylim(3e-3, 2.0)
ax1.set_ylabel(r"reconnection rate  $E_A = \langle|\dot\psi|\rangle_{\rm sheets}$"
               "  (5-pt smoothed)")
ax1.set_title("Sym / AB1 / AB2: rate never settles before sheets interact "
              "(dashed = sustained $L_y/2$ crossing)")
ax1.legend(loc="lower right", frameon=False)
ax1.grid(alpha=0.25, lw=0.5)

# AB2 degenerate stretch t=76-105 (wisl scan failed -> full-box value):
# this fooled the un-masked sustained rule into the artifact crossing t=79
ax2.axvspan(76, 105, color="#009E73", alpha=0.12, lw=0)
ax2.annotate("AB2 degenerate\nwisl frames\n(artifact t=79)", xy=(90, 44),
             fontsize=8, color="#009E73", ha="center")

ax2.axhline(0.5 * LY, color="k", ls=":", lw=1.2)
ax2.text(392, 0.5 * LY + 0.8, r"$L_y/2$", ha="right", fontsize=9)
ax2.set_ylabel(r"island width  $\max(w_{\rm isl}^{\rm up}, w_{\rm isl}^{\rm lo})$")
ax2.set_xlabel(r"$t\;[d_i/c_{A0}]$")
ax2.set_xlim(0, 400)
ax2.set_ylim(0, LY * 1.04)
ax2.grid(alpha=0.25, lw=0.5)

out = os.path.join(HERE, "rate_comparison_Sym_AB1_AB2.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("peaks:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in info.items()})
print("->", out)
