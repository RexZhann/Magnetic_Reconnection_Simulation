"""Figures for specific results, with hand-placed annotations.

Generic plotting lives in visualization_tool.plotting. These recipes reproduce
particular figures and contain numbers read off specific runs, so they are
only meaningful for those runs.
"""

from __future__ import annotations

import os

from .plotting.campaign import plot_rate_comparison


def stage0_rate_comparison(campaign_dir="output/test29_campaign", save_path=None):
    """Sym vs AB1 vs AB2: does the asymmetric rate settle on a plateau before
    the islands reach Ly/2? (former test29_campaign/plot_rate_comparison.py)"""
    save_path = save_path or os.path.join(campaign_dir, "rate_comparison_Sym_AB1_AB2.png")
    tiers = [(t, os.path.join(campaign_dir, t)) for t in ("Sym", "AB1", "AB2")]

    def annotate(ax1, ax2, info):
        # Sym plateau band (Stage 0 verdict)
        ax1.axhspan(0.0373 - 0.0021, 0.0373 + 0.0021, xmin=218.3 / 400, xmax=263.3 / 400,
                    color=info["Sym"]["color"], alpha=0.18, lw=0)
        ax1.annotate("Sym plateau\nE=0.0373", xy=(240, 0.043), color=info["Sym"]["color"],
                     fontsize=9, ha="center")
        for tier in ("Sym", "AB1"):
            d = info[tier]
            ax1.annotate(f"{tier} peak {d['peak']:.3f}\n@ t={d['tpk']:.0f}",
                         xy=(d["tpk"], d["peak"]), xytext=(d["tpk"] - 52, d["peak"] * 1.7),
                         color=d["color"], fontsize=9,
                         arrowprops=dict(arrowstyle="-", color=d["color"], lw=0.8))
        c2 = info["AB2"]["color"]
        ax1.annotate("AB2 spikes $\\gtrsim 1$: $\\psi$ O/X identity jumps\n"
                     "during merger phase (not physical rates);\n"
                     "A/B cross-check fails 117-166% there",
                     xy=(6, 0.55), fontsize=8, color=c2, ha="left", va="top")
        ax1.set_ylim(3e-3, 2.0)
        # AB2 degenerate stretch t=76-105 (w_isl scan failed -> full-box value)
        ax2.axvspan(76, 105, color=c2, alpha=0.12, lw=0)
        ax2.annotate("AB2 degenerate\nwisl frames\n(artifact t=79)", xy=(90, 44),
                     fontsize=8, color=c2, ha="center")

    # AB2's "peak" is psi-diagnostic contamination, not a rate: no marker
    return plot_rate_comparison(
        tiers, save_path, Ly=51.2, mark_peaks=("Sym", "AB1"), extra=annotate, xlim=(0, 400),
        title="Sym / AB1 / AB2: rate never settles before sheets interact "
              "(dashed = sustained $L_y/2$ crossing)")
