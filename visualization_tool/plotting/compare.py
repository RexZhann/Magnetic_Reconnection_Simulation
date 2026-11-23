"""Overlay reconnection diagnostics of several runs (former output/compare_21_23.py)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..physics import time_series
from ._common import save


def compare_runs(runs: dict, save_path=None, title=None, t_max=None):
    """``runs`` maps a legend label -> list of snapshots. 2x2 panels:
    reconnected flux, inflow speed, peak Jz, max |Bz|."""
    panels = [
        ("flux", r"$\Phi=\int_0^{L_x/2}|B_y|\,dx$ ($y=0$)", "reconnected flux"),
        ("max_vy", r"$\max|v_y|$", "inflow speed (rate proxy)"),
        ("max_jz", r"$\max J_z$", "peak current density"),
        ("max_Bz", r"$\max|B_z|$", r"Hall quadrupole $|B_z|$"),
    ]
    series = {lbl: time_series(snaps) for lbl, snaps in runs.items()}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title or " vs ".join(runs), fontsize=12, fontweight="bold")
    for ax, (key, ylabel, ttl) in zip(axes.flat, panels):
        for k, (lbl, ts) in enumerate(series.items()):
            ax.plot(ts["t"], ts[key], f"C{k}-", lw=2.0, label=lbl)
        ax.set_xlabel("t"); ax.set_ylabel(ylabel); ax.set_title(ttl)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        if t_max is not None:
            ax.set_xlim(0, t_max)
    fig.tight_layout()
    return save(fig, save_path)
