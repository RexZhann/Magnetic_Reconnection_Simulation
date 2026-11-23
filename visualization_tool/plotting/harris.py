"""Snapshot figures for single-sheet reconnection runs (tests 11-12, 19-23, 25-28, 30).

Merges the former output/test11/harris_viz.py and output/test23/harris_viz.py:
Hall runs (non-zero Bz) automatically get the Bz quadrupole row/panel.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from ..io import nearest
from ..physics import compute_jz, has_hall, time_series
from ._common import save


def _regime(snaps) -> str:
    return "Hall MHD" if has_hall(snaps) else "resistive MHD"


def plot_keyframes(snaps, times=(2.0, 8.0, 20.0), save_path=None, title=None):
    """Columns = key times (nearest snapshots). Rows: Jz + field lines,
    pressure, and (Hall runs) out-of-plane Bz."""
    idxs = [nearest(snaps, t) for t in times]
    hall = has_hall(snaps)
    Jz_lim = np.percentile(np.abs(np.concatenate(
        [compute_jz(snaps[i]).ravel() for i in idxs])), 99)
    if hall:
        Bz_lim = np.percentile(np.abs(np.concatenate(
            [snaps[i]["Bz"].ravel() for i in idxs])), 99.5)

    nrows, ncols = (3 if hall else 2), len(idxs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle(title or f"Harris sheet reconnection ({_regime(snaps)}): key frames",
                 fontsize=14, fontweight="bold")

    for col, idx in enumerate(idxs):
        d = snaps[idx]
        x1d = np.linspace(d["x"][0, 0], d["x"][0, -1], d["nx"])
        y1d = np.linspace(d["y"][0, 0], d["y"][-1, 0], d["ny"])

        ax = axes[0, col]
        im = ax.pcolormesh(d["x"], d["y"], compute_jz(d), shading="auto",
                           cmap="RdBu_r", vmin=-Jz_lim, vmax=Jz_lim)
        ax.streamplot(x1d, y1d, d["Bx"], d["By"], density=1.0, linewidth=0.6,
                      color="k", arrowsize=0.7)
        ax.set_title(rf"$J_z$ + field lines, $t={d['t']:.1f}$", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.85, label=r"$J_z$")

        ax = axes[1, col]
        im = ax.pcolormesh(d["x"], d["y"], d["p"], shading="auto", cmap="inferno")
        ax.set_title(rf"pressure $p$, $t={d['t']:.1f}$", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.85, label=r"$p$")

        if hall:
            ax = axes[2, col]
            im = ax.pcolormesh(d["x"], d["y"], d["Bz"], shading="auto",
                               cmap="RdBu_r", vmin=-Bz_lim, vmax=Bz_lim)
            ax.set_title(rf"$B_z$ (Hall quadrupole), $t={d['t']:.1f}$", fontsize=11)
            fig.colorbar(im, ax=ax, shrink=0.85, label=r"$B_z$")
            ax.text(0.02, 0.97, rf"$|B_z|_\mathrm{{max}}={np.max(np.abs(d['Bz'])):.4f}$",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    for ax in axes.flat:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    return save(fig, save_path)


def plot_rate(snaps, save_path=None, title=None):
    """Time series: inflow speed, peak Jz, reconnected flux, midplane |By|,
    and (Hall runs) max |Bz|."""
    ts = time_series(snaps)
    panels = [
        ("max_vy", r"$\max|v_y|$", "inflow speed (rate proxy)"),
        ("max_jz", r"$\max J_z$", "peak current density"),
        ("flux", r"$\int_0^{L_x/2}|B_y|\,dx$ at $y=0$", "reconnected flux"),
        ("max_By_mid", r"$\max|B_y|$ at $y\approx 0$", "reconnected field"),
    ]
    if has_hall(snaps):
        panels.append(("max_Bz", r"$\max|B_z|$", r"Hall quadrupole $B_z$"))
    ncols = 3 if len(panels) > 4 else 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(title or f"Reconnection diagnostics ({_regime(snaps)})",
                 fontsize=13, fontweight="bold")
    for k, (ax, (key, ylabel, ttl)) in enumerate(zip(axes.flat, panels)):
        ax.plot(ts["t"], ts[key], f"C{k}", lw=1.5)
        ax.set_xlabel("t"); ax.set_ylabel(ylabel); ax.set_title(ttl)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    return save(fig, save_path)


def make_gif(snaps, save_path, fps=5, title=None):
    """Animation: Jz | (Hall: Bz) | p with colour limits fixed over all frames."""
    hall = has_hall(snaps)
    Jz_list = [compute_jz(s) for s in snaps]
    Jz_lim = np.percentile(np.abs(np.concatenate([J.ravel() for J in Jz_list])), 99.5)
    p_all = np.concatenate([s["p"].ravel() for s in snaps])
    p_lim = (p_all.min(), np.percentile(p_all, 99.5))

    fields = [("Jz", r"current density $J_z$", "RdBu_r", (-Jz_lim, Jz_lim))]
    if hall:
        Bz_lim = np.percentile(np.abs(np.concatenate([s["Bz"].ravel() for s in snaps])), 99.5)
        fields.append(("Bz", r"Hall quadrupole $B_z$", "RdBu_r", (-Bz_lim, Bz_lim)))
    fields.append(("p", r"pressure $p$", "inferno", p_lim))

    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 4.5))
    d0 = snaps[0]
    ims = []
    for ax, (key, lbl, cmap, (lo, hi)) in zip(axes, fields):
        arr = Jz_list[0] if key == "Jz" else d0[key]
        im = ax.pcolormesh(d0["x"], d0["y"], arr, shading="auto", cmap=cmap,
                           vmin=lo, vmax=hi)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(lbl)
        ims.append((key, im))
    head = title or f"Harris sheet reconnection ({_regime(snaps)})"
    ttl = fig.suptitle(f"{head}   t = {d0['t']:.1f}", fontsize=12, fontweight="bold")
    fig.tight_layout()

    def update(frame):
        d = snaps[frame]
        for key, im in ims:
            im.set_array((Jz_list[frame] if key == "Jz" else d[key]).ravel())
        ttl.set_text(f"{head}   t = {d['t']:.1f}")
        return [im for _, im in ims]

    anim = FuncAnimation(fig, update, frames=len(snaps), interval=1000 // fps, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved: {save_path}")
    return save_path
