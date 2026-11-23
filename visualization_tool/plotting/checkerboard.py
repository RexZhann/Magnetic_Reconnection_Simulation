"""Hall-HLL checkerboard diagnostic (former output/debug_hll_checkerboard.py).

1. Corner whistler speed c_w = pi d_i |B|_corner / (rho_corner h) on a snapshot.
2. 1D unit test: one HLL Faraday step on a grid-scale checkerboard in By,
   for normal / X-point / near-null background fields (damped vs amplified).
3. Figure: Jz | |B| | log10 c_w with c_w contours, to see whether low-c_w
   corners coincide with the checkerboard.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..physics import compute_jz, grid_spacing
from ._common import save


def corner_cw(snap, di=1.0, rho_floor=0.1):
    """c_w at the cell corners, shape (nx+1, ny+1) (index [I, J]) (periodic x, clamped y)."""
    nx, ny = snap["nx"], snap["ny"]
    dx, dy = grid_spacing(snap)
    h = min(dx, dy)
    I = np.arange(nx + 1)[:, None]
    J = np.arange(ny + 1)[None, :]
    iL, iR = (I - 1) % nx, I % nx
    jD, jU = np.maximum(J - 1, 0), np.minimum(J, ny - 1)

    def avg(f):
        return 0.25 * (f[jD, iL] + f[jD, iR] + f[jU, iL] + f[jU, iR])

    rho = np.maximum(avg(snap["rho"]), rho_floor)
    B = np.sqrt(avg(snap["Bx"]) ** 2 + avg(snap["By"]) ** 2 + avg(snap["Bz"]) ** 2)
    return np.pi * di * B / (rho * h)


def checkerboard_step_test(di=1.0, h=4 * np.pi / 128, dt=1.77e-4, eps=1e-3,
                           rho_bg=0.2, n=32):
    """Growth of a By checkerboard after one HLL Faraday step, per background field."""
    out = []
    for B_bg in (1.0, 0.10, 0.01):
        cw = np.pi * di * B_bg / (rho_bg * h)
        by = B_bg + eps * (-1.0) ** np.arange(n)
        emf = 0.5 * cw * (by[np.arange(n + 1) % n] - by[(np.arange(n + 1) - 1) % n])
        by_new = by + (dt / h) * (emf[1:] - emf[:-1])
        growth = (np.std(by_new) - np.std(by)) / np.std(by)
        out.append(dict(B_bg=B_bg, c_w=cw, nu=cw * dt / h, growth=growth))
        print(f"  B_bg={B_bg:<5} c_w={cw:9.4f}  nu=c_w*dt/h={cw*dt/h:.4f}  "
              f"growth/step={growth:+.4f} ({'damped' if growth < 0 else 'AMPLIFIED'})")
    return out


def plot_checkerboard(snap, save_path=None, di=1.0, rho_floor=0.1):
    cw = corner_cw(snap, di, rho_floor)
    print(f"  c_w: min={cw.min():.4f} max={cw.max():.4f} mean={cw.mean():.4f}; "
          + ", ".join(f"<{v}: {100 * (cw < v).mean():.1f}%" for v in (1, 5, 10)))
    dx, dy = grid_spacing(snap)
    x, y = snap["x"], snap["y"]
    xc = np.linspace(x[0, 0] - dx / 2, x[0, -1] + dx / 2, snap["nx"] + 1)
    yc = np.linspace(y[0, 0] - dy / 2, y[-1, 0] + dy / 2, snap["ny"] + 1)
    Jz = compute_jz(snap)
    lim = np.percentile(np.abs(Jz), 99)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"HLL checkerboard diagnostics, t={snap['t']:.1f}",
                 fontsize=13, fontweight="bold")
    im = axes[0].pcolormesh(x, y, Jz, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0].set_title(r"$J_z$"); fig.colorbar(im, ax=axes[0])
    im = axes[1].pcolormesh(x, y, np.sqrt(snap["Bx"]**2 + snap["By"]**2 + snap["Bz"]**2),
                            shading="auto", cmap="plasma")
    axes[1].set_title(r"$|B|$"); fig.colorbar(im, ax=axes[1])
    im = axes[2].pcolormesh(xc, yc, np.log10(cw.T + 1e-3), shading="auto", cmap="YlOrRd_r")
    cs = axes[2].contour(xc, yc, cw.T, levels=[5, 10, 20], colors=["k", "b", "g"],
                         linewidths=0.8)
    axes[2].clabel(cs, fmt="c_w=%.0f", fontsize=8)
    axes[2].set_title(r"$\log_{10} c_w$ (low = weak HLL damping)")
    fig.colorbar(im, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    return save(fig, save_path)
