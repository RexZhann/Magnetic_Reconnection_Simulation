"""Derived quantities computed from snapshots (dicts from io.load_snapshot)."""

from __future__ import annotations

import numpy as np

from .io import times

try:                                  # numpy >= 2.0
    _trapz = np.trapezoid
except AttributeError:                # numpy < 2.0
    _trapz = np.trapz


def grid_spacing(d) -> tuple[float, float]:
    dx = d["x"][0, 1] - d["x"][0, 0] if d["nx"] > 1 else 1.0
    dy = d["y"][1, 0] - d["y"][0, 0] if d["ny"] > 1 else 1.0
    return dx, dy


def compute_jz(d) -> np.ndarray:
    """Jz = dBy/dx - dBx/dy (2nd-order central differences, np.gradient)."""
    dx, dy = grid_spacing(d)
    return np.gradient(d["By"], dx, axis=1) - np.gradient(d["Bx"], dy, axis=0)


def has_hall(snaps, tol: float = 1e-6) -> bool:
    """True if the run developed a non-trivial Bz (Hall signature)."""
    return float(np.max(np.abs(snaps[-1]["Bz"]))) > tol


def reconnected_flux(snaps) -> np.ndarray:
    """Phi = int_{x<=x_mid} |By| dx on the midplane row y ~ 0."""
    phi = []
    for s in snaps:
        jmid = s["ny"] // 2
        x, By = s["x"][jmid, :], s["By"][jmid, :]
        left = x <= 0.5 * (x[0] + x[-1])
        phi.append(_trapz(np.abs(By[left]), x[left]))
    return np.array(phi)


def time_series(snaps) -> dict:
    """Standard reconnection diagnostics, one value per snapshot."""
    jmid = snaps[0]["ny"] // 2
    return {
        "t":      times(snaps),
        "max_vy": np.array([np.max(np.abs(s["vy"])) for s in snaps]),
        "max_jz": np.array([np.max(compute_jz(s)) for s in snaps]),
        "max_By_mid": np.array([np.max(np.abs(s["By"][jmid, :])) for s in snaps]),
        "max_Bz": np.array([np.max(np.abs(s["Bz"])) for s in snaps]),
        "flux":   reconnected_flux(snaps),
    }


def quadrupole_bz(d) -> float:
    """Mean |Bz| over the four quadrants (Hall quadrupole strength)."""
    ny, nx = d["Bz"].shape
    Bz = d["Bz"]
    return float(sum(np.mean(np.abs(q)) for q in (
        Bz[:ny // 2, :nx // 2], Bz[ny // 2:, :nx // 2],
        Bz[:ny // 2, nx // 2:], Bz[ny // 2:, nx // 2:]))) / 4.0


def max_abs_jz_periodic(d) -> float:
    """max |Jz| with periodic (np.roll) central differences."""
    dx, dy = grid_spacing(d)
    dBy_dx = (np.roll(d["By"], -1, axis=1) - np.roll(d["By"], 1, axis=1)) / (2 * dx)
    dBx_dy = (np.roll(d["Bx"], -1, axis=0) - np.roll(d["Bx"], 1, axis=0)) / (2 * dy)
    return float(np.max(np.abs(dBy_dx - dBx_dy)))


def current_sheet_width(d) -> float:
    """FWHM in y of |Jz| along the central x column (periodic differences)."""
    dx, dy = grid_spacing(d)
    dBy_dx = (np.roll(d["By"], -1, axis=1) - np.roll(d["By"], 1, axis=1)) / (2 * dx)
    dBx_dy = (np.roll(d["Bx"], -1, axis=0) - np.roll(d["Bx"], 1, axis=0)) / (2 * dy)
    col = np.abs((dBy_dx - dBx_dy)[:, d["nx"] // 2])
    peak = col.max()
    if peak < 1e-12:
        return float("nan")
    above = np.where(col >= 0.5 * peak)[0]
    return float((above[-1] - above[0] + 1) * dy)


def midplane_flux_max(d) -> float:
    """max |int_0^x By dx'| along the central y row (reconnected-flux proxy)."""
    dx, _ = grid_spacing(d)
    return float(np.max(np.abs(np.cumsum(d["By"][d["ny"] // 2, :]) * dx)))


def flux_function(Bx, By, dx, dy) -> np.ndarray:
    """Az on an (nx, ny) grid from Bx = dAz/dy, By = -dAz/dx; mean removed."""
    ny = By.shape[1]
    a0 = np.cumsum(Bx[0, :]) * dy
    a = a0[np.newaxis, :] - np.cumsum(
        np.vstack([np.zeros(ny), 0.5 * (By[:-1, :] + By[1:, :])]), axis=0) * dx
    return a - a.mean()
