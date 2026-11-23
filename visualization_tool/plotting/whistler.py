"""Whistler dispersion test (test 24): Hall-HLL damping rate vs theory.

Former output/test24/whistler_viz.py. Setup: Bx=1, rho=1, p=0.1, d_i=0.1,
dBy = A cos(kx), dBz = A sin(kx), k=2pi, A=0.01.
Theory (short-wave limit): omega = d_i k^2 v_A, Hall-HLL gamma = (pi/2) omega.
The initial state is not a pure whistler eigenmode (dvy=0): ideal MHD
generates an Alfven component that couples in after t~0.1, so gamma is
fitted on the early segment t <= t_early only.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..io import times
from ._common import save


def _pm(err, fmt):
    """'\\pm err' for mathtext labels; empty when the fit error is undefined."""
    return rf"\pm{err:{fmt}}" if np.isfinite(err) else r"\ (\mathrm{error\ n/a})"


def _model(t, gamma, a0):
    return a0 * np.exp(-gamma * t)


def fit_damping(snaps, di=0.1, k=2 * np.pi, a0=0.01, t_early=0.10):
    """Fit A(t) = A0 exp(-gamma t) to max|By| for t <= t_early."""
    from scipy.optimize import curve_fit      # lazy: scipy only needed here
    t = times(snaps)
    A = np.array([np.max(np.abs(s["By"])) for s in snaps])
    omega = di * k ** 2
    gamma_th = 0.5 * np.pi * omega
    early = t <= t_early + 1e-9
    popt, pcov = curve_fit(_model, t[early], A[early], p0=[gamma_th, a0], maxfev=5000)
    return dict(t=t, A=A, early=early, omega=omega, gamma_theory=gamma_th,
                gamma_fit=popt[0], a0_fit=popt[1], gamma_err=float(np.sqrt(pcov[0, 0])),
                t_early=t_early, a0=a0)


def plot_decay(snaps, save_path=None, di=0.1, k=2 * np.pi, a0=0.01, t_early=0.10):
    r = fit_damping(snaps, di, k, a0, t_early)
    t, A, early = r["t"], r["A"], r["early"]
    g_th, g_fit, g_err = r["gamma_theory"], r["gamma_fit"], r["gamma_err"]
    print(f"  omega (short-wave) = {r['omega']:.4f};  gamma theory = {g_th:.4f};  "
          f"gamma fit (t<={t_early}) = {g_fit:.4f} +/- {g_err:.4f};  "
          f"gamma/omega = {g_fit / r['omega']:.4f} (theory pi/2 = {np.pi/2:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(rf"Whistler Hall-HLL damping test ($d_i={di}$, $k={k:.3g}$)",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    ax.semilogy(t, A, "C0o-", ms=3, lw=1.2, label=r"$\max|B_y|$ (whistler)")
    ax.semilogy(t, [np.max(np.abs(s["Bz"])) for s in snaps], "C1s-", ms=3, lw=1.2,
                label=r"$\max|B_z|$ (whistler)")
    ax.semilogy(t, np.array([np.max(np.abs(s["vy"])) for s in snaps]) + 1e-12, "C2^-",
                ms=3, lw=1.2, label=r"$\max|v_y|$ (Alfven)")
    tl = np.linspace(0, t[-1], 500)
    ax.semilogy(tl, _model(tl, g_th, a0), "k--", lw=1.5,
                label=rf"theory $\gamma={g_th:.2f}$")
    ax.semilogy(tl, _model(tl, g_fit, r["a0_fit"]), "C3-", lw=1.5,
                label=rf"early fit $\gamma={g_fit:.2f}{_pm(g_err, '.2f')}$")
    ax.axvline(t_early, color="gray", ls=":", lw=1, label=f"fit cut t={t_early}")
    ax.set_xlabel("t"); ax.set_ylabel("amplitude")
    ax.set_title("component amplitudes (Alfven-whistler coupling visible)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax.set_ylim(1e-6, 2 * a0)

    ax = axes[1]
    te = np.linspace(0, t_early, 200)
    ax.plot(t[early], A[early], "C0o", ms=5, label=r"$\max|B_y|$ (numerical)")
    ax.plot(te, _model(te, g_th, a0), "k--", lw=1.5, label=rf"theory $\gamma={g_th:.3f}$")
    ax.plot(te, _model(te, g_fit, r["a0_fit"]), "C3-", lw=1.5,
            label=rf"fit $\gamma={g_fit:.3f}{_pm(g_err, '.3f')}$")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\max|B_y|$")
    ax.set_title(rf"early whistler decay ($t\leq{t_early}$)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.text(0.97, 0.97,
            f"omega = {r['omega']:.4f}\ngamma theory = {g_th:.4f}\n"
            f"gamma fit = {g_fit:.4f} +/- {g_err:.4f}\n"
            f"gamma/omega fit = {g_fit / r['omega']:.3f}\n"
            f"gamma/omega theory = pi/2 = {np.pi/2:.3f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    fig.tight_layout()
    return save(fig, save_path)


def plot_signal(snaps, save_path=None, di=0.1, k=2 * np.pi, a0=0.01):
    """By(x) profiles at early times and By(x~0, t) with the theory envelope."""
    gamma_th = 0.5 * np.pi * di * k ** 2
    t = times(snaps)
    A = np.array([np.max(np.abs(s["By"])) for s in snaps])
    early = np.where(A > 0.01 * a0)[0]
    n_show = min(len(early), 8)
    idxs = np.linspace(0, len(early) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(rf"Whistler time evolution (Hall-HLL damping, $d_i={di}$)",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    for c, idx in enumerate(idxs):
        s = snaps[early[idx]]
        ax.plot(s["x"][0, :], s["By"][0, :], color=plt.cm.viridis(c / max(n_show - 1, 1)),
                lw=1.5, label=f"t={s['t']:.2f}")
    xr = snaps[0]["x"][0, :]
    ax.plot(xr, a0 * np.cos(k * xr), "k--", lw=1, alpha=0.5, label="initial A cos(kx)")
    ax.set_xlabel("x"); ax.set_ylabel(r"$B_y$"); ax.set_title(r"$B_y(x,t)$ (early)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    ax = axes[1]
    env = a0 * np.exp(-gamma_th * t)
    ax.plot(t, [s["By"][0, 0] for s in snaps], "C0o-", ms=2, lw=1,
            label=r"numerical $B_y(x\approx0,t)$")
    ax.plot(t, env, "k--", lw=1.2, label=rf"theory envelope $\gamma={gamma_th:.3f}$")
    ax.plot(t, -env, "k--", lw=1.2)
    ax.set_xlabel("t"); ax.set_ylabel(r"$B_y(x\approx0)$")
    ax.set_title(r"$B_y$ time series with theory envelope")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    return save(fig, save_path)
