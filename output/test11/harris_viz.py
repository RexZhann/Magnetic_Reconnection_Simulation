"""
harris_viz.py — Visualisation for Harris Sheet Magnetic Reconnection
=====================================================================
Supports test 11 (resistive MHD) and test 12 (Hall MHD).

Usage (from project root):
    # Resistive MHD (test 11)
    ./build/mhd2d 11 128 64 2 1
    ./build/mhd2d 11 512 256 2 1

    # Hall MHD (test 12) — GEM challenge
    ./build/mhd2d 12 128 64 2 1
    ./build/mhd2d 12 512 256 2 1

    python output/test11/harris_viz.py   # set TEST_ID and PATTERN below

Requires: numpy, matplotlib, pillow  (pip install pillow)
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering – change to "TkAgg" if you want
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ──────────────────────────────────────────────────────────────
# Set TEST_ID to 11 (resistive MHD) or 12 (Hall MHD) to match your run.
TEST_ID        = 11
PATTERN        = f"test{TEST_ID}_128x64_hlld_ct_snap*.dat"
KEYFRAME_TIMES = (2.0, 8.0, 20.0)   # target t for the 3 key frames
GIF_FPS        = 5
OUT_KEYFRAMES  = os.path.join(_HERE, f"test{TEST_ID}_keyframes.png")
OUT_RATE       = os.path.join(_HERE, f"test{TEST_ID}_rate.png")
OUT_GIF        = os.path.join(_HERE, f"test{TEST_ID}_reconnection.gif")
# ──────────────────────────────────────────────────────────────────────────────


# ── 1. Snapshot loader ─────────────────────────────────────────────────────────

def load_snapshot(fname):
    """Load a snapshot file produced by write_snapshot_file().
    Header: nx  ny  gamma  glm  t    (5 fields; last = simulation time)
    Columns: x y rho vx vy vz p Bx By Bz psi e divB
    """
    with open(fname) as f:
        header = f.readline().split()
    nx, ny = int(header[0]), int(header[1])
    d = {}
    d["nx"], d["ny"] = nx, ny
    d["gama"] = float(header[2])
    d["t"]    = float(header[4]) if len(header) >= 5 else 0.0
    data = np.loadtxt(fname, skiprows=1)
    d["x"]    = data[:, 0].reshape(ny, nx)
    d["y"]    = data[:, 1].reshape(ny, nx)
    d["rho"]  = data[:, 2].reshape(ny, nx)
    d["vx"]   = data[:, 3].reshape(ny, nx)
    d["vy"]   = data[:, 4].reshape(ny, nx)
    d["vz"]   = data[:, 5].reshape(ny, nx)
    d["p"]    = data[:, 6].reshape(ny, nx)
    d["Bx"]   = data[:, 7].reshape(ny, nx)
    d["By"]   = data[:, 8].reshape(ny, nx)
    d["Bz"]   = data[:, 9].reshape(ny, nx)
    d["psi"]  = data[:, 10].reshape(ny, nx)
    d["e"]    = data[:, 11].reshape(ny, nx)
    d["divB"] = data[:, 12].reshape(ny, nx)
    return d


def load_snapshots(pattern=PATTERN):
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob(os.path.join(_HERE, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found.\n"
            f"  Searched in: {os.getcwd()}  and  {_HERE}\n"
            f"  Run the simulation first:\n"
            f"    ./build/mhd2d {TEST_ID} 128 64 2 1\n"
            f"  Snapshot files appear in output/ as test{TEST_ID}_*_snap*.dat"
        )
    snaps = [load_snapshot(f) for f in files]
    print(f"  Loaded {len(snaps)} snapshots   t = {snaps[0]['t']:.2f} → {snaps[-1]['t']:.2f}")
    return snaps


# ── 2. Physics helpers ─────────────────────────────────────────────────────────

def compute_jz(d):
    """Jz = ∂By/∂x − ∂Bx/∂y  (2nd-order central finite differences)."""
    dx = d["x"][0, 1] - d["x"][0, 0]
    dy = d["y"][1, 0] - d["y"][0, 0]
    return np.gradient(d["By"], dx, axis=1) - np.gradient(d["Bx"], dy, axis=0)


def has_hall(snaps):
    """True if the run produced a non-trivial Bz (Hall MHD signature)."""
    return np.max(np.abs(snaps[-1]["Bz"])) > 1e-6


def diagnostics(snaps):
    """Return (t, max|vy|, max Jz, max|By| at y≈0, max|Bz|) as arrays."""
    t      = np.array([s["t"] for s in snaps])
    max_vy = np.array([np.max(np.abs(s["vy"])) for s in snaps])
    max_jz = np.array([np.max(compute_jz(s)) for s in snaps])
    jmid   = snaps[0]["ny"] // 2
    max_By = np.array([np.max(np.abs(s["By"][jmid, :])) for s in snaps])
    max_Bz = np.array([np.max(np.abs(s["Bz"])) for s in snaps])
    return t, max_vy, max_jz, max_By, max_Bz


# ── 3. Key-frame figure ────────────────────────────────────────────────────────

def plot_keyframes(snaps, times=KEYFRAME_TIMES, save_path=OUT_KEYFRAMES):
    """
    For resistive MHD (test 11): 3-column × 2-row (Jz+streamlines / pressure).
    For Hall MHD    (test 12): 3-column × 3-row (adds Bz quadrupole row).
    """
    t_arr = np.array([s["t"] for s in snaps])
    idxs  = [np.argmin(np.abs(t_arr - t)) for t in times]
    hall  = has_hall(snaps)

    Jz_lim = np.percentile(
        np.abs(np.concatenate([compute_jz(snaps[i]).ravel() for i in idxs])), 99)

    nrows = 3 if hall else 2
    label = "Hall MHD" if hall else "Resistive MHD"
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 4.5 * nrows))
    fig.suptitle(f"Harris Sheet Reconnection ({label}) — Key Frames",
                 fontsize=14, fontweight="bold")

    if hall:
        Bz_lim = np.percentile(
            np.abs(np.concatenate([snaps[i]["Bz"].ravel() for i in idxs])), 99)

    for col, idx in enumerate(idxs):
        d   = snaps[idx]
        Jz  = compute_jz(d)
        x1d = np.linspace(d["x"][0, 0], d["x"][0, -1], d["nx"])
        y1d = np.linspace(d["y"][0, 0], d["y"][-1, 0], d["ny"])

        # Row 0: Jz + field lines
        ax0 = axes[0, col]
        im0 = ax0.pcolormesh(d["x"], d["y"], Jz, shading="auto",
                              cmap="RdBu_r", vmin=-Jz_lim, vmax=Jz_lim)
        ax0.streamplot(x1d, y1d, d["Bx"], d["By"],
                       density=1.0, linewidth=0.6, color="k", arrowsize=0.7)
        ax0.set_title(rf"$J_z$ + field lines,  $t={d['t']:.1f}$", fontsize=11)
        ax0.set_xlabel("x"); ax0.set_ylabel("y")
        plt.colorbar(im0, ax=ax0, shrink=0.85, label=r"$J_z$")

        # Row 1: Thermal pressure
        ax1 = axes[1, col]
        im1 = ax1.pcolormesh(d["x"], d["y"], d["p"], shading="auto", cmap="inferno")
        ax1.set_title(rf"Pressure $p$,  $t={d['t']:.1f}$", fontsize=11)
        ax1.set_xlabel("x"); ax1.set_ylabel("y")
        plt.colorbar(im1, ax=ax1, shrink=0.85, label=r"$p$")

        # Row 2 (Hall only): Out-of-plane Bz — the Hall quadrupole signature
        if hall:
            ax2 = axes[2, col]
            im2 = ax2.pcolormesh(d["x"], d["y"], d["Bz"], shading="auto",
                                  cmap="bwr", vmin=-Bz_lim, vmax=Bz_lim)
            ax2.set_title(rf"Out-of-plane $B_z$ (Hall quadrupole),  $t={d['t']:.1f}$",
                          fontsize=11)
            ax2.set_xlabel("x"); ax2.set_ylabel("y")
            plt.colorbar(im2, ax=ax2, shrink=0.85, label=r"$B_z$")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── 4. Reconnection-rate time series ──────────────────────────────────────────

def plot_rate(snaps, save_path=OUT_RATE):
    """Time series: inflow speed, peak current, reconnected By, and (Hall) max Bz."""
    t, max_vy, max_jz, max_By, max_Bz = diagnostics(snaps)
    hall = has_hall(snaps)

    ncols = 4 if hall else 3
    label = "Hall MHD" if hall else "Resistive MHD"
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4))
    fig.suptitle(f"Reconnection Diagnostics vs Time ({label})",
                 fontsize=13, fontweight="bold")

    axes[0].plot(t, max_vy, "C0", lw=1.5)
    axes[0].set_xlabel("t"); axes[0].set_ylabel(r"$\max|v_y|$")
    axes[0].set_title("Inflow velocity (reconnection-rate proxy)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, max_jz, "C1", lw=1.5)
    axes[1].set_xlabel("t"); axes[1].set_ylabel(r"$\max J_z$")
    axes[1].set_title("Peak current density")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, max_By, "C2", lw=1.5)
    axes[2].set_xlabel("t"); axes[2].set_ylabel(r"$\max|B_y|$ at $y\approx 0$")
    axes[2].set_title("Reconnected field (flux proxy)")
    axes[2].grid(alpha=0.3)

    if hall:
        axes[3].plot(t, max_Bz, "C3", lw=1.5)
        axes[3].set_xlabel("t"); axes[3].set_ylabel(r"$\max|B_z|$")
        axes[3].set_title(r"Hall quadrupole $B_z$ growth")
        axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── 5. Animated GIF ────────────────────────────────────────────────────────────

def make_gif(snaps, save_path=OUT_GIF, fps=GIF_FPS):
    """
    Resistive MHD: Jz (left) + pressure (right).
    Hall MHD:      Jz (left) + Bz quadrupole (centre) + pressure (right).
    Colour limits fixed across all frames.
    """
    hall    = has_hall(snaps)
    Jz_list = [compute_jz(s) for s in snaps]
    Jz_lim  = np.percentile(np.abs(np.concatenate([J.ravel() for J in Jz_list])), 99.5)
    p_all   = np.concatenate([s["p"].ravel() for s in snaps])
    p_lim   = (p_all.min(), np.percentile(p_all, 99.5))

    if hall:
        Bz_all  = np.concatenate([s["Bz"].ravel() for s in snaps])
        Bz_lim  = np.percentile(np.abs(Bz_all), 99.5)
        fig, (axJ, axBz, axP) = plt.subplots(1, 3, figsize=(18, 4.5))
    else:
        fig, (axJ, axP) = plt.subplots(1, 2, figsize=(13, 4.5))

    d0 = snaps[0]
    imJ = axJ.pcolormesh(d0["x"], d0["y"], Jz_list[0], shading="auto",
                          cmap="RdBu_r", vmin=-Jz_lim, vmax=Jz_lim)
    imP = axP.pcolormesh(d0["x"], d0["y"], d0["p"], shading="auto",
                          cmap="inferno", vmin=p_lim[0], vmax=p_lim[1])
    plt.colorbar(imJ, ax=axJ, shrink=0.85, label=r"$J_z$")
    plt.colorbar(imP, ax=axP, shrink=0.85, label=r"$p$")
    axJ.set_xlabel("x"); axJ.set_ylabel("y"); axJ.set_title(r"Current density $J_z$")
    axP.set_xlabel("x"); axP.set_ylabel("y"); axP.set_title(r"Pressure $p$")

    if hall:
        imBz = axBz.pcolormesh(d0["x"], d0["y"], d0["Bz"], shading="auto",
                                cmap="bwr", vmin=-Bz_lim, vmax=Bz_lim)
        plt.colorbar(imBz, ax=axBz, shrink=0.85, label=r"$B_z$")
        axBz.set_xlabel("x"); axBz.set_ylabel("y")
        axBz.set_title(r"Hall quadrupole $B_z$")

    label = "Hall MHD" if hall else "Resistive MHD"
    ttl = fig.suptitle(f"Harris Sheet Reconnection ({label})   t = {d0['t']:.1f}",
                        fontsize=12, fontweight="bold")
    fig.tight_layout()

    def update(frame):
        d = snaps[frame]
        imJ.set_array(Jz_list[frame].ravel())
        imP.set_array(d["p"].ravel())
        ttl.set_text(f"Harris Sheet Reconnection ({label})   t = {d['t']:.1f}")
        artists = [imJ, imP]
        if hall:
            imBz.set_array(d["Bz"].ravel())
            artists.append(imBz)
        return artists

    anim = FuncAnimation(fig, update, frames=len(snaps),
                          interval=1000 // fps, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading snapshots...")
    snaps = load_snapshots(PATTERN)

    print("Generating key-frame figure...")
    plot_keyframes(snaps)

    print("Generating reconnection-rate plot...")
    plot_rate(snaps)

    print("Generating animated GIF...")
    make_gif(snaps)

    print("Done.")
