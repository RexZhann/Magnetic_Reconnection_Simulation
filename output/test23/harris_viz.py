"""
harris_viz.py -- Hall-HLL 稳定化 Harris 电流片可视化（test 23）
==============================================================
运行方式（项目根目录）：
    ./build/mhd2d 23 128 64 2 1
    python output/test23/harris_viz.py

输出（保存至 output/test23/）：
    harris_keyframes_hall_hll.png   -- 3 个关键时刻 Jz+磁力线 | Bz 四极
    harris_rate_hall_hll.png        -- 时间序列诊断量
    harris_reconnection_hall_hll.gif -- 动画
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.dirname(_HERE)

TEST_ID        = 23
PATTERN        = f"test{TEST_ID}_128x64_hlld_ct_snap*.dat"
KEYFRAME_TIMES = (5.0, 10.0, 14.0)
GIF_FPS        = 5
OUT_KEYFRAMES  = os.path.join(_HERE, "harris_keyframes_hall_hll.png")
OUT_RATE       = os.path.join(_HERE, "harris_rate_hall_hll.png")
OUT_GIF        = os.path.join(_HERE, "harris_reconnection_hall_hll.gif")


def load_snapshot(fname):
    with open(fname) as f:
        header = f.readline().split()
    nx, ny = int(header[0]), int(header[1])
    d = {"nx": nx, "ny": ny, "gama": float(header[2]),
         "t": float(header[4]) if len(header) >= 5 else 0.0}
    data = np.loadtxt(fname, skiprows=1)
    for k, key in enumerate(["x","y","rho","vx","vy","vz","p","Bx","By","Bz","psi","e","divB"]):
        d[key] = data[:, k].reshape(ny, nx)
    return d


def load_snapshots(pattern=PATTERN):
    files = sorted(glob.glob(os.path.join(_HERE, pattern)))
    if not files:
        files = sorted(glob.glob(os.path.join(_OUTPUT_DIR, pattern)))
    if not files:
        raise FileNotFoundError(
            f"找不到匹配 '{pattern}' 的快照文件。\n"
            f"  已搜索：{_HERE}  和  {_OUTPUT_DIR}\n"
            f"  请先运行：./build/mhd2d 23 128 64 2 1"
        )
    snaps = [load_snapshot(f) for f in files]
    print(f"  已加载 {len(snaps)} 个快照   t = {snaps[0]['t']:.2f} 至 {snaps[-1]['t']:.2f}")
    return snaps


def compute_jz(d):
    dx = d["x"][0, 1] - d["x"][0, 0]
    dy = d["y"][1, 0] - d["y"][0, 0]
    return np.gradient(d["By"], dx, axis=1) - np.gradient(d["Bx"], dy, axis=0)


def diagnostics(snaps):
    t      = np.array([s["t"] for s in snaps])
    max_vy = np.array([np.max(np.abs(s["vy"])) for s in snaps])
    max_jz = np.array([np.max(compute_jz(s)) for s in snaps])
    jmid   = snaps[0]["ny"] // 2
    max_By = np.array([np.max(np.abs(s["By"][jmid, :])) for s in snaps])
    max_bz = np.array([np.max(np.abs(s["Bz"])) for s in snaps])
    return t, max_vy, max_jz, max_By, max_bz


def reconnected_flux(snaps):
    phi = []
    for s in snaps:
        jmid = s["ny"] // 2
        x    = s["x"][jmid, :]
        By   = s["By"][jmid, :]
        xmid = 0.5 * (x[0] + x[-1])
        phi.append(np.trapz(np.abs(By[x <= xmid]), x[x <= xmid]))
    return np.array(phi)


def plot_keyframes(snaps, times=KEYFRAME_TIMES, save_path=OUT_KEYFRAMES):
    """N行 x 2列：Jz+磁力线（左）| Bz 四极（右）"""
    t_arr = np.array([s["t"] for s in snaps])
    idxs  = [np.argmin(np.abs(t_arr - t)) for t in times]
    nrows = len(idxs)

    Jz_lim = np.percentile(
        np.abs(np.concatenate([compute_jz(snaps[i]).ravel() for i in idxs])), 99)
    Bz_lim = np.percentile(
        np.abs(np.concatenate([snaps[i]["Bz"].ravel() for i in idxs])), 99.5)

    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows),
                             constrained_layout=True)
    if nrows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        r"Harris 电流片：Hall MHD + Hall-HLL 稳定化（$d_i=1$）——关键时刻",
        fontsize=13, fontweight="bold")

    for row, idx in enumerate(idxs):
        d   = snaps[idx]
        Jz  = compute_jz(d)
        x1d = np.linspace(d["x"][0, 0], d["x"][0, -1], d["nx"])
        y1d = np.linspace(d["y"][0, 0], d["y"][-1, 0], d["ny"])

        ax = axes[row, 0]
        im = ax.pcolormesh(d["x"], d["y"], Jz, shading="auto",
                           cmap="RdBu_r", vmin=-Jz_lim, vmax=Jz_lim)
        ax.streamplot(x1d, y1d, d["Bx"], d["By"],
                      density=1.0, linewidth=0.6, color="k", arrowsize=0.7)
        ax.set_title(rf"$J_z$ + 磁力线，$t={d['t']:.1f}$", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        cb = fig.colorbar(im, ax=ax, label=r"$J_z$")
        cb.ax.set_aspect(20)

        ax = axes[row, 1]
        im = ax.pcolormesh(d["x"], d["y"], d["Bz"], shading="auto",
                           cmap="RdBu_r", vmin=-Bz_lim, vmax=Bz_lim)
        ax.set_title(rf"$B_z$（Hall 四极），$t={d['t']:.1f}$", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        cb = fig.colorbar(im, ax=ax, label=r"$B_z$")
        cb.ax.set_aspect(20)
        peak_bz = np.max(np.abs(d["Bz"]))
        ax.text(0.02, 0.97, rf"$|B_z|_\mathrm{{max}}={peak_bz:.4f}$",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存：{save_path}")


def plot_rate(snaps, save_path=OUT_RATE):
    """2x2 时间序列：流入速度、峰值电流密度、重联通量、四极 Bz"""
    t, max_vy, max_jz, max_By, max_bz = diagnostics(snaps)
    phi = reconnected_flux(snaps)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        r"重联诊断量：Hall MHD + Hall-HLL 稳定化（$d_i=1$）",
        fontsize=12, fontweight="bold")

    axes[0, 0].plot(t, max_vy, "C0", lw=1.5)
    axes[0, 0].set_xlabel("t"); axes[0, 0].set_ylabel(r"$\max|v_y|$")
    axes[0, 0].set_title("流入速度（重联率代理量）")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, max_jz, "C1", lw=1.5)
    axes[0, 1].set_xlabel("t"); axes[0, 1].set_ylabel(r"$\max J_z$")
    axes[0, 1].set_title("峰值电流密度")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, phi, "C2", lw=1.5)
    axes[1, 0].set_xlabel("t"); axes[1, 0].set_ylabel(r"$\int_0^{L_x/2}|B_y|\,dx$ at $y=0$")
    axes[1, 0].set_title("重联磁通量（中平面积分）")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t, max_bz, "C3", lw=1.5)
    axes[1, 1].set_xlabel("t"); axes[1, 1].set_ylabel(r"$\max|B_z|$")
    axes[1, 1].set_title(r"Hall 四极 $|B_z|$ 增长")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存：{save_path}")


def make_gif(snaps, save_path=OUT_GIF, fps=GIF_FPS):
    Jz_list = [compute_jz(s) for s in snaps]
    Jz_lim  = np.percentile(np.abs(np.concatenate([J.ravel() for J in Jz_list])), 99.5)
    Bz_all  = np.concatenate([s["Bz"].ravel() for s in snaps])
    Bz_lim  = np.percentile(np.abs(Bz_all), 99.5)
    p_all   = np.concatenate([s["p"].ravel() for s in snaps])
    p_lim   = (p_all.min(), np.percentile(p_all, 99.5))

    fig, (axJ, axBz, axP) = plt.subplots(1, 3, figsize=(18, 4.5))
    d0 = snaps[0]
    imJ  = axJ.pcolormesh(d0["x"], d0["y"], Jz_list[0], shading="auto",
                           cmap="RdBu_r", vmin=-Jz_lim, vmax=Jz_lim)
    imBz = axBz.pcolormesh(d0["x"], d0["y"], d0["Bz"], shading="auto",
                            cmap="RdBu_r", vmin=-Bz_lim, vmax=Bz_lim)
    imP  = axP.pcolormesh(d0["x"], d0["y"], d0["p"], shading="auto",
                           cmap="inferno", vmin=p_lim[0], vmax=p_lim[1])

    plt.colorbar(imJ,  ax=axJ,  shrink=0.85, label=r"$J_z$")
    plt.colorbar(imBz, ax=axBz, shrink=0.85, label=r"$B_z$")
    plt.colorbar(imP,  ax=axP,  shrink=0.85, label=r"$p$")
    for ax, lbl in [(axJ,  r"电流密度 $J_z$"),
                    (axBz, r"Hall 四极 $B_z$"),
                    (axP,  r"热压强 $p$")]:
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(lbl)

    ttl = fig.suptitle(
        f"Hall MHD + Hall-HLL 稳定化重联   t = {d0['t']:.1f}",
        fontsize=12, fontweight="bold")
    fig.tight_layout()

    def update(frame):
        d = snaps[frame]
        imJ.set_array(Jz_list[frame].ravel())
        imBz.set_array(d["Bz"].ravel())
        imP.set_array(d["p"].ravel())
        ttl.set_text(f"Hall MHD + Hall-HLL 稳定化重联   t = {d['t']:.1f}")
        return [imJ, imBz, imP]

    anim = FuncAnimation(fig, update, frames=len(snaps),
                          interval=1000 // fps, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  已保存：{save_path}")


if __name__ == "__main__":
    print("加载快照...")
    snaps = load_snapshots(PATTERN)

    print("生成关键时刻图...")
    plot_keyframes(snaps)

    print("生成重联率时间序列图...")
    plot_rate(snaps)

    print("生成动画 GIF...")
    make_gif(snaps)

    print("完成。")
