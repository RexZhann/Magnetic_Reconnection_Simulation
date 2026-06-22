"""
whistler_viz.py -- Hall-HLL 哨声波色散测试分析（test 24）
=========================================================
运行方式（项目根目录）：
    ./build/mhd2d 24 64 2 2 1
    python output/test24/whistler_viz.py

物理设置：
    背景场 Bx=1，ρ=1，p=0.1，di=0.1
    初始扰动 δBy = A cos(kx)，δBz = A sin(kx)，k=2π，A=0.01
    Hall-HLL 稳定化方案（Path B）

解析预测：
    哨声波频率  ω  = di·k²·vA = 0.1·(2π)²·1 ≈ 3.948
    Hall-HLL 阻尼率 γ = (π/2)·ω ≈ 6.20
    CFL 约束   dt ≤ mincell²/(π·di·max(B²/ρ))

输出：
    whistler_decay.png   -- log(振幅) vs t 及指数衰减拟合
    whistler_signal.png  -- By(x=0, t) 时间演化
"""

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.dirname(_HERE)

TEST_ID  = 24
PATTERN  = f"test{TEST_ID}_64x2_hlld_ct_snap*.dat"
DI       = 0.1
K        = 2.0 * np.pi
A0       = 0.01

OUT_DECAY  = os.path.join(_HERE, "whistler_decay.png")
OUT_SIGNAL = os.path.join(_HERE, "whistler_signal.png")


def load_snapshot(fname):
    with open(fname) as f:
        header = f.readline().split()
    nx, ny = int(header[0]), int(header[1])
    d = {"nx": nx, "ny": ny,
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
            f"找不到 '{pattern}'。\n  运行：./build/mhd2d 24 64 2 2 1"
        )
    snaps = [load_snapshot(f) for f in files]
    print(f"  已加载 {len(snaps)} 个快照   t = {snaps[0]['t']:.4f} 至 {snaps[-1]['t']:.4f}")
    return snaps


def amplitude(snap):
    """返回 By 扰动的空间最大振幅（沿 x 方向）。"""
    return np.max(np.abs(snap["By"]))


def plot_decay(snaps, save_path=OUT_DECAY):
    """
    拟合初期衰减阶段（Alfvén 波生长前）的 γ 并与理论值对比。
    初始条件 δvy=0 不是纯哨声本征函数：理想 MHD 扫描生成 Alfvén 速度分量
    (max|vy| 由 0 增长)，t>0.1 后与哨声波耦合造成 max|By| 非单调。
    因此只对初期线性衰减段（t <= T_EARLY）拟合 γ。
    """
    t_arr = np.array([s["t"] for s in snaps])
    A_arr = np.array([amplitude(s) for s in snaps])
    Bz_arr = np.array([np.max(np.abs(s["Bz"])) for s in snaps])
    vy_arr = np.array([np.max(np.abs(s["vy"])) for s in snaps])

    # 解析预测（短波近似 k*di=0.628 → 修正系数约 1.17）
    omega_theory = DI * K**2          # ω_w = di·k²·vA（短波近似），vA=1
    gamma_theory = (np.pi / 2) * omega_theory

    # 仅对初期 Alfvén 污染前的阶段（t <= 0.10）拟合
    T_EARLY = 0.10
    early = t_arr <= T_EARLY + 1e-9
    t_fit = t_arr[early]
    A_fit = A_arr[early]

    def model(t, gamma, A0_fit):
        return A0_fit * np.exp(-gamma * t)

    popt, pcov = curve_fit(model, t_fit, A_fit,
                           p0=[gamma_theory, A0], maxfev=5000)
    gamma_fit, A0_fit = popt
    gamma_err = np.sqrt(np.diag(pcov))[0]

    print(f"\n【哨声波 Hall-HLL 阻尼分析】")
    print(f"  短波近似 ω  = di·k² = {omega_theory:.4f}")
    print(f"  理论 γ（Hall-HLL）= (π/2)·ω = {gamma_theory:.4f}")
    print(f"  初期拟合 γ（t≤{T_EARLY}）= {gamma_fit:.4f} ± {gamma_err:.4f}")
    print(f"  γ/ω（拟合）= {gamma_fit/omega_theory:.4f}   理论 = π/2 = {np.pi/2:.4f}")
    print(f"  说明：t>{T_EARLY} 时理想 MHD 生成 Alfvén 波（δvy 增长），导致 max|By| 非单调。")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"哨声波 Hall-HLL 稳定化测试（test 24，$d_i=0.1$，$k=2\pi$）",
                 fontsize=13, fontweight="bold")

    # 左图：三个分量幅值随时间演化
    ax = axes[0]
    ax.semilogy(t_arr, A_arr,  "C0o-", ms=3, lw=1.2, label=r"$\max|B_y|$（哨声波）")
    ax.semilogy(t_arr, Bz_arr, "C1s-", ms=3, lw=1.2, label=r"$\max|B_z|$（哨声波）")
    ax.semilogy(t_arr, vy_arr+1e-12, "C2^-", ms=3, lw=1.2, label=r"$\max|v_y|$（Alfvén 波）")
    # 理论衰减线（仅早期段）
    t_line = np.linspace(0, t_arr[-1], 500)
    ax.semilogy(t_line, model(t_line, gamma_theory, A0),
                "k--", lw=1.5, label=rf"理论 $e^{{-\gamma t}}$，$\gamma={gamma_theory:.2f}$")
    ax.semilogy(t_line, model(t_line, gamma_fit, A0_fit),
                "C3-", lw=1.5, label=rf"初期拟合 $\gamma={gamma_fit:.2f}\pm{gamma_err:.2f}$（$t\leq{T_EARLY}$）")
    ax.axvline(T_EARLY, color="gray", ls=":", lw=1, label=f"拟合截止 t={T_EARLY}")
    ax.set_xlabel("t"); ax.set_ylabel("幅值（半对数）")
    ax.set_title("各分量幅值演化（Alfvén-哨声耦合可见）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax.set_ylim(1e-6, 2e-2)

    # 右图：初期衰减线性坐标放大
    ax = axes[1]
    early_idx = np.where(early)[0]
    ax.plot(t_arr[early], A_arr[early], "C0o", ms=5, label=r"$\max|B_y|$（数值）")
    t_e = np.linspace(0, T_EARLY, 200)
    ax.plot(t_e, model(t_e, gamma_theory, A0),
            "k--", lw=1.5, label=rf"理论 $\gamma={gamma_theory:.3f}$")
    ax.plot(t_e, model(t_e, gamma_fit, A0_fit),
            "C3-", lw=1.5, label=rf"拟合 $\gamma={gamma_fit:.3f}\pm{gamma_err:.3f}$")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\max|B_y|$")
    ax.set_title(rf"初期哨声波衰减（$t \leq {T_EARLY}$）")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 标注关键数值
    txt = (f"ω 解析 = {omega_theory:.4f}（短波近似）\n"
           f"γ 理论 = (π/2)·ω = {gamma_theory:.4f}\n"
           f"γ 拟合 = {gamma_fit:.4f} ± {gamma_err:.4f}\n"
           f"γ/ω 拟合 = {gamma_fit/omega_theory:.3f}\n"
           f"γ/ω 理论 = π/2 = {np.pi/2:.3f}")
    axes[1].text(0.97, 0.97, txt, transform=axes[1].transAxes,
                 va="top", ha="right", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存：{save_path}")


def plot_signal(snaps, save_path=OUT_SIGNAL):
    """
    绘制 By(x, t=const) 空间分布（早期几个快照）和
    By(x=0, t) 随时间演化，显示振荡与衰减。
    """
    omega_theory = DI * K**2
    gamma_theory = (np.pi / 2) * omega_theory

    # 早期快照（振幅可见时期）的空间分布
    t_arr  = np.array([s["t"] for s in snaps])
    A_arr  = np.array([amplitude(s) for s in snaps])
    early  = np.where(A_arr > A0 * 0.01)[0]  # 振幅 > 1% 初始值的快照
    n_show = min(len(early), 8)
    idxs   = np.linspace(0, len(early)-1, n_show, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"哨声波时间演化（Hall-HLL 阻尼，$d_i=0.1$）",
                 fontsize=13, fontweight="bold")

    # 左图：几个时刻的 By 空间剖面
    ax = axes[0]
    cmap = plt.cm.viridis
    for k_idx, idx in enumerate(idxs):
        s = snaps[early[idx]]
        color = cmap(k_idx / max(n_show - 1, 1))
        ax.plot(s["x"][0, :], s["By"][0, :], color=color,
                lw=1.5, label=f"t={s['t']:.2f}")
    x_ref = snaps[0]["x"][0, :]
    ax.plot(x_ref, A0 * np.cos(K * x_ref), "k--", lw=1, alpha=0.5, label="初始 A cos(kx)")
    ax.set_xlabel("x"); ax.set_ylabel(r"$B_y$")
    ax.set_title(r"$B_y(x,t)$ 空间分布（早期）")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # 右图：By(x=0, t) 时间序列（含理论包络）
    ax = axes[1]
    t_all  = np.array([s["t"] for s in snaps])
    By_x0  = np.array([s["By"][0, 0] for s in snaps])  # x≈0 处（第 0 列）
    envelope_th = A0 * np.exp(-gamma_theory * t_all)

    ax.plot(t_all, By_x0, "C0o-", ms=2, lw=1, label=r"数值 $B_y(x\approx0,t)$")
    ax.plot(t_all,  envelope_th, "k--", lw=1.2, label=rf"理论包络 $e^{{-\gamma t}}$，$\gamma={gamma_theory:.3f}$")
    ax.plot(t_all, -envelope_th, "k--", lw=1.2)
    ax.set_xlabel("t"); ax.set_ylabel(r"$B_y(x\approx0)$")
    ax.set_title(r"$B_y$ 时间序列（含理论包络）")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存：{save_path}")


if __name__ == "__main__":
    print("加载快照...")
    snaps = load_snapshots(PATTERN)

    print("\n生成阻尼率拟合图...")
    plot_decay(snaps)

    print("\n生成哨声波时间演化图...")
    plot_signal(snaps)

    print("\n完成。")
