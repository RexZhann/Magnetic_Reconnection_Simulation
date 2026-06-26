"""
analyze.py — η_H 扫描后处理：物理诊断 + 标度律拟合 + 三图输出
===========================================================================
用法（从项目根目录运行）：
    python output/eta_H_sweep/analyze.py

依赖：output/eta_H_sweep/sweep_results.csv（run_sweep.py 生成）
输出：
  output/eta_H_sweep/fig1_survival_map.png      — 存活图（热图）
  output/eta_H_sweep/fig2_overdamp_curves.png   — 过阻尼代价曲线
  output/eta_H_sweep/fig3_scaling_law.png       — Δx² 标度律 log-log 拟合
  output/eta_H_sweep/diagnostics.csv            — 物理诊断量（来自快照）
  output/eta_H_sweep/scaling_fit.txt            — 幂律拟合报告
"""

import os, re, struct, math, glob as glob_mod
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ─── 常量 ───────────────────────────────────────────────────────────────────
CSV_IN   = "output/eta_H_sweep/sweep_results.csv"
OUT_DIR  = "output/eta_H_sweep"
NVAR     = 9    # ρ, vx, vy, vz, p, Bx, By, Bz, ψ（原始量）
RHO, VX, VY, VZ, P, BX, BY, BZ, PSI = range(NVAR)
# ────────────────────────────────────────────────────────────────────────────


def read_dat(path: str):
    """
    读取 ASCII .dat 文件（write_snapshot_file / write_output_file 输出）。
    头行格式：
      快照: nx ny gamma glm t    (5 个值)
      主输出: nx ny gamma glm    (4 个值)
    数据行格式（每行 13 列）：
      x y  rho vx vy vz p Bx By Bz psi  e divB
    返回 (w, nx, ny, t)，w.shape = (NVAR, nx, ny)，各变量按 NVAR 索引。
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().split()
        nx, ny = int(header[0]), int(header[1])
        t_val = float(header[4]) if len(header) >= 5 else 0.0
        data = np.loadtxt(f)  # shape (nx*ny, 13)

    # 布局：cols 0,1 = x,y; cols 2..10 = 9 原始变量; col11=e; col12=divB
    if data.ndim == 1:
        data = data.reshape(1, -1)
    prim = data[:, 2:2 + NVAR]        # shape (nx*ny, NVAR)
    w = prim.reshape(ny, nx, NVAR).transpose(2, 1, 0)  # (NVAR, nx, ny)
    return w, nx, ny, t_val


def read_final_snap(label: str):
    """读取该 run label 的最后一个快照（末帧或最终输出文件）。"""
    folder = f"output/{label}"
    # 优先找 _snap*.dat（最高编号），其次找 .dat 主文件
    snaps = sorted(glob_mod.glob(f"{folder}/*_snap*.dat"))
    if snaps:
        return read_dat(snaps[-1])
    dats = sorted(glob_mod.glob(f"{folder}/*.dat"))
    if dats:
        return read_dat(dats[-1])
    return None


def compute_reconnection_rate(w, nx, ny, Lx, Ly):
    """
    重联率 = dΨ/dt 的代理：末帧磁通量函数最大值（粗估）。
    Ψ(x,y) = ∫₀ˣ By dx'（仅取末帧量级）。
    """
    dx = Lx / nx
    By = w[BY]  # shape (nx, ny)
    # 沿 y 中心行积分
    mid_y = ny // 2
    psi_line = np.cumsum(By[:, mid_y]) * dx
    return float(np.max(np.abs(psi_line)))


def compute_current_sheet_width(w, nx, ny, Ly):
    """
    电流片宽度：Jz = ∂By/∂x − ∂Bx/∂y 的 FWHM（沿 x 中心剖面，取 y 方向）。
    """
    dy = Ly / ny
    dx = Ly / nx   # 近似（等网格时 dx==dy 不对，但取量级足够）
    Bx = w[BX]
    By = w[BY]
    # 中心差分
    dBy_dx = (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / (2 * (Ly / nx * nx / ny))
    dBx_dy = (np.roll(Bx, -1, axis=1) - np.roll(Bx,  1, axis=1)) / (2 * dy)
    Jz = dBy_dx - dBx_dy
    # 沿 x 中心列取 y 剖面
    mid_x = nx // 2
    jz_col = np.abs(Jz[mid_x, :])
    peak = np.max(jz_col)
    if peak < 1e-12:
        return float("nan")
    half = peak / 2.0
    above = np.where(jz_col >= half)[0]
    if len(above) == 0:
        return float("nan")
    fwhm_cells = above[-1] - above[0] + 1
    return fwhm_cells * dy


def compute_max_jz(w, nx, ny, Lx, Ly):
    """返回 max|Jz|（无量纲）。"""
    dy = Ly / ny
    dx = Lx / nx
    Bx = w[BX]
    By = w[BY]
    dBy_dx = (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / (2 * dx)
    dBx_dy = (np.roll(Bx, -1, axis=1) - np.roll(Bx,  1, axis=1)) / (2 * dy)
    Jz = dBy_dx - dBx_dy
    return float(np.max(np.abs(Jz)))


def compute_quadrupole_bz(w, nx, ny):
    """四极 Bz 强度：四象限均值绝对值之和（Hall 特征量）。"""
    mid_x, mid_y = nx // 2, ny // 2
    Bz = w[BZ]
    q11 = np.mean(np.abs(Bz[:mid_x, :mid_y]))
    q12 = np.mean(np.abs(Bz[:mid_x, mid_y:]))
    q21 = np.mean(np.abs(Bz[mid_x:, :mid_y]))
    q22 = np.mean(np.abs(Bz[mid_x:, mid_y:]))
    return float(q11 + q12 + q21 + q22) / 4.0


def run_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """对存活的 run 读取快照做物理诊断。"""
    rows = []
    for _, r in df[df["survived"]].iterrows():
        label = r["label"]
        nx, ny = int(r["nx"]), int(r["ny"])
        pi = math.pi
        Lx = 4 * pi   # harris_sheet.hpp: Lx=4π, Ly=2π（正方形网格 dx=dy=4π/nx）
        Ly = 2 * pi

        snap = read_final_snap(label)
        if snap is None:
            continue
        w, snx, sny, t_snap = snap
        if snx != nx or sny != ny:
            continue  # 尺寸不匹配，跳过

        rec_rate  = compute_reconnection_rate(w, nx, ny, Lx, Ly)
        cs_width  = compute_current_sheet_width(w, nx, ny, Ly)
        max_jz    = compute_max_jz(w, nx, ny, Lx, Ly)
        quad_bz   = compute_quadrupole_bz(w, nx, ny)
        min_rho_s = float(np.min(w[RHO]))

        rows.append({
            "nx": nx, "ny": ny, "dx": r["dx"],
            "eta_H": r["eta_H"],
            "t_snap": t_snap,
            "rec_rate_proxy": rec_rate,
            "cs_width": cs_width,
            "max_jz": max_jz,
            "quad_bz": quad_bz,
            "min_rho_snap": min_rho_s,
        })
    return pd.DataFrame(rows)


def find_eta_min(df_grid: pd.DataFrame):
    """
    对单档分辨率的扫描结果找 η_H^min：最小的存活 η_H 值。
    """
    surv = df_grid[df_grid["survived"]]["eta_H"]
    if surv.empty:
        return float("nan")
    return float(surv.min())


def power_law(x, C, p):
    return C * np.array(x) ** p


def plot_survival_map(df: pd.DataFrame, savepath: str):
    """图1：存活热图（分辨率 × η_H，绿=存活，红=崩溃）。"""
    grids = sorted(df.groupby(["nx", "ny"]).groups.keys())
    fig, axes = plt.subplots(1, len(grids), figsize=(5 * len(grids), 4), sharey=False)
    if len(grids) == 1:
        axes = [axes]

    for ax, (nx, ny) in zip(axes, grids):
        sub = df[(df["nx"] == nx) & (df["ny"] == ny)].sort_values("eta_H", ascending=False)
        etas  = sub["eta_H"].values
        surv  = sub["survived"].values.astype(int)
        colors = ["#2ecc71" if s else "#e74c3c" for s in surv]
        y_pos = range(len(etas))

        ax.barh(list(y_pos), [1] * len(etas), color=colors, edgecolor="k", linewidth=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([f"{e:.2e}" for e in etas], fontsize=8)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_title(f"{nx}×{ny}", fontsize=11)
        ax.set_ylabel("η_H" if (nx, ny) == grids[0] else "")

        # 在条上标注 t_final
        for i, (s, t_f) in enumerate(zip(surv, sub["t_final"].values)):
            label_txt = "✓" if s else f"✗ t={t_f:.1f}"
            ax.text(0.5, i, label_txt, ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#2ecc71", label="存活"),
                  Patch(facecolor="#e74c3c", label="崩溃")]
    fig.legend(handles=legend_els, loc="upper right", fontsize=9)
    fig.suptitle("η_H 稳定性扫描 — 存活图", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"  已保存 {savepath}")


def plot_overdamp_curves(diag: pd.DataFrame, savepath: str):
    """图2：过阻尼代价曲线（存活 run 的 max_jz 和 quad_bz vs η_H，按分辨率分色）。"""
    if diag.empty:
        print("  诊断数据为空，跳过图2。")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = {64: "#3498db", 128: "#e67e22", 256: "#8e44ad"}
    markers = {64: "o", 128: "s", 256: "^"}

    for nx in sorted(diag["nx"].unique()):
        sub = diag[diag["nx"] == nx].sort_values("eta_H")
        c = colors.get(nx, "gray")
        m = markers.get(nx, "x")
        ny = sub["ny"].iloc[0]
        lbl = f"{nx}×{ny}"

        ax1.plot(sub["eta_H"], sub["max_jz"], color=c, marker=m, label=lbl)
        ax2.plot(sub["eta_H"], sub["quad_bz"], color=c, marker=m, label=lbl)

    ax1.set_xscale("log")
    ax1.set_xlabel("η_H", fontsize=11)
    ax1.set_ylabel("max |Jz|", fontsize=11)
    ax1.set_title("电流密度峰值（过阻尼导致峰值降低）", fontsize=10)
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("η_H", fontsize=11)
    ax2.set_ylabel("|Bz|_quad（四象限均值）", fontsize=11)
    ax2.set_title("Hall 四极 Bz（η_H 增大则被抹除）", fontsize=10)
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("过阻尼代价诊断（各分辨率存活 run）", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"  已保存 {savepath}")


def plot_scaling_law(eta_min_dict: dict, savepath: str, fit_report: str):
    """
    图3：log-log 拟合 η_H^min vs Δx。
    eta_min_dict: {(nx,ny,dx): eta_min}
    """
    items = [(dx, em) for (nx, ny, dx), em in eta_min_dict.items()
             if not math.isnan(em)]
    if len(items) < 2:
        print(f"  数据点 < 2，跳过图3。报告：\n{fit_report}")
        return

    items.sort(key=lambda x: x[0])
    dxs, ems = zip(*items)
    dxs, ems = np.array(dxs), np.array(ems)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(dxs, ems, "ko", ms=8, label="测量值 η_H^min")

    if len(items) >= 2:
        try:
            popt, _ = curve_fit(power_law, dxs, ems, p0=[1.0, 2.0])
            C, p = popt
            dx_fit = np.logspace(np.log10(dxs.min() * 0.8),
                                 np.log10(dxs.max() * 1.2), 50)
            ax.loglog(dx_fit, power_law(dx_fit, C, p), "r--",
                      label=f"拟合: η_H^min = {C:.4f}·Δx^{p:.3f}")
        except Exception as e:
            print(f"  拟合失败: {e}")

    # 叠加 Δx²/π² 参考线
    dx_ref = np.logspace(np.log10(dxs.min() * 0.8),
                         np.log10(dxs.max() * 1.2), 50)
    ax.loglog(dx_ref, dx_ref**2 / (math.pi**2), "b:", alpha=0.6,
              label="参考: Δx²/π²（理论锚点）")

    ax.set_xlabel("Δx（网格间距）", fontsize=12)
    ax.set_ylabel("η_H^min（稳定下界）", fontsize=12)
    ax.set_title("Δx² 标度律：η_H^min ∝ Δx^p", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"  已保存 {savepath}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_IN):
        print(f"找不到 {CSV_IN}，请先运行 run_sweep.py。")
        return

    df = pd.read_csv(CSV_IN)
    print(f"读入 {len(df)} 条记录，存活 {df['survived'].sum()} 次。")

    # ─── 物理诊断（读快照）──────────────────────────────────────────────────
    print("\n[1/4] 物理诊断（读取末帧快照）...")
    diag = run_diagnostics(df)
    if not diag.empty:
        diag_path = f"{OUT_DIR}/diagnostics.csv"
        diag.to_csv(diag_path, index=False)
        print(f"  诊断写入 {diag_path}")
    else:
        print("  无法读取任何快照，诊断跳过。")

    # ─── η_H^min 提取 ───────────────────────────────────────────────────────
    print("\n[2/4] 提取各分辨率 η_H^min...")
    eta_min_dict = {}
    pi = math.pi
    summary_lines = []
    for (nx, ny), gdf in df.groupby(["nx", "ny"]):
        dx = float(gdf["dx"].iloc[0])
        anchor = dx**2 / pi**2
        eta_min = find_eta_min(gdf)
        eta_min_dict[(nx, ny, dx)] = eta_min
        ratio = eta_min / anchor if not math.isnan(eta_min) else float("nan")
        crash_t = gdf[~gdf["survived"]]["t_final"].min() if not gdf[gdf["survived"]].empty else float("nan")
        line = (f"  {nx}×{ny}  Δx={dx:.4f}  锚点={anchor:.3e}  "
                f"η_H^min={eta_min:.3e}  比值={ratio:.3f}  首次崩溃t≈{crash_t:.1f}")
        print(line)
        summary_lines.append(line)

    # ─── 幂律拟合 ───────────────────────────────────────────────────────────
    print("\n[3/4] 幂律拟合 η_H^min ∝ Δx^p ...")
    fit_report = "=== η_H^min Δx 标度律拟合报告 ===\n\n"
    fit_report += "\n".join(summary_lines) + "\n\n"

    items_for_fit = [(dx, em) for (nx, ny, dx), em in eta_min_dict.items()
                     if not math.isnan(em) and em > 0]
    if len(items_for_fit) >= 2:
        dxs_f, ems_f = zip(*sorted(items_for_fit))
        dxs_f, ems_f = np.array(dxs_f), np.array(ems_f)
        try:
            popt, pcov = curve_fit(power_law, dxs_f, ems_f, p0=[1.0, 2.0])
            C, p = popt
            perr = np.sqrt(np.diag(pcov))
            fit_report += f"拟合结果：η_H^min = C · Δx^p\n"
            fit_report += f"  C = {C:.6f} ± {perr[0]:.6f}\n"
            fit_report += f"  p = {p:.4f} ± {perr[1]:.4f}\n"
            fit_report += f"\n  理论预期：p ≈ 2.0  参考值：C = 1/π² ≈ {1/pi**2:.4f}\n"
            fit_report += f"\n外推公式：η_H^min(Δx) ≈ {C:.4f} · Δx^{p:.3f}\n"
            print(f"  C={C:.4f}, p={p:.4f}（理论 p≈2.0）")
        except Exception as e:
            fit_report += f"拟合失败: {e}\n"
            print(f"  拟合失败: {e}")
    else:
        fit_report += "数据点不足（需 ≥ 2），无法拟合。\n"
        print("  数据点不足，跳过拟合。")

    fit_txt_path = f"{OUT_DIR}/scaling_fit.txt"
    with open(fit_txt_path, "w", encoding="utf-8") as f:
        f.write(fit_report)
    print(f"  拟合报告写入 {fit_txt_path}")

    # ─── 出图 ───────────────────────────────────────────────────────────────
    print("\n[4/4] 生成三张图...")
    plot_survival_map(df,  f"{OUT_DIR}/fig1_survival_map.png")
    plot_overdamp_curves(diag, f"{OUT_DIR}/fig2_overdamp_curves.png")
    plot_scaling_law(eta_min_dict, f"{OUT_DIR}/fig3_scaling_law.png", fit_report)

    print("\n分析完成。")
    print(f"  图1: {OUT_DIR}/fig1_survival_map.png")
    print(f"  图2: {OUT_DIR}/fig2_overdamp_curves.png")
    print(f"  图3: {OUT_DIR}/fig3_scaling_law.png")
    print(f"  拟合: {OUT_DIR}/scaling_fit.txt")


if __name__ == "__main__":
    main()
