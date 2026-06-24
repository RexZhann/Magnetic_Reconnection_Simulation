"""
debug_hll_checkerboard.py
=========================
诊断 test 23 t=5 时刻棋盘振荡的根本原因。

步骤 1：读 t=5 快照，计算每角点 c_w = π·di·|B|_corner / (ρ_corner·mincell)
         绘制 c_w 分布图，标出 c_w→0 的区域。

步骤 2：棋盘衰减单元测试。
         构造 2D 静态背景场 + grid-scale 棋盘扰动 δBy[i][J]=ε·(-1)^i
         按离散 Faraday 公式计算一步 HLL 的 ΔBy；
         分别在 B_bg=1（正常区）和 B_bg=0.01（磁零点）下测试：
           - 若 |ΔBy/By| > 0 → 放大（不稳定）
           - 若 |ΔBy/By| < 0 → 衰减（正常阻尼）
         打印 amplification_factor = (by_new - by_bg) / epsilon

步骤 3：在 t=5 图上叠加 c_w 等值线，确认低 c_w 区域与棋盘重合。
"""

import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# ─── 参数 ──────────────────────────────────────────────────────────────────
DI       = 1.0          # ion inertial length for test 23
RHO_FL   = 0.1          # same floor as C++ code
LX, LY   = 4*np.pi, 2*np.pi
NX, NY   = 128, 64
DX = LX / NX           # ≈ 0.09817
DY = LY / NY           # ≈ 0.09817
MINCELL  = min(DX, DY)
PI       = np.pi

# ─── 加载 t=5 快照 ─────────────────────────────────────────────────────────
def load_snapshot(fname):
    with open(fname) as f:
        h = f.readline().split()
    nx, ny = int(h[0]), int(h[1])
    t = float(h[4]) if len(h) >= 5 else 0.0
    data = np.loadtxt(fname, skiprows=1)
    d = {"t": t, "nx": nx, "ny": ny}
    for k, key in enumerate(["x","y","rho","vx","vy","vz","p","Bx","By","Bz","psi","e","divB"]):
        d[key] = data[:, k].reshape(ny, nx)
    return d

pattern = os.path.join(_HERE, "test23/test23_128x64_hlld_ct_snap*.dat")
files = sorted(glob.glob(pattern))
if not files:
    raise FileNotFoundError(f"No test23 snapshots found at {pattern}")

# 找 t≈5 的快照
snaps = [load_snapshot(f) for f in files]
t_arr = np.array([s["t"] for s in snaps])
idx5  = np.argmin(np.abs(t_arr - 5.0))
s5    = snaps[idx5]
print(f"Using snapshot at t={s5['t']:.2f}")

Bx = s5["Bx"]   # shape (ny, nx)
By = s5["By"]
Bz = s5["Bz"]
rho = s5["rho"]
x2d = s5["x"]
y2d = s5["y"]

# ─── 步骤 1：计算角点 c_w ────────────────────────────────────────────────────
# 角点 (I,J) 平均周围四个格点：(I-1,J-1),(I,J-1),(I-1,J),(I,J) (0-indexed interior)
# 与 C++ 的 w[I+1][J+1]~w[I+2][J+2] 对应（去除 2 层 ghost 后即 [I][J]~[I+1][J+1]）

# cell-center arrays: rho[j][i], Bx[j][i], By[j][i], Bz[j][i]
# i=0..NX-1, j=0..NY-1

# 角点 c_w 数组：shape (NX+1, NY+1)
cw = np.zeros((NX+1, NY+1))
for I in range(NX+1):
    for J in range(NY+1):
        # 四邻近格点索引（周期 x，transmissive y → clamp）
        iL = (I-1) % NX     # left cell
        iR = I % NX         # right cell (periodic in x)
        jD = max(J-1, 0)    # lower cell
        jU = min(J, NY-1)   # upper cell

        rho_c = 0.25*(rho[jD,iL] + rho[jD,iR] + rho[jU,iL] + rho[jU,iR])
        Bx_c  = 0.25*(Bx[jD,iL]  + Bx[jD,iR]  + Bx[jU,iL]  + Bx[jU,iR])
        By_c  = 0.25*(By[jD,iL]  + By[jD,iR]  + By[jU,iL]  + By[jU,iR])
        Bz_c  = 0.25*(Bz[jD,iL]  + Bz[jD,iR]  + Bz[jU,iL]  + Bz[jU,iR])
        rho_c = max(rho_c, RHO_FL)
        B2_c  = Bx_c**2 + By_c**2 + Bz_c**2
        cw[I,J] = PI * DI * np.sqrt(B2_c) / (rho_c * MINCELL)

print(f"\n[步骤 1] c_w 统计")
print(f"  min  c_w = {cw.min():.4f}  at corner I={np.argmin(cw)//( NY+1)}, J={np.argmin(cw)%(NY+1)}")
print(f"  max  c_w = {cw.max():.4f}")
print(f"  mean c_w = {cw.mean():.4f}")
print(f"  c_w < 1  : {(cw < 1).sum()} corners ({100*(cw<1).sum()/(NX+1)/(NY+1):.1f}%)")
print(f"  c_w < 5  : {(cw < 5).sum()} corners ({100*(cw<5).sum()/(NX+1)/(NY+1):.1f}%)")
print(f"  c_w < 10 : {(cw < 10).sum()} corners ({100*(cw<10).sum()/(NX+1)/(NY+1):.1f}%)")

# ─── 步骤 2：棋盘衰减单元测试 ───────────────────────────────────────────────
print(f"\n[步骤 2] 棋盘衰减单元测试")
print(f"  参数: DI={DI}, NX={NX}, DX={DX:.5f}, MINCELL={MINCELL:.5f}, CFL=0.3")

# 构造 nx=8 的 1D 测试（足够演示原理）
def hll_one_step(by_init, bx_init, B_bg, rho_bg, dt):
    """
    对 1D 周期域上 nx=8 的面心 By 施加一步 HLL 稳定化，
    使用均匀 c_w（由 B_bg 和 rho_bg 决定）。
    返回新 By。
    """
    nx = len(by_init)
    c_w = PI * DI * B_bg / (rho_bg * MINCELL)

    # EMF at corners (I=0..nx), periodic
    emf_z = np.zeros(nx+1)
    for I in range(nx+1):
        i_right = I % nx        # by[i_right]
        i_left  = (I-1) % nx   # by[i_left]
        jump_by = by_init[i_right] - by_init[i_left]
        emf_z[I] = (c_w * 0.5) * jump_by
        # no Bx term in 1D

    # Faraday update
    by_new = by_init.copy()
    for i in range(nx):
        by_new[i] += (dt / DX) * (emf_z[i+1] - emf_z[i])
    return by_new, c_w

# 时间步：使用与 test23 早期相近的 dt
dt_test = 1.77e-4

epsilon = 1e-3

for B_bg_val, label in [(1.0, "B_bg=1.0 (normal region)"),
                         (0.10, "B_bg=0.10 (reconnection X-point)"),
                         (0.01, "B_bg=0.01 (near magnetic null)")]:
    rho_bg_val = 0.2  # background density

    # 构造棋盘：by[i] = B_bg + epsilon*(-1)^i
    nx_test = 32
    i_arr   = np.arange(nx_test)
    by_init = B_bg_val + epsilon * (-1)**i_arr
    bx_init = np.zeros(nx_test)

    by_new, c_w_val = hll_one_step(by_init, bx_init, B_bg_val, rho_bg_val, dt_test)

    # 棋盘振幅（奇偶差）
    amp_before = np.std(by_init)
    amp_after  = np.std(by_new)
    growth = (amp_after - amp_before) / amp_before

    nu = c_w_val * dt_test / MINCELL  # HLL 扩散数（应 ≤ CFL）
    print(f"  {label}")
    print(f"    c_w      = {c_w_val:.4f}")
    print(f"    nu=c_w·dt/h = {nu:.4f}")
    print(f"    std(by) before = {amp_before:.6e}")
    print(f"    std(by) after  = {amp_after:.6e}")
    print(f"    growth/step    = {growth:+.4f}  ({'DAMPED' if growth < 0 else 'AMPLIFIED!'})")
    print()

# ─── 步骤 3：绘制 c_w 分布 + Jz，找出低 c_w 区与棋盘重叠 ───────────────────
def compute_jz(d):
    dx = d["x"][0,1]-d["x"][0,0]
    dy = d["y"][1,0]-d["y"][0,0]
    return np.gradient(d["By"], dx, axis=1) - np.gradient(d["Bx"], dy, axis=0)

Jz5   = compute_jz(s5)
Jz_lim = np.percentile(np.abs(Jz5), 99)

# 角点坐标（以角点中心为准）
x_corner = np.linspace(x2d[0,0] - DX/2, x2d[0,-1] + DX/2, NX+1)
y_corner = np.linspace(y2d[0,0] - DY/2, y2d[-1,0] + DY/2, NY+1)
Xc, Yc = np.meshgrid(x_corner, y_corner, indexing='ij')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"HLL Checkerboard Diagnostics (test 23, t={s5['t']:.1f})", fontsize=13, fontweight="bold")

# 左图：Jz（棋盘可见）
ax = axes[0]
im = ax.pcolormesh(x2d, y2d, Jz5, shading="auto", cmap="RdBu_r", vmin=-Jz_lim, vmax=Jz_lim)
ax.set_title(r"$J_z = \partial B_y/\partial x - \partial B_x/\partial y$")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, label="$J_z$")

# 中图：|B|（背景场强度）
B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
ax = axes[1]
im = ax.pcolormesh(x2d, y2d, B_mag, shading="auto", cmap="plasma")
ax.set_title(r"$|B|$ (background field magnitude)")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, label="$|B|$")

# 右图：c_w 分布（低 c_w → 红色预警）
ax = axes[2]
cw_T = cw.T  # shape (NY+1, NX+1) for pcolormesh
im = ax.pcolormesh(Xc.T, Yc.T, np.log10(cw_T + 1e-3), shading="auto", cmap="YlOrRd_r")
# 叠加 c_w=10 等值线
cs = ax.contour(Xc.T, Yc.T, cw_T, levels=[5, 10, 20], colors=["k","b","g"], linewidths=0.8)
ax.clabel(cs, fmt="c_w=%.0f", fontsize=8)
ax.set_title(r"$\log_{10}(c_w)$  (low = weak HLL damping)")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, label=r"$\log_{10}(c_w)$")

plt.tight_layout()
out = os.path.join(_HERE, "hll_checkerboard_diag.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[步骤 3] 诊断图已保存: {out}")

# 打印低 c_w 区域的物理坐标
threshold = 5.0
low_mask = cw < threshold  # shape (NX+1, NY+1)
if low_mask.any():
    low_I, low_J = np.where(low_mask)
    low_x = x_corner[low_I]
    low_y = y_corner[low_J]
    # 只打印 y≈0 中平面附近（|y|<1）
    midplane = np.abs(low_y) < 1.0
    if midplane.any():
        print(f"\n  c_w < {threshold} 且 |y|<1 的角点 (共 {midplane.sum()} 个):")
        # 按 x 排序
        order = np.argsort(low_x[midplane])
        xs = low_x[midplane][order]
        ys = low_y[midplane][order]
        cws = cw[low_I[midplane], low_J[midplane]][order]
        for xi, yi, cwi in zip(xs[::max(1,len(xs)//20)],
                                ys[::max(1,len(ys)//20)],
                                cws[::max(1,len(cws)//20)]):
            print(f"    x={xi:.2f}, y={yi:.2f}, c_w={cwi:.3f}")
else:
    print(f"\n  全域 c_w >= {threshold}，无低 c_w 区。")
