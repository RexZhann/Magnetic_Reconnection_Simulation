"""
compare_21_23.py -- test 21（超电阻+亚循环）vs test 23（Hall-HLL）重联动力学对比
==================================================================================
运行方式：python output/compare_21_23.py
输出：output/compare_21_23.png
"""

import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()


def load_snapshot(fname):
    with open(fname) as f:
        h = f.readline().split()
    nx, ny = int(h[0]), int(h[1])
    t = float(h[4]) if len(h) >= 5 else 0.
    data = np.loadtxt(fname, skiprows=1)
    d = {"t": t, "nx": nx, "ny": ny}
    for k, key in enumerate(["x","y","rho","vx","vy","vz","p","Bx","By","Bz","psi","e","divB"]):
        d[key] = data[:, k].reshape(ny, nx)
    return d


def load_series(pattern):
    files = sorted(glob.glob(pattern))
    return [load_snapshot(f) for f in files]


def reconnected_flux(snaps):
    phi = []
    for s in snaps:
        jmid = s["ny"] // 2
        x = s["x"][jmid, :]; By = s["By"][jmid, :]
        xmid = 0.5 * (x[0] + x[-1])
        phi.append(np.trapz(np.abs(By[x <= xmid]), x[x <= xmid]))
    return np.array(phi)


def compute_jz(d):
    dx = d["x"][0, 1] - d["x"][0, 0]
    dy = d["y"][1, 0] - d["y"][0, 0]
    return np.gradient(d["By"], dx, axis=1) - np.gradient(d["Bx"], dy, axis=0)


snaps21 = load_series(os.path.join(_HERE, "test21/test21_128x64_hlld_ct_snap*.dat"))
snaps23 = load_series(os.path.join(_HERE, "test23/test23_128x64_hlld_ct_snap*.dat"))

t21 = np.array([s["t"] for s in snaps21])
t23 = np.array([s["t"] for s in snaps23])
phi21 = reconnected_flux(snaps21)
phi23 = reconnected_flux(snaps23)
bz21  = np.array([np.max(np.abs(s["Bz"])) for s in snaps21])
bz23  = np.array([np.max(np.abs(s["Bz"])) for s in snaps23])
vy21  = np.array([np.max(np.abs(s["vy"])) for s in snaps21])
vy23  = np.array([np.max(np.abs(s["vy"])) for s in snaps23])
jz21  = np.array([np.max(compute_jz(s)) for s in snaps21])
jz23  = np.array([np.max(compute_jz(s)) for s in snaps23])

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    r"Test 21（超电阻+亚循环，$\eta_H=10^{-3}$）vs Test 23（Hall-HLL）动力学对比"
    "\n$d_i=1$，Harris 电流片，$128\times64$，$t_{end}=15$",
    fontsize=12, fontweight="bold")

kw = dict(lw=2.0)

ax = axes[0, 0]
ax.plot(t21, phi21, "C0-", label=r"Test 21：超电阻+亚循环", **kw)
ax.plot(t23, phi23, "C1-", label=r"Test 23：Hall-HLL 稳定化", **kw)
ax.set_xlabel("t"); ax.set_ylabel(r"$\Phi = \int_0^{L_x/2}|B_y|\,dx$（$y=0$）")
ax.set_title("重联磁通量")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t21, vy21, "C0-", label="Test 21", **kw)
ax.plot(t23, vy23, "C1-", label="Test 23", **kw)
ax.set_xlabel("t"); ax.set_ylabel(r"$\max|v_y|$")
ax.set_title("流入速度（重联率代理量）")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(t21, jz21, "C0-", label="Test 21", **kw)
ax.plot(t23, jz23, "C1-", label="Test 23", **kw)
ax.set_xlabel("t"); ax.set_ylabel(r"$\max J_z$")
ax.set_title("峰值电流密度")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(t21, bz21, "C0-", label="Test 21", **kw)
ax.plot(t23, bz23, "C1-", label="Test 23", **kw)
ax.set_xlabel("t"); ax.set_ylabel(r"$\max|B_z|$")
ax.set_title(r"Hall 四极 $B_z$ 幅值")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

for ax in axes.flat:
    ax.set_xlim(0, 15)

plt.tight_layout()
out = os.path.join(_HERE, "compare_21_23.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"已保存：{out}")
