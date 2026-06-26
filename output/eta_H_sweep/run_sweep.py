"""
run_sweep.py — η_H 稳定性扫描驱动脚本（Hall-MHD Harris sheet, test 21）
===========================================================================
用法（从项目根目录运行）：
    python output/eta_H_sweep/run_sweep.py

功能：
  1. 对三档分辨率（64×32 / 128×64 / 256×128）各扫一系列 η_H 值
  2. 每次调用 build/mhd2d.exe 21 {nx} {ny} 2 1 {eta_H} {t_end} {label}
     输出写入 output/sweep_{label}/（含快照 + divBl2.log）
  3. 从 stdout 解析存活判据、崩溃时间、末期 min_rho、末期 N_sub
  4. 结果汇总写入 output/eta_H_sweep/sweep_results.csv
"""

import subprocess, os, sys, csv, re, time, math

# ─── 可调参数 ───────────────────────────────────────────────────────────────
BUILD    = r"build\mhd2d.exe"   # 相对路径（从项目根目录运行）
OMP_THREADS = "4"

# (nx, ny, t_end)  — 256×128 缩短到 t=8 节省时间
GRIDS = [
    (64,  32,  15.0),
    (128, 64,  15.0),
    (256, 128,  8.0),
]

# 各分辨率的 η_H 扫描序列（从大到小；遇到连续两次崩溃可提前终止）
ETA_H_SEQ = {
    (64,  32):  [3.2e-2, 1.6e-2, 8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5],
    (128, 64):  [8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 3e-4, 1.5e-4, 7.5e-5, 3e-5, 1e-5],
    (256, 128): [2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5],
}

OUT_CSV = "output/eta_H_sweep/sweep_results.csv"
# ────────────────────────────────────────────────────────────────────────────


def eta_label(eta_H: float) -> str:
    """格式化为字符串，用于目录和 CSV（如 1.000e-03）"""
    return f"{eta_H:.3e}"


def parse_stdout(stdout: str, t_end: float):
    """
    从 run_simulation 的 stdout 提取：
      survived     : 仿真是否跑到 t_end
      t_crash      : 崩溃/终止时的 t（-1 表示正常结束）
      t_final      : 最后打印的 t（来自 "Finished: X steps, t=Y"）
      min_rho_last : 最后一次 Step 日志里的 min_rho
      N_sub_last   : 最后一次 Step 日志里的 N_sub（-1 表示无子循环）
      max_v_last   : 最后一次 Step 日志里的 max|v|
      crash_reason : 崩溃原因字符串
    """
    result = {
        "survived": False,
        "t_crash": -1.0,
        "t_final": 0.0,
        "min_rho_last": float("nan"),
        "N_sub_last": -1,
        "max_v_last": float("nan"),
        "crash_reason": "",
    }

    # 最后一条 "Finished: X steps, t=Y" 行
    m = re.search(r"Finished:\s+\d+\s+steps,\s+t=([0-9.e+\-]+)", stdout)
    if m:
        result["t_final"] = float(m.group(1))
        # 容忍 1e-3 的浮点误差
        result["survived"] = result["t_final"] >= t_end - 1e-3

    # FATAL 行（*** FATAL: unphysical state at step N (dt=X)）
    if re.search(r"\*\*\* FATAL", stdout):
        result["crash_reason"] = "FATAL_unphysical"

    # 最后一条 Step 日志
    # 数字模式：可选负号 + 科学计数法/普通浮点
    _NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
    step_lines = re.findall(
        rf"Step\s+\d+\s+t=({_NUM})\s+"
        rf"dt={_NUM}\s+"
        rf"min_rho=({_NUM})\s+"
        rf".*?max\|v\|=({_NUM})"
        rf"(?:.*?N_sub=(\d+))?",
        stdout)
    if step_lines:
        last = step_lines[-1]
        result["t_crash"]      = float(last[0])
        result["min_rho_last"] = float(last[1])
        result["max_v_last"]   = float(last[2])
        result["N_sub_last"]   = int(last[3]) if last[3] else -1

    # 若未到 t_end 且没有 FATAL，视为早退（可能是 dt<1e-8 触发的 break）
    if not result["survived"] and not result["crash_reason"]:
        if result["t_final"] < t_end - 0.5:
            result["crash_reason"] = "early_exit"

    # 若 min_rho 触底（= rho_floor=0.02）且仿真未存活
    if (not result["survived"] and
            not math.isnan(result["min_rho_last"]) and
            result["min_rho_last"] <= 0.021):
        if not result["crash_reason"]:
            result["crash_reason"] = "floor_hit"

    return result


def run_one(nx: int, ny: int, eta_H: float, t_end: float) -> dict:
    """执行单次仿真，返回解析结果字典"""
    label = f"sweep_etaH{eta_label(eta_H)}_{nx}x{ny}"
    cmd = [BUILD, "21", str(nx), str(ny), "2", "1",
           f"{eta_H:.6e}", f"{t_end:.4f}", label]

    print(f"  运行: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    wall = time.time() - t0

    parsed = parse_stdout(proc.stdout + proc.stderr, t_end)
    parsed["label"]   = label
    parsed["nx"]      = nx
    parsed["ny"]      = ny
    parsed["eta_H"]   = eta_H
    parsed["t_end"]   = t_end
    parsed["wall_s"]  = round(wall, 1)
    parsed["exit_code"] = proc.returncode

    # 把完整日志存到输出目录
    log_path = f"output/{label}/run.log"
    if os.path.isdir(f"output/{label}"):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n" + proc.stdout)
            f.write("\n=== STDERR ===\n" + proc.stderr)

    status = "存活" if parsed["survived"] else f"崩溃({parsed['crash_reason']})"
    print(f"    → {status}, t_final={parsed['t_final']:.3f}, "
          f"min_rho={parsed['min_rho_last']:.4f}, "
          f"N_sub={parsed['N_sub_last']}, 耗时={wall:.0f}s",
          flush=True)
    return parsed


def run_sweep():
    os.makedirs("output/eta_H_sweep", exist_ok=True)

    all_rows = []
    fieldnames = ["nx", "ny", "dx", "eta_H", "anchor_etaH", "t_end",
                  "survived", "t_final", "t_crash", "min_rho_last",
                  "max_v_last", "N_sub_last", "crash_reason",
                  "wall_s", "exit_code", "label"]

    for nx, ny, t_end in GRIDS:
        pi = math.pi
        Lx = 4 * pi  # x 方向域长 [= 4π λ]（harris_sheet.hpp: Lx=4π, Ly=2π）
        dx = Lx / nx  # 正方形网格：dx = dy = 4π/nx（nx/ny=2:1 保持方格）
        anchor = dx * dx / (pi * pi)  # η_H^crit = Δx²/π²（哨声波稳定锚点）

        print(f"\n{'='*60}")
        print(f"分辨率 {nx}×{ny}  Δx={dx:.4f}  锚点η_H={anchor:.3e}  t_end={t_end}")
        print(f"{'='*60}")

        consecutive_crashes = 0
        for eta_H in ETA_H_SEQ.get((nx, ny), []):
            row = run_one(nx, ny, eta_H, t_end)
            row["dx"] = dx
            row["anchor_etaH"] = anchor
            all_rows.append(row)

            if row["survived"]:
                consecutive_crashes = 0
            else:
                consecutive_crashes += 1
                if consecutive_crashes >= 2:
                    print(f"  连续两次崩溃，停止 {nx}×{ny} 扫描。")
                    break

        # 写中间结果（每档分辨率后刷新 CSV）
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"  已更新 {OUT_CSV}")

    print(f"\n扫描完成！结果保存在 {OUT_CSV}")
    print(f"共 {len(all_rows)} 次运行。")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = OMP_THREADS
    run_sweep()
