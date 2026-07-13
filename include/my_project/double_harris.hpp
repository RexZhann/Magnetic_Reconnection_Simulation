#pragma once
// =============================================================================
// CS2008 双周期双 Harris 片构型（test id = 29）—— 对称档 pilot
// =============================================================================
//
// 移植自 Cassak & Shay 2008 Section 3（任务单给出的公式要点）：
//   双片平衡（their Eq. 3-4 的对称档 B01=B02=B0, ρ01=ρ02=ρ0）：
//     Bx(y) = B0 [ tanh((y+Ly/4)/w0) − tanh((y−Ly/4)/w0) − 1 ]
//     片位于 y = ±Ly/4，双向周期域；初始片厚 w0 = 2.0 d_i（照抄，非本项目
//     惯用的 λ=0.5）。
//   矢势（CT 面心初始化用，∇·B = 机器精度）：
//     Az_eq(y) = B0 [ w0 ln cosh((y+Ly/4)/w0) − w0 ln cosh((y−Ly/4)/w0) − y ]
//     净磁通 ∮Bx dy ≈ 0（指数小 ~e^{-Ly/(4 w0)}），周期 wrap 兼容。
//   压强规则（β_min = 4）：
//     P_total = (B0²/2)(1+β_min) = 2.5 B0²；上游 p_min = 2.0。
//   【实现假设，任务未完全指明】均匀温度 T0 = p_min/ρ0 = 2.0，压平衡由密度
//     承担：ρ(y) = (P_total − Bx²/2)/T0（上游 ρ=1，片心 ρ=1.25），
//     与动理学式双片设定一致；p(y) = ρ T0。
//   扰动三件套（幅值照抄 CS2008，统一乘 scale 便于冒烟关断）：
//     1. 相干磁扰动  B1 = −(0.01 B0 Lz/2π) ẑ×∇[sin(2πx/Lx) sin²(2πz/Lz)]
//        ⇔ δAz_coh = +(0.01 B0 Ly/2π) sin(2πx/Lx) sin²(2πy/Ly)
//        （their z → 本代码 y；ẑ×∇ψ 与 ∇×(ψẑ) 的符号换算已含）
//     2. 随机磁扰动 |δB| ~ 5e-5 B0：加在 Az 角点上（δAz_rand = 5e-5·B0·h·hash，
//        h = 格距，面差分后 |δB|~5e-5 量级），确定性位置哈希保可复现 + div-B 精确
//     3. 随机速度扰动 0.08 cA0（cA0 = B0/√ρ0 = 1），cell IC 中同款哈希
// =============================================================================

#include "my_project/types.hpp"
#include <cmath>

namespace my_project {

struct DoubleHarrisParams {
    static constexpr double pi = 3.14159265358979323846;
    double B0     = 1.0;
    double rho0   = 1.0;
    double w0     = 2.0;        // 初始片半厚（CS2008 取 2 d_i）
    double beta_min = 4.0;      // 上游 β 下限规则
    double Lx     = 51.2;
    double Ly     = 25.6;
    double amp_coh = 0.01;      // 相干磁扰动系数（照抄）
    double amp_br  = 5.0e-5;    // 随机磁扰动幅值 / B0
    double amp_vr  = 0.08;      // 随机速度扰动幅值 / cA0
    double scale   = 1.0;       // 三件套统一开关/缩放（冒烟用 ~0）
    double hgrid   = 0.1;       // 格距（随机 Az 幅值换算用），case 29 里填

    double P_total() const { return 0.5 * B0 * B0 * (1.0 + beta_min); }
    double T0()      const { return 0.5 * B0 * B0 * beta_min / rho0; }

    double bx_eq(double y) const {
        return B0 * (std::tanh((y + 0.25 * Ly) / w0)
                   - std::tanh((y - 0.25 * Ly) / w0) - 1.0);
    }
    double az_eq(double y) const {
        return B0 * (w0 * std::log(std::cosh((y + 0.25 * Ly) / w0))
                   - w0 * std::log(std::cosh((y - 0.25 * Ly) / w0)) - y);
    }
    // 确定性位置哈希 ∈ [-0.5, 0.5)（shader 惯用式；纯函数 → 面/角采样一致）
    static double hash(double x, double y, double seed) {
        double s = std::sin(x * 12.9898 + y * 78.233 + seed * 37.719) * 43758.5453;
        return s - std::floor(s) - 0.5;
    }
    double az_coh(double x, double y) const {
        const double s = std::sin(2.0 * pi * y / Ly);
        return amp_coh * B0 * (Ly / (2.0 * pi))
               * std::sin(2.0 * pi * x / Lx) * s * s;
    }
    // 角点总矢势（面心 B 由此差分 → ∇·B 机器精度、双向周期一致）
    double az_total(double x, double y) const {
        return az_eq(y) + scale * (az_coh(x, y)
                                   + amp_br * B0 * hgrid * hash(x, y, 1.0));
    }
};

inline double dh_bx_face(double x, double y, double dy, const DoubleHarrisParams& p) {
    return (p.az_total(x, y + 0.5 * dy) - p.az_total(x, y - 0.5 * dy)) / dy;
}
inline double dh_by_face(double x, double y, double dx, const DoubleHarrisParams& p) {
    return -(p.az_total(x + 0.5 * dx, y) - p.az_total(x - 0.5 * dx, y)) / dx;
}

// 单元格中心原始变量 {ρ, vx, vy, vz, p, Bx, By, Bz, ψ}
// （CT 初始化后会用面平均覆写 Bx/By，这里给近似解析值即可）
inline Vec dh_cell_ic(double x, double y, const DoubleHarrisParams& p) {
    const double bx  = p.bx_eq(y);
    const double rho = (p.P_total() - 0.5 * bx * bx) / p.T0();
    const double pr  = rho * p.T0();
    const double cA0 = p.B0 / std::sqrt(p.rho0);
    const double vx  = p.scale * p.amp_vr * cA0 * DoubleHarrisParams::hash(x, y, 2.0);
    const double vy  = p.scale * p.amp_vr * cA0 * DoubleHarrisParams::hash(x, y, 3.0);
    // 相干扰动的解析 B（cell 近似；面值以 az_total 差分为准）
    constexpr double pi = DoubleHarrisParams::pi;
    const double sy  = std::sin(2.0 * pi * y / p.Ly);
    const double cy  = std::cos(2.0 * pi * y / p.Ly);
    const double dbx = p.scale * p.amp_coh * p.B0 * 2.0 * sy * cy
                       * std::sin(2.0 * pi * x / p.Lx);
    const double dby = -p.scale * p.amp_coh * p.B0 * (p.Ly / p.Lx)
                       * std::cos(2.0 * pi * x / p.Lx) * sy * sy;
    return { rho, vx, vy, 0.0, pr, bx + dbx, dby, 0.0, 0.0 };
}

// test 29 参数：域/格距/扰动开关取自 cfg（psi0 复用为 scale）
inline DoubleHarrisParams dh_params_from_config(const RunConfig& cfg) {
    DoubleHarrisParams p;
    p.Lx    = cfg.x1 - cfg.x0;
    p.Ly    = cfg.y1 - cfg.y0;
    p.hgrid = p.Lx / cfg.nx;
    p.scale = cfg.psi0;
    return p;
}

// =============================================================================
// 广义非对称双片（test 31 campaign 全档）—— CS2008 Eq. (3)-(4) 逐式移植
//
//   Eq. (3)（their z → 本代码 y；01 = 外侧 |y|>Ly/4，02 = 内侧 |y|<Ly/4）：
//     Bx(y) = −B01 tanh((|y|−Ly/4)/w0)   （外侧）
//     Bx(y) = −B02 tanh((|y|−Ly/4)/w0)   （内侧）
//     双片由 z=0 反射构造（their [10]：Bx/ρ/Pi 对 y=0 偶延拓）。
//   Eq. (4)：ρ(y) = ½(ρ01+ρ02) + ½(ρ01−ρ02) tanh((|y|−Ly/4)/w0)
//     —— 密度独立 tanh 剖面、无片心峰；压平衡全部由 Pi 承担：
//     Pi(y) = P_total − Bx²/2，P_total = (Bmax²/2)(1+β_min)，β_min=4，
//     Bmax = max(B01,B02)（各档 P1/P2/T1/T2/β01/β02 与其 Table 1 对照打印）。
//   ⚠ 净通量：B01≠B02 时 ∮Bx dy = (B02−B01)·Ly/2 ≠ 0 —— B 本身双向周期
//     （Bx(±Ly/2) = −B01·tanh(Ly/4/w0) 相等），只是 Az 带线性规范项非周期；
//     CT 面心初始化只用 Az 的局部差分，不受影响。
//   ⚠ 与 case 29 pilot/本机 Sym 档的差异：那两处用的是"均匀温度、密度承担
//     平衡"的实现假设（当时任务未指明、已标注）；campaign 档自本结构起
//     按原文 Eq. (4)。ρ01=ρ02=1 时本剖面密度均匀（无 1.25 片心峰）。
//   扰动三件套（相干 + 随机磁 + 随机速度）与 Sym 档同式同种子（hash 1/2/3）。
// =============================================================================
struct AsymDoubleHarrisParams {
    static constexpr double pi = 3.14159265358979323846;
    double B01 = 1.0, B02 = 1.0, rho01 = 1.0, rho02 = 1.0;
    double beta_min = 4.0;
    double w0     = 2.0;
    double Lx     = 102.4;
    double Ly     = 51.2;
    double amp_coh = 0.01;
    double amp_br  = 5.0e-5;
    double amp_vr  = 0.08;
    double scale   = 1.0;
    double hgrid   = 0.1;

    double Bmax()    const { return std::max(B01, B02); }
    double P_total() const { return 0.5 * Bmax() * Bmax() * (1.0 + beta_min); }

    double bx_eq(double y) const {
        const double zz = std::fabs(y), q = 0.25 * Ly;
        const double B = (zz > q) ? B01 : B02;
        return -B * std::tanh((zz - q) / w0);
    }
    double rho_eq(double y) const {
        const double zz = std::fabs(y), q = 0.25 * Ly;
        return 0.5 * (rho01 + rho02)
             + 0.5 * (rho01 - rho02) * std::tanh((zz - q) / w0);
    }
    double p_eq(double y) const {
        const double b = bx_eq(y);
        return P_total() - 0.5 * b * b;
    }
    // Az：Eq.(3) 的逐段积分，y=0 反射下 Az 为奇函数（Bx 偶 ⇔ Az 奇）。
    double az_half(double zz) const {
        const double q = 0.25 * Ly;
        auto lc = [](double x) { return std::log(std::cosh(x)); };
        if (zz <= q) return -B02 * w0 * (lc((zz - q) / w0) - lc(q / w0));
        return B02 * w0 * lc(q / w0) - B01 * w0 * lc((zz - q) / w0);
    }
    double az_eq(double y) const { return (y >= 0.0) ? az_half(y) : -az_half(-y); }

    // 扰动幅值按 CS2008 [13] 用归一化单位 B0=1，不随档位 Bmax 缩放。
    // （2026-07-13 修复：此前误乘 Bmax()，AB1 的 kick/随机 δB 被放大 2 倍；
    //   AB1 已完成 run 不追溯，caveat 见 kick_fix_record.md。）
    double az_coh(double x, double y) const {
        const double s = std::sin(2.0 * pi * y / Ly);
        return amp_coh * 1.0 * (Ly / (2.0 * pi))
               * std::sin(2.0 * pi * x / Lx) * s * s;
    }
    double az_total(double x, double y) const {
        return az_eq(y) + scale * (az_coh(x, y)
                                   + amp_br * 1.0 * hgrid
                                     * DoubleHarrisParams::hash(x, y, 1.0));
    }
};

inline double adh_bx_face(double x, double y, double dy, const AsymDoubleHarrisParams& p) {
    return (p.az_total(x, y + 0.5 * dy) - p.az_total(x, y - 0.5 * dy)) / dy;
}
inline double adh_by_face(double x, double y, double dx, const AsymDoubleHarrisParams& p) {
    return -(p.az_total(x + 0.5 * dx, y) - p.az_total(x - 0.5 * dx, y)) / dx;
}

inline Vec adh_cell_ic(double x, double y, const AsymDoubleHarrisParams& p) {
    const double rho = p.rho_eq(y);
    const double pr  = p.p_eq(y);
    const double cA0 = 1.0;   // B0=ρ0=1 归一（CS2008 [12]）
    const double vx  = p.scale * p.amp_vr * cA0 * DoubleHarrisParams::hash(x, y, 2.0);
    const double vy  = p.scale * p.amp_vr * cA0 * DoubleHarrisParams::hash(x, y, 3.0);
    constexpr double pi = AsymDoubleHarrisParams::pi;
    const double sy  = std::sin(2.0 * pi * y / p.Ly);
    const double cy  = std::cos(2.0 * pi * y / p.Ly);
    // cell 解析形式与上面 az_coh 同一幅值（0.01·B0，B0=1），同步修复
    const double dbx = p.scale * p.amp_coh * 1.0 * 2.0 * sy * cy
                       * std::sin(2.0 * pi * x / p.Lx);
    const double dby = -p.scale * p.amp_coh * 1.0 * (p.Ly / p.Lx)
                       * std::cos(2.0 * pi * x / p.Lx) * sy * sy;
    return { rho, vx, vy, 0.0, pr, p.bx_eq(y) + dbx, dby, 0.0, 0.0 };
}

inline AsymDoubleHarrisParams adh_params_from_config(const RunConfig& cfg) {
    AsymDoubleHarrisParams p;
    p.B01   = cfg.dh_B01;  p.B02   = cfg.dh_B02;
    p.rho01 = cfg.dh_rho01; p.rho02 = cfg.dh_rho02;
    p.Lx    = cfg.x1 - cfg.x0;
    p.Ly    = cfg.y1 - cfg.y0;
    p.hgrid = p.Lx / cfg.nx;
    p.scale = cfg.psi0;
    return p;
}

} // namespace my_project
