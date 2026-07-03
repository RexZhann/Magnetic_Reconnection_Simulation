#pragma once
// =============================================================================
// 非对称磁场重联初始条件 (test id = 25)
// =============================================================================
//
// 平衡态（B1 ≠ B2 非对称）：
//   Bx(y)  = 0.5*(B1-B2) + 0.5*(B1+B2)*tanh(y/λ)
//              → B1  (y→+∞，弱场侧)   → -B2  (y→-∞，强场侧)
//   ρ(y)   = 0.5*(ρ1+ρ2) + 0.5*(ρ1-ρ2)*tanh(y/λ)
//   p(y)   = P_total − Bx(y)²/2   （总压守恒）
//
// 矢势（Tóth 2000 §4.1，CT 离散 ∇·B=0 机器精度）：
//   Az(x,y) = Az_equil(y) + ψ₀·cos(kx·x)·cos(ky·(y − y_NL))
//
//   Az_equil = 0.5*(B1-B2)*y + 0.5*(B1+B2)*λ·ln(cosh(y/λ))
//
//   扰动 kick-start：与 test 21 同物理幅值 δBy_max = ψ₀·kx = 0.05
//   中心偏移到中性线 y_NL = λ·arctanh((B2-B1)/(B2+B1))
//
// 参数 (B1=1, B2=2, ρ1=2, ρ2=1, P_total=3, ψ₀=0.2):
//   max|Bx| = B2=2  →  p_min = 3−2 = 1 >> p_floor ✓
//   δBy_max = ψ₀·kx = 0.2*(1/4) = 0.05 = test 21 的 0.05 ✓
//   X 点偏向弱场侧 (y > 0) ✓
// =============================================================================

#include "my_project/types.hpp"
#include <cmath>

namespace my_project {

struct AsymHarrisParams {
    static constexpr double pi = 3.14159265358979323846;

    // 平衡量
    double B1      = 1.0;       // Bx → y→+∞（弱场）
    double B2      = 2.0;       // |Bx| → y→-∞（强场）
    double rho1    = 2.0;       // 密度 → y→+∞
    double rho2    = 1.0;       // 密度 → y→-∞
    double lam     = 0.5;       // 电流片半宽度 λ
    double P_total = 3.0;       // 恒定总压 p + Bx²/2

    // kick-start 扰动：与 test 21 同物理幅值，但中心移到中性线 y_NL
    double psi0    = 0.2;       // 矢势扰动幅值（ψ₀·kx = 0.05 = test 21）

    // 域几何
    double Lx      = 8.0 * pi;  // x ∈ [-4π, 4π]
    double Ly      = 4.0 * pi;  // y ∈ [-2π, 2π]

    // 模式 1 的波数（与域大小匹配，单个 X 点）
    double kx() const { return 2.0 * pi / Lx; }    // = 1/4
    double ky() const { return pi / Ly; }            // = 1/4

    // 中性线位置：Bx_eq(y_NL) = 0 → tanh(y_NL/λ) = (B2-B1)/(B2+B1)
    double y_NL() const { return lam * std::atanh((B2 - B1) / (B2 + B1)); }

    // 平衡场
    double Bx_eq(double y) const {
        return 0.5*(B1 - B2) + 0.5*(B1 + B2) * std::tanh(y / lam);
    }
    double rho_eq(double y) const {
        return 0.5*(rho1 + rho2) + 0.5*(rho1 - rho2) * std::tanh(y / lam);
    }
    double p_eq(double y) const {
        const double bx = Bx_eq(y);
        return P_total - 0.5 * bx * bx;
    }

    // 矢势（含 kick-start 扰动，以 y_NL 为中心）
    //   Bx = ∂Az/∂y，By = -∂Az/∂x
    double Az(double x, double y) const {
        const double yn = y - y_NL();
        return 0.5*(B1 - B2)*y
             + 0.5*(B1 + B2)*lam * std::log(std::cosh(y / lam))
             + psi0 * std::cos(kx()*x) * std::cos(ky()*yn);
    }
};

// 单元格中心原始变量：{ρ, vx, vy, vz, p, Bx, By, Bz, ψ}
Vec asym_cell_ic(double x, double y, const AsymHarrisParams& ap);

// CT 面心磁场（Tóth 2000 §4.1 矢势线积分），初始 ∇·B = 机器精度
double asym_bx_face(double x, double y, double dy, const AsymHarrisParams& ap);
double asym_by_face(double x, double y, double dx, const AsymHarrisParams& ap);

} // namespace my_project
