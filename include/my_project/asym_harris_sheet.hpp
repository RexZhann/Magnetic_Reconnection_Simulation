#pragma once
// Asymmetric-reconnection initial conditions (test 25).
//
// Equilibrium (B1 ≠ B2):
//   Bx(y) = 0.5(B1−B2) + 0.5(B1+B2) tanh(y/λ)   → B1 (y→+∞), −B2 (y→−∞)
//   ρ(y)  = 0.5(ρ1+ρ2) + 0.5(ρ1−ρ2) tanh(y/λ)
//   p(y)  = P_total − Bx²/2
// Vector potential (Tóth 2000 §4.1, CT machine-precision ∇·B = 0):
//   Az = Az_eq(y) + ψ0 cos(kx x) cos(ky (y − y_NL))
// Kick-start amplitude matches test 21 (δBy_max = ψ0·kx = 0.05), centred on
// the neutral line y_NL = λ·arctanh((B2−B1)/(B2+B1)).

#include "my_project/types.hpp"
#include <algorithm>
#include <cmath>

namespace my_project {

struct AsymHarrisParams {
    static constexpr double pi = 3.14159265358979323846;

    double B1      = 1.0;       // Bx as y→+∞ (weak side)
    double B2      = 2.0;       // |Bx| as y→−∞ (strong side)
    double rho1    = 2.0;       // density as y→+∞
    double rho2    = 1.0;       // density as y→−∞
    double lam     = 0.5;       // current-sheet half-width
    double P_total = 3.0;       // constant total pressure p + Bx²/2

    double psi0    = 0.2;       // perturbation amplitude (ψ0·kx = 0.05)

    double Lx      = 8.0 * pi;
    double Ly      = 4.0 * pi;

    // Mode-1 wavenumbers (single X-point)
    double kx() const { return 2.0 * pi / Lx; }
    double ky() const { return pi / Ly; }

    // Neutral line: Bx_eq(y_NL) = 0
    double y_NL() const { return lam * std::atanh((B2 - B1) / (B2 + B1)); }

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

    // Vector potential incl. kick (Bx = ∂Az/∂y, By = −∂Az/∂x)
    double Az(double x, double y) const {
        const double yn = y - y_NL();
        return 0.5*(B1 - B2)*y
             + 0.5*(B1 + B2)*lam * std::log(std::cosh(y / lam))
             + psi0 * std::cos(kx()*x) * std::cos(ky()*yn);
    }
};

// test 27 asymmetry scan: params from RunConfig (B1 = rho2 = 1 fixed;
// P_total = max(B²)/2 + 1 keeps the strong-side pressure floor at 1).
inline AsymHarrisParams asym_scan_params(const RunConfig& cfg) {
    AsymHarrisParams ap;
    ap.B2      = cfg.asym_B2;
    ap.rho1    = cfg.asym_rho1;
    ap.P_total = 0.5 * std::max(ap.B1 * ap.B1, ap.B2 * ap.B2) + 1.0;
    ap.Lx      = cfg.x1 - cfg.x0;
    ap.Ly      = cfg.y1 - cfg.y0;
    ap.psi0    = cfg.psi0;
    return ap;
}

// Cell-centred primitives {ρ, vx, vy, vz, p, Bx, By, Bz, ψ}
Vec asym_cell_ic(double x, double y, const AsymHarrisParams& ap);

// Face-centred B from Az line integral (machine-precision ∇·B = 0)
double asym_bx_face(double x, double y, double dy, const AsymHarrisParams& ap);
double asym_by_face(double x, double y, double dx, const AsymHarrisParams& ap);

} // namespace my_project
