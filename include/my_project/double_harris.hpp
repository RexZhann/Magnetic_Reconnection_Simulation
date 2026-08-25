#pragma once
// CS2008 doubly-periodic double Harris sheet (test 29, symmetric pilot).
// Ported from Cassak & Shay 2008 §3 (their z → this code's y):
//   Bx(y) = B0 [tanh((y+Ly/4)/w0) − tanh((y−Ly/4)/w0) − 1], sheets at ±Ly/4,
//   w0 = 2 d_i; Az_eq from its integral (CT init, machine-precision ∇·B = 0).
//   Pressure rule β_min = 4: P_total = 2.5 B0². Implementation assumption
//   (not fixed by the source): uniform T0 = 2, density carries the balance:
//   ρ(y) = (P_total − Bx²/2)/T0, p = ρT0.
// Perturbation trio (CS2008 amplitudes, all multiplied by `scale`):
//   coherent δAz = (0.01 B0 Ly/2π) sin(2πx/Lx) sin²(2πy/Ly);
//   random δB ~ 5e-5 B0 via deterministic corner hash on Az (reproducible,
//   div-B exact); random velocity 0.08 cA0.

#include "my_project/types.hpp"
#include <cmath>

namespace my_project {

struct DoubleHarrisParams {
    static constexpr double pi = 3.14159265358979323846;
    double B0     = 1.0;
    double rho0   = 1.0;
    double w0     = 2.0;        // initial sheet half-thickness (CS2008: 2 d_i)
    double beta_min = 4.0;      // upstream beta floor rule
    double Lx     = 51.2;
    double Ly     = 25.6;
    double amp_coh = 0.01;      // coherent perturbation coefficient
    double amp_br  = 5.0e-5;    // random B amplitude / B0
    double amp_vr  = 0.08;      // random velocity amplitude / cA0
    double scale   = 1.0;       // master switch/scale for the trio (~0 = off)
    double hgrid   = 0.1;       // grid spacing (random-Az amplitude conversion)

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
    // Deterministic positional hash ∈ [−0.5, 0.5): pure function, so face and
    // corner sampling agree.
    static double hash(double x, double y, double seed) {
        double s = std::sin(x * 12.9898 + y * 78.233 + seed * 37.719) * 43758.5453;
        return s - std::floor(s) - 0.5;
    }
    double az_coh(double x, double y) const {
        const double s = std::sin(2.0 * pi * y / Ly);
        return amp_coh * B0 * (Ly / (2.0 * pi))
               * std::sin(2.0 * pi * x / Lx) * s * s;
    }
    // Total corner Az; face B is differenced from this.
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

// Cell-centred primitives {ρ, vx, vy, vz, p, Bx, By, Bz, ψ}.
// CT init overwrites Bx/By with face averages; analytic values suffice here.
inline Vec dh_cell_ic(double x, double y, const DoubleHarrisParams& p) {
    const double bx  = p.bx_eq(y);
    const double rho = (p.P_total() - 0.5 * bx * bx) / p.T0();
    const double pr  = rho * p.T0();
    const double cA0 = p.B0 / std::sqrt(p.rho0);
    const double vx  = p.scale * p.amp_vr * cA0 * DoubleHarrisParams::hash(x, y, 2.0);
    const double vy  = p.scale * p.amp_vr * cA0 * DoubleHarrisParams::hash(x, y, 3.0);
    // Analytic coherent δB (cell approximation; faces use az_total differences)
    constexpr double pi = DoubleHarrisParams::pi;
    const double sy  = std::sin(2.0 * pi * y / p.Ly);
    const double cy  = std::cos(2.0 * pi * y / p.Ly);
    const double dbx = p.scale * p.amp_coh * p.B0 * 2.0 * sy * cy
                       * std::sin(2.0 * pi * x / p.Lx);
    const double dby = -p.scale * p.amp_coh * p.B0 * (p.Ly / p.Lx)
                       * std::cos(2.0 * pi * x / p.Lx) * sy * sy;
    return { rho, vx, vy, 0.0, pr, bx + dbx, dby, 0.0, 0.0 };
}

// test 29 params from cfg (psi0 reused as scale)
inline DoubleHarrisParams dh_params_from_config(const RunConfig& cfg) {
    DoubleHarrisParams p;
    p.Lx    = cfg.x1 - cfg.x0;
    p.Ly    = cfg.y1 - cfg.y0;
    p.hgrid = p.Lx / cfg.nx;
    p.scale = cfg.psi0;
    return p;
}

// ============================================================================
// Generalised asymmetric double sheet (test 31 campaign) — CS2008 Eq. (3)-(4).
//   Eq. (3): Bx = −B01 tanh((|y|−Ly/4)/w0) outside |y|>Ly/4, −B02 tanh inside;
//   sheets built by even reflection about y=0.
//   Eq. (4): ρ(y) = ½(ρ01+ρ02) + ½(ρ01−ρ02) tanh((|y|−Ly/4)/w0); pressure
//   carries the balance: Pi = P_total − Bx²/2, P_total = (Bmax²/2)(1+β_min).
// Net flux: for B01 ≠ B02, ∮Bx dy ≠ 0 — B itself is periodic; only Az gains
//   a linear gauge term, which the local CT differences never see.
// Note: differs from the case-29 pilot, which used the uniform-T assumption;
//   campaign tiers follow Eq. (4) exactly (ρ01=ρ02=1 → uniform density).
// Perturbation trio identical to the symmetric case (hash seeds 1/2/3).
// ============================================================================
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
    // Piecewise integral of Eq. (3); Az is odd under the y=0 reflection.
    double az_half(double zz) const {
        const double q = 0.25 * Ly;
        auto lc = [](double x) { return std::log(std::cosh(x)); };
        if (zz <= q) return -B02 * w0 * (lc((zz - q) / w0) - lc(q / w0));
        return B02 * w0 * lc(q / w0) - B01 * w0 * lc((zz - q) / w0);
    }
    double az_eq(double y) const { return (y >= 0.0) ? az_half(y) : -az_half(-y); }

    // Perturbation amplitudes use normalised B0 = 1 (CS2008 [13]); they do
    // NOT scale with the tier's Bmax.
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
    const double cA0 = 1.0;   // B0 = rho0 = 1 normalisation
    const double vx  = p.scale * p.amp_vr * cA0 * DoubleHarrisParams::hash(x, y, 2.0);
    const double vy  = p.scale * p.amp_vr * cA0 * DoubleHarrisParams::hash(x, y, 3.0);
    constexpr double pi = AsymDoubleHarrisParams::pi;
    const double sy  = std::sin(2.0 * pi * y / p.Ly);
    const double cy  = std::cos(2.0 * pi * y / p.Ly);
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
