#pragma once
// Harris current-sheet initial conditions for 2-D resistive/ideal MHD.
//
// References:
//   Harris 1962, Il Nuovo Cimento 23, 115 — equilibrium
//     Bx = B0 tanh(y/λ), p = p_bg + (B0²/2) sech²(y/λ) (total pressure const).
//   Birn et al. 2001 (GEM), JGR 106, 3715 — domain Lx=4πλ, Ly=2πλ and
//     perturbation δAz = ψ0 cos(kx x) cos(ky y) (one X-point).
//   Loureiro et al. 2007, Phys. Plasmas 14, 100703 — uniform-ρ MHD
//     convention, β∞ = 0.2.
//   Furth/Killeen/Rosenbluth 1963 — tearing instability for kx < 1/λ
//     (fundamental mode kx = 1/(2λ) qualifies).
//   Tóth 2000, JCP 161, 605 §4.1 — CT init via Az line integral,
//     machine-precision ∇·B = 0.

#include "my_project/types.hpp"
#include <cmath>

namespace my_project {

// All physical parameters in one place.
// Normalisation: length → λ, field → B0, density → ρ_bg,
//                velocity → vA∞ = B0/√ρ_bg = 1, time → λ/vA∞.
struct HarrisSheetParams {
    static constexpr double pi = 3.14159265358979323846;

    // Equilibrium (Harris 1962)
    double B0     = 1.0;   // asymptotic field; sets vA∞ = 1
    double lam    = 0.5;   // current-sheet half-width λ
    double rho_bg = 1.0;   // uniform background density (Loureiro 2007)
    double beta_bg = 0.2;  // background β∞ = 2p_bg/B0²

    // Perturbation (Birn et al. 2001): max(δBy) = ψ0·kx = ψ0/(2λ) = 0.1
    double psi0   = 0.1;

    // Uniform resistivity: S = Lx/η ≈ 2500 (Sweet-Parker R_SP = S^{-1/2};
    // S > 1e4 triggers plasmoid instability).
    double eta = 0.005;

    // Domain (Birn et al. 2001): exactly one tearing-mode wavelength.
    double Lx = 4.0 * pi;
    double Ly = 2.0 * pi;

    // Derived
    double p_bg()  const { return 0.5 * beta_bg * B0 * B0; }
    double kx()    const { return 2.0 * pi / Lx; }
    double ky()    const { return pi / Ly; }

    // Harris pressure balance: p + Bx²/2 = const
    double p_eq(double y) const {
        const double s = 1.0 / std::cosh(y / lam);
        return p_bg() + 0.5 * B0 * B0 * s * s;
    }

    double Bx_eq(double y) const { return B0 * std::tanh(y / lam); }

    // Full vector potential (Bx = ∂Az/∂y, By = −∂Az/∂x)
    double Az(double x, double y) const {
        return B0 * lam * std::log(std::cosh(y / lam))
             + psi0 * std::cos(kx() * x) * std::cos(ky() * y);
    }
};

// RunConfig factory for test 11. Typical: ./build/mhd2d 11 512 256 2 1
RunConfig make_harris_config(int nx, int ny,
                             DivBCleaningKind divb, SolverKind solver);

// Cell-centred primitives {ρ, vx, vy, vz, p, Bx, By, Bz, ψ}
Vec harris_cell_ic(double x, double y, const HarrisSheetParams& hp);

// Face-centred B from the Az finite difference across one cell (Tóth 2000
// §4.1); guarantees ∇·B = 0 to machine precision at initialisation.
double harris_bx_face(double x, double y, double dy,
                      const HarrisSheetParams& hp);
double harris_by_face(double x, double y, double dx,
                      const HarrisSheetParams& hp);

} // namespace my_project
