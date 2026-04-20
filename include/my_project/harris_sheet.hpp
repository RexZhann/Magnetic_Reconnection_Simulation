#pragma once
// =============================================================================
// Harris current-sheet initial conditions for 2-D resistive/ideal MHD
// =============================================================================
//
// EQUILIBRIUM
//   Harris, E.G. (1962). "On a plasma sheath separating regions of oppositely
//   directed magnetic field." Il Nuovo Cimento 23 (1), 115–121.
//   doi:10.1007/BF02733547
//
//   The exact kinetic equilibrium is:
//     Bx(y)   = B0 tanh(y/λ)
//     n(y)    = n0 sech²(y/λ) + n_bg            [particle density]
//     T       = B0² / (2 μ0 n0)                  [uniform temperature from ∇p = J×B]
//     p(y)    = (n(y)/n0) × (B0²/2)             [= p_bg + (B0²/2) sech²(y/λ)]
//   which satisfies  ∂(p + Bx²/2)/∂y = 0  exactly.
//
//   For MHD the density is kept uniform (Loureiro et al. 2007, see below),
//   retaining only the pressure equilibrium condition.
//
// DOMAIN / PERTURBATION
//   Birn, J., Drake, J.F., Shay, M.A., et al. (2001). "Geospace Environmental
//   Modeling (GEM) Magnetic Reconnection Challenge." J. Geophys. Res. 106 (A3),
//   3715–3719.  doi:10.1029/1999JA900449
//
//   Domain: Lx = 4π λ, Ly = 2π λ  (fits exactly one tearing mode).
//   Single-mode flux-function perturbation δAz = ψ0 cos(kx x) cos(ky y)
//   seeds one X-point at the origin with max(δBy) = ψ0 kx ≈ 5% B0.
//
// PLASMOID / RECONNECTION RATE CONTEXT
//   Loureiro, N.F., Schekochihin, A.A., Cowley, S.C. (2007). "Instability of
//   current sheets and formation of plasmoid chains." Phys. Plasmas 14 (10),
//   100703.  doi:10.1063/1.2783986
//
//   Uniform density (ρ = ρ_bg), β∞ = 2p_bg/B0² = 0.2, γ = 5/3 follow
//   the convention of this paper and subsequent plasmoid-instability studies.
//
// TEARING-MODE STABILITY CRITERION
//   Furth, H.P., Killeen, J., Rosenbluth, M.N. (1963). "Finite-resistivity
//   instabilities of a sheet pinch." Phys. Fluids 6 (4), 459–484.
//   doi:10.1063/1.1706761
//
//   Modes with kx < 1/λ are tearing-unstable.  With Lx = 4π λ the
//   fundamental mode kx = 1/(2λ) satisfies this condition comfortably.
//
// CT INITIALISATION (∇·B = 0 to machine precision)
//   Stokes / line-integral method: Tóth, G. (2000). "The ∇·B = 0 constraint
//   in shock-capturing magnetohydrodynamics codes." J. Comput. Phys. 161,
//   605–652.  doi:10.1006/jcph.2000.6519
//   Section 4.1, eq. (35)-(36):
//     bx[i][j] = [Az(xi, yj+1) − Az(xi, yj)] / Δy
//     by[i][j] = −[Az(xi+1, yj) − Az(xi, yj)] / Δx
// =============================================================================

#include "my_project/types.hpp"
#include <cmath>

namespace my_project {

// ---------------------------------------------------------------------------
// HarrisSheetParams — all physical parameters in one place
// ---------------------------------------------------------------------------
// Normalisation:  length → λ,  field → B0,  density → ρ_bg,
//                 velocity → vA∞ = B0/√ρ_bg = 1,  time → λ/vA∞
// ---------------------------------------------------------------------------
struct HarrisSheetParams {
    static constexpr double pi = 3.14159265358979323846;

    // ------------------------------------------------------------------
    // Equilibrium  (Harris 1962)
    // ------------------------------------------------------------------
    double B0     = 1.0;   // Asymptotic magnetic field [code B-unit].
                           //   Sets Alfvén speed vA∞ = B0/√ρ_bg = 1.

    double lam    = 0.5;   // Current-sheet half-width λ [code length unit].
                           //   Birn et al. (2001) use λ = 0.5 di (ion inertial
                           //   length).  Here λ/Ly = 1/(4π) ≈ 0.08 so the sheet
                           //   is well-resolved at 512 × 256 cells.

    double rho_bg = 1.0;   // Uniform background density ρ_bg.
                           //   Loureiro et al. (2007) keep ρ = const; the Harris
                           //   kinetic density sech²-profile is not needed for
                           //   ideal/resistive MHD pressure balance.

    double beta_bg = 0.2;  // Background plasma β∞ = 2p_bg / B0².
                           //   Loureiro et al. (2007) use β∞ = 0.2 (p_bg = 0.1).
                           //   Birn et al. (2001) suggest β∞ ≈ 0.5 for kinetic
                           //   runs; 0.2 is more common in MHD reconnection studies.

    // ------------------------------------------------------------------
    // Perturbation  (Birn et al. 2001; Biskamp & Welter 1989)
    // ------------------------------------------------------------------
    double psi0   = 0.1;   // Flux-function perturbation amplitude ψ0.
                           //   The perturbation is derived from:
                           //     δAz = ψ0 cos(kx x) cos(ky y)
                           //   giving  max(δBy) = ψ0 kx = ψ0/(2λ) = 0.1
                           //   (≈ 10 % of B0 — strong enough to seed rapid
                           //   reconnection without disrupting equilibrium).
                           //   Use ψ0 = 0.01 for a gentler linear-phase study.

    // ------------------------------------------------------------------
    // Resistivity  (Sweet & Parker 1958; Loureiro et al. 2007)
    // ------------------------------------------------------------------
    // Sweet, P.A. (1958). "The neutral point theory of solar flares."
    //   IAU Symp. 6, 123.
    // Parker, E.N. (1957). "Sweet's mechanism for merging magnetic fields in
    //   conducting fluids." J. Geophys. Res. 62, 509.
    //
    // Lundquist number S = vA∞ · Lx / η.
    //   S = 2000 (η ≈ 0.006) → Sweet-Parker rate R_SP = S^{-1/2} ≈ 0.022,
    //   barely sufficient to see reconnection in t_end = 20.
    //   S = 500  (η ≈ 0.025) → R_SP ≈ 0.045, visible in a shorter run.
    //   S > 10^4 triggers plasmoid instability (Loureiro et al. 2007).
    //
    // Explicit-diffusion stability (dt ≤ dx²/(2η)):
    //   At 512×256, dx ≈ 0.025 → dt_diff ≈ 0.06 for η = 0.005,
    //   comfortably larger than the Alfvénic CFL dt ≈ 0.008.
    double eta = 0.005;    // Uniform resistivity [code units].
                           //   Gives S ≈ Lx/η ≈ 2500.  Decrease to 0.02–0.05
                           //   for faster (lower-S) reconnection demonstrations.

    // ------------------------------------------------------------------
    // Domain geometry  (Birn et al. 2001)
    // ------------------------------------------------------------------
    double Lx = 4.0 * pi;  // Domain length in x [= 4π λ ≈ 12.57].
                            //   Fits exactly one tearing-mode wavelength:
                            //   kx = 2π/Lx = 1/(2λ) < 1/λ  (FKR unstable).

    double Ly = 2.0 * pi;  // Domain height in y [= 2π λ ≈ 6.28].
                            //   Sufficiently large so the y-boundary does not
                            //   perturb the sheet (sheet half-thickness ≈ 2λ = 1).

    // ------------------------------------------------------------------
    // Derived quantities
    // ------------------------------------------------------------------
    double p_bg()  const { return 0.5 * beta_bg * B0 * B0; }
    double kx()    const { return 2.0 * pi / Lx; }
    double ky()    const { return pi / Ly; }

    // Harris pressure profile  (Harris 1962, eq. 5):
    //   p(y) = p_bg + (B0²/2) sech²(y/λ)
    //   Total pressure p + Bx²/2 = p_bg + B0²/2 = const  ∀ y.
    double p_eq(double y) const {
        const double s = 1.0 / std::cosh(y / lam);
        return p_bg() + 0.5 * B0 * B0 * s * s;
    }

    // Equilibrium Bx  (Harris 1962)
    double Bx_eq(double y) const { return B0 * std::tanh(y / lam); }

    // Full vector potential including perturbation
    // (Tóth 2000, Section 4.1; Birn et al. 2001, eq. 1):
    //   Az(x,y) = B0 λ ln(cosh(y/λ))  +  ψ0 cos(kx x) cos(ky y)
    // Recover B via:   Bx = ∂Az/∂y,   By = −∂Az/∂x
    double Az(double x, double y) const {
        return B0 * lam * std::log(std::cosh(y / lam))
             + psi0 * std::cos(kx() * x) * std::cos(ky() * y);
    }
};

// ---------------------------------------------------------------------------
// Configuration helper — returns a RunConfig for test id = 11
// ---------------------------------------------------------------------------
// Recommended invocation:
//   ./build/mhd2d 11 512 256 2 1   (512×256, CT, HLLD)
//
// Grid spacing at 512×256:  Δx = Δy ≈ 0.025 ≈ λ/20
// Alfvén crossing time τA   = Lx / vA∞ ≈ 12.57
// Simulation end time t_end = 20 τA·(λ/Lx) — adjust for longer runs
// ---------------------------------------------------------------------------
RunConfig make_harris_config(int nx, int ny,
                             DivBCleaningKind divb, SolverKind solver);

// ---------------------------------------------------------------------------
// Cell-centred initial condition
// Returns primitive variables {ρ, vx, vy, vz, p, Bx, By, Bz, ψ} at (x,y).
// ---------------------------------------------------------------------------
Vec harris_cell_ic(double x, double y, const HarrisSheetParams& hp);

// ---------------------------------------------------------------------------
// Exact face-centred B via vector-potential line integral (Tóth 2000 §4.1)
//
//   bx[i][j]  =  [Az(x,  y+Δy/2) − Az(x,  y−Δy/2)] / Δy      (x-face Bx)
//   by[i][j]  = −[Az(x+Δx/2, y) − Az(x−Δx/2, y)] / Δx        (y-face By)
//
// Using these two formulas instead of pointwise sampling guarantees
// ∇·B = 0 to machine precision at initialisation for the CT scheme.
// The proof follows from ∮B·dl = 0 on every cell face pair (Stokes theorem).
// ---------------------------------------------------------------------------
double harris_bx_face(double x, double y, double dy,
                      const HarrisSheetParams& hp);
double harris_by_face(double x, double y, double dx,
                      const HarrisSheetParams& hp);

} // namespace my_project
