// =============================================================================
// Harris current-sheet initial conditions  (test id = 11)
// =============================================================================
//
// See include/my_project/harris_sheet.hpp for the full literature references
// and derivation of every parameter.
//
// Quick-reference key papers:
//   [H62]  Harris (1962)        — kinetic equilibrium (Bx, p profiles)
//   [B01]  Birn et al. (2001)   — GEM domain, perturbation form
//   [L07]  Loureiro et al. (2007) — uniform-ρ MHD convention, β∞ = 0.2
//   [T00]  Tóth (2000)          — CT vector-potential initialisation
// =============================================================================

#include "my_project/harris_sheet.hpp"

namespace my_project {

// ---------------------------------------------------------------------------
// RunConfig factory
// ---------------------------------------------------------------------------
RunConfig make_harris_config(int nx, int ny,
                             DivBCleaningKind divb, SolverKind solver) {
    const HarrisSheetParams hp;
    RunConfig cfg;
    cfg.test   = 11;
    cfg.nx     = nx;
    cfg.ny     = ny;
    cfg.gamma  = 5.0 / 3.0;     // standard for space / laboratory plasmas [L07]
    cfg.t_end  = 20.0;           // ≈ 1.6 τA∞ (Alfvén crossing times based on Lx)
    cfg.cfl    = (solver == SolverKind::FORCE) ? 0.4 : 0.3;
    cfg.solver = solver;
    cfg.divb   = divb;
    cfg.x0     = -0.5 * hp.Lx;  // x ∈ [−2π, 2π]   [B01]
    cfg.x1     =  0.5 * hp.Lx;
    cfg.y0     = -0.5 * hp.Ly;  // y ∈ [−π, π]      [B01]
    cfg.y1     =  0.5 * hp.Ly;
    cfg.bcx    = BC::Periodic;       // x-periodic: reconnected flux can wrap around [B01]
    cfg.bcy    = BC::Transmissive;   // y-open: plasma can freely escape the sheet
    cfg.eta    = hp.eta;             // uniform resistivity η (Sweet & Parker 1958)
    return cfg;
}

// ---------------------------------------------------------------------------
// Cell-centred IC
// ---------------------------------------------------------------------------
// Primitive state vector: {ρ, vx, vy, vz, p, Bx, By, Bz, ψ}
//
// Equilibrium component [H62]:
//   Bx_eq(y) = B0 tanh(y/λ)
//   p_eq(y)  = p_bg + (B0²/2) sech²(y/λ)       [total pressure = const]
//
// Perturbation — single X-point from flux function [B01]:
//   δAz      = ψ0 cos(kx x) cos(ky y)
//   δBx      =  ∂δAz/∂y  = −ψ0 ky cos(kx x) sin(ky y)
//   δBy      = −∂δAz/∂x  =  ψ0 kx sin(kx x) cos(ky y)
//
// Together:  ∇·(Beq + δB) = 0 analytically.
// The CT face-B initialisation (harris_bx_face / harris_by_face) additionally
// enforces this to machine precision on the discrete staggered grid [T00].
// ---------------------------------------------------------------------------
Vec harris_cell_ic(double x, double y, const HarrisSheetParams& hp) {
    // Perturbed field components (analytic derivatives of Az)
    const double Bx = hp.Bx_eq(y)
                    - hp.psi0 * hp.ky() * std::cos(hp.kx() * x) * std::sin(hp.ky() * y);
    const double By =   hp.psi0 * hp.kx() * std::sin(hp.kx() * x) * std::cos(hp.ky() * y);

    return {
        hp.rho_bg,    // [0]  ρ  — uniform [L07]; Harris kinetic profile not used in MHD
        0.0,          // [1]  vx — no bulk flow
        0.0,          // [2]  vy
        0.0,          // [3]  vz
        hp.p_eq(y),   // [4]  p  — Harris pressure balance [H62]
        Bx,           // [5]  Bx — equilibrium + perturbation
        By,           // [6]  By — perturbation only (By_eq = 0)
        0.0,          // [7]  Bz — no guide field
        0.0           // [8]  ψ  — GLM divergence-cleaning scalar (zero at t=0)
    };
}

// ---------------------------------------------------------------------------
// Exact face-centred B via vector-potential line integral  [T00 §4.1]
// ---------------------------------------------------------------------------
//
// The vector potential of the full (equilibrium + perturbation) field is:
//   Az(x,y) = B0 λ ln(cosh(y/λ))  +  ψ0 cos(kx x) cos(ky y)
//
// Face-normal B is obtained by a finite-difference of Az across one cell
// (= the line integral of A along the face perimeter, by Stokes' theorem):
//
//   bx[i][j]  =  ΔAz / Δy  at (xi, yj + Δy/2)
//   by[i][j]  = −ΔAz / Δx  at (xi + Δx/2, yj)
//
// Discrete divergence:
//   (bx[i+1][j] − bx[i][j])/Δx + (by[i][j+1] − by[i][j])/Δy
//   = [Az(xi+1,yj+1)−Az(xi+1,yj) − Az(xi,yj+1)+Az(xi,yj)] / (ΔxΔy)
//   + [−Az(xi+1,yj+1)+Az(xi,yj+1) + Az(xi+1,yj)−Az(xi,yj)] / (ΔxΔy)
//   = 0   (exact cancellation)
//
// This holds for any Az, regardless of the perturbation amplitude or
// grid resolution, giving machine-precision ∇·B at t = 0.

double harris_bx_face(double x, double y, double dy,
                      const HarrisSheetParams& hp) {
    // bx at x-face centred at (x, y):
    //   y-location on this face spans [y − Δy/2, y + Δy/2]
    return (hp.Az(x, y + 0.5 * dy) - hp.Az(x, y - 0.5 * dy)) / dy;
}

double harris_by_face(double x, double y, double dx,
                      const HarrisSheetParams& hp) {
    // by at y-face centred at (x, y):
    //   x-location on this face spans [x − Δx/2, x + Δx/2]
    return -(hp.Az(x + 0.5 * dx, y) - hp.Az(x - 0.5 * dx, y)) / dx;
}

} // namespace my_project
