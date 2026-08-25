// Harris current-sheet initial conditions (test 11).
// References: Harris 1962 (equilibrium); Birn et al. 2001 (GEM domain and
// perturbation); Loureiro et al. 2007 (uniform-rho MHD convention);
// Tóth 2000 (CT vector-potential init). Derivations: harris_sheet.hpp.

#include "my_project/harris_sheet.hpp"

namespace my_project {

RunConfig make_harris_config(int nx, int ny,
                             DivBCleaningKind divb, SolverKind solver) {
    const HarrisSheetParams hp;
    RunConfig cfg;
    cfg.test   = 11;
    cfg.nx     = nx;
    cfg.ny     = ny;
    cfg.gamma  = 5.0 / 3.0;
    cfg.t_end  = 50.0;
    cfg.cfl    = (solver == SolverKind::FORCE) ? 0.4 : 0.3;
    cfg.solver = solver;
    cfg.divb   = divb;
    cfg.x0     = -0.5 * hp.Lx;       // x ∈ [−2π, 2π]
    cfg.x1     =  0.5 * hp.Lx;
    cfg.y0     = -0.5 * hp.Ly;       // y ∈ [−π, π]
    cfg.y1     =  0.5 * hp.Ly;
    cfg.bcx    = BC::Periodic;
    cfg.bcy    = BC::Transmissive;
    cfg.eta    = hp.eta;
    return cfg;
}

// Cell-centred IC: Harris equilibrium + single X-point perturbation from
// δAz = ψ0 cos(kx x) cos(ky y); ∇·(Beq + δB) = 0 analytically.
Vec harris_cell_ic(double x, double y, const HarrisSheetParams& hp) {
    const double Bx = hp.Bx_eq(y)
                    - hp.psi0 * hp.ky() * std::cos(hp.kx() * x) * std::sin(hp.ky() * y);
    const double By =   hp.psi0 * hp.kx() * std::sin(hp.kx() * x) * std::cos(hp.ky() * y);

    return {
        hp.rho_bg,    // rho (uniform)
        0.0,          // vx
        0.0,          // vy
        0.0,          // vz
        hp.p_eq(y),   // p (Harris pressure balance)
        Bx,           // equilibrium + perturbation
        By,           // perturbation only
        0.0,          // Bz
        0.0           // psi (GLM scalar)
    };
}

// Face-centred B from the finite difference of Az across one cell
// (= line integral, Tóth 2000 §4.1). The discrete face-difference ∇·B
// cancels exactly for any Az, giving machine-precision ∇·B at t = 0.
double harris_bx_face(double x, double y, double dy,
                      const HarrisSheetParams& hp) {
    return (hp.Az(x, y + 0.5 * dy) - hp.Az(x, y - 0.5 * dy)) / dy;
}

double harris_by_face(double x, double y, double dx,
                      const HarrisSheetParams& hp) {
    return -(hp.Az(x + 0.5 * dx, y) - hp.Az(x - 0.5 * dx, y)) / dx;
}

} // namespace my_project
