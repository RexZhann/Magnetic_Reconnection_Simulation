#include "my_project/solver.hpp"

#include "my_project/harris_sheet.hpp"
#include "my_project/riemann.hpp"
#include "my_project/state.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace my_project {

namespace {
using Clock = std::chrono::steady_clock;
using Sec = std::chrono::duration<double>;

double elapsed(Clock::time_point a, Clock::time_point b) {
    return Sec(b - a).count();
}

// CT mode parameters (both optional, pass nullptr for GLM/None):
//   face_bn : face-centered normal B at each slic interface (size n+4).
//             face_bn[i] for i=1..n+1 is the Bx (or By in rotated y-sweep) on
//             the face between slic cells i and i+1.  Both Riemann states have
//             their slot-5 (normal B) replaced by this value before the solve so
//             no spurious normal-B jump is fed to HLLD.
//   emf_out : receives the raw F[6] from the Riemann solver at each interface.
//             For x-sweeps Ez = -F[6]; for y-sweeps (rotated) Ez = +F[6].
//             The calling sweep function passes the correct sign to store_emf.
void slic_step(ScratchBuf& sc, Row& wp, int n, double dt, double dx,
               const RunConfig& cfg, DivergenceController& divb,
               const double* face_bn = nullptr,
               double* emf_out  = nullptr,
               double* cemf_out = nullptr,
               double* sgn_out  = nullptr) {
    const int N = n + 4;
    sc.ensure(N);
    Row& uc = sc.uc; Row& d0 = sc.d0; Row& d1 = sc.d1; Row& delta = sc.delta;
    Row& xL = sc.xL; Row& xR = sc.xR; Row& hL = sc.hL; Row& hR = sc.hR;
    Row& iflx = sc.iflx;

    for (int i = 0; i < N; ++i) uc[i] = pri2con(wp[i], cfg.gamma);

    for (int i = 1; i < N - 1; ++i) {
        for (int k = 0; k < NVAR; ++k) {
            d0[i][k] = uc[i][k] - uc[i - 1][k];
            d1[i][k] = uc[i + 1][k] - uc[i][k];
        }
    }

    for (int i = 1; i < N - 1; ++i) {
        for (int k = 0; k < NVAR; ++k) {
            double den = d1[i][k];
            double r = (std::fabs(den) < 1e-30) ? 0.0 : d0[i][k] / den;
            double phi = minmod(r);
            xL[i][k] = uc[i][k] - 0.5 * phi * d1[i][k];
            xR[i][k] = uc[i][k] + 0.5 * phi * d1[i][k];
        }
        if (xL[i][0] < 0.0 || xR[i][0] < 0.0) {
            xL[i] = uc[i]; xR[i] = uc[i];
        }
    }

    auto check_pressure = [&](const Vec& u) -> double {
        const double rho = std::max(u[0], 1e-14);
        const double vx = u[1] / rho, vy = u[2] / rho, vz = u[3] / rho;
        const double Bx = u[5], By = u[6], Bz = u[7];
        return (cfg.gamma - 1.0) * (u[4] - 0.5 * rho * (vx * vx + vy * vy + vz * vz)
               - 0.5 * (Bx * Bx + By * By + Bz * Bz));
    };

    const bool glm = divb.glm_enabled();
    const double ch  = divb.characteristic_speed();

    for (int i = 1; i < N - 1; ++i) {
        Vec fL = phys_flux(xL[i], cfg.gamma, glm, ch);
        Vec fR = phys_flux(xR[i], cfg.gamma, glm, ch);
        for (int k = 0; k < NVAR; ++k) {
            hL[i][k] = xL[i][k] - 0.5 * (dt / dx) * (fR[k] - fL[k]);
            hR[i][k] = xR[i][k] - 0.5 * (dt / dx) * (fR[k] - fL[k]);
        }
        if (hL[i][0] < 0.0 || hR[i][0] < 0.0 || check_pressure(hL[i]) < 0.0 || check_pressure(hR[i]) < 0.0) {
            hL[i] = uc[i]; hR[i] = uc[i];
        }
    }

    for (int i = 1; i < n + 2; ++i) {
        if (face_bn != nullptr) {
            // CT: override the normal B in both Riemann states with the
            // authoritative face-centered value so no spurious Bx jump
            // enters the solver.
            Vec uL = hR[i];
            Vec uR = hL[i + 1];
            uL[5] = face_bn[i];
            uR[5] = face_bn[i];
            if (cfg.solver == SolverKind::FORCE) {
                iflx[i] = force_flux(uL, uR, dt, dx, cfg.gamma, false, 0.0);
            } else {
                iflx[i] = hlld_flux(uL, uR, cfg.gamma, false, 0.0);
            }
            if (emf_out) emf_out[i] = iflx[i][6];
            // CT-Contact: centered (dissipation-free) EMF = average of L/R induction fluxes.
            // In x-sweep frame F[6] = vx*By - vy*Bx; same formula after y-sweep rotation.
            if (cemf_out) {
                double FcL = (uL[1] / uL[0]) * uL[6] - (uL[2] / uL[0]) * uL[5];
                double FcR = (uR[1] / uR[0]) * uR[6] - (uR[2] / uR[0]) * uR[5];
                cemf_out[i] = 0.5 * (FcL + FcR);
            }
            // sign(mass flux F[rho] = rho*vx): positive = flow rightward.
            if (sgn_out) sgn_out[i] = (iflx[i][0] >= 0.0) ? 1.0 : -1.0;
        } else {
            if (cfg.solver == SolverKind::FORCE) {
                iflx[i] = force_flux(hR[i], hL[i + 1], dt, dx, cfg.gamma, glm, ch);
            } else {
                iflx[i] = hlld_flux(hR[i], hL[i + 1], cfg.gamma, glm, ch);
            }
        }
    }

    for (int i = 2; i < n + 2; ++i) {
        for (int k = 0; k < NVAR; ++k) {
            if (!glm && k == 5) continue;  // normal B is handled by CT Faraday update
            uc[i][k] -= (dt / dx) * (iflx[i][k] - iflx[i - 1][k]);
        }
    }
    for (int i = 2; i < n + 2; ++i) wp[i] = con2pri(uc[i], cfg.gamma);
}

std::string solver_suffix(SolverKind s) {
    return s == SolverKind::FORCE ? "_force" : "_hlld";
}

std::string divb_suffix(DivBCleaningKind d) {
    if (d == DivBCleaningKind::GLM) return "_glm";
    if (d == DivBCleaningKind::CT) return "_ct";
    return "_noglm";
}

} // namespace

void ScratchBuf::ensure(int N) {
    if (N <= cap) return;
    cap = N;
    auto resize_row = [&](Row& r) { r.assign(N, Vec(NVAR, 0.0)); };
    resize_row(uc); resize_row(d0); resize_row(d1); resize_row(delta);
    resize_row(xL); resize_row(xR); resize_row(hL); resize_row(hR);
    resize_row(iflx); resize_row(s_row);
    face_bn_buf.assign(N, 0.0);
    emf_buf.assign(N, 0.0);
    cemf_buf.assign(N, 0.0);
    sgn_buf.assign(N, 0.0);
}

RunConfig make_config_for_test(int test, int nx, int ny,
                               DivBCleaningKind divb,
                               SolverKind solver) {
    RunConfig cfg;
    cfg.test = test;
    cfg.nx = nx;
    cfg.ny = ny;
    cfg.divb = divb;
    cfg.solver = solver;
    cfg.cfl = (solver == SolverKind::FORCE) ? 0.4 : 0.3;
    switch (test) {
        case 0:
            cfg.x0 = 0; cfg.x1 = 1; cfg.y0 = 0; cfg.y1 = 1; cfg.gamma = 1.4; cfg.t_end = 0.25;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Transmissive; break;
        case 1:
            cfg.x0 = 0; cfg.x1 = 800; cfg.y0 = 0; cfg.y1 = 800; cfg.gamma = 2.0; cfg.t_end = 80.0;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Transmissive; break;
        case 2:
            cfg.x0 = 0; cfg.x1 = 800; cfg.y0 = 0; cfg.y1 = 800; cfg.gamma = 2.0; cfg.t_end = 80.0;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Transmissive; break;
        case 3:
            cfg.x0 = 0; cfg.x1 = 1; cfg.y0 = 0; cfg.y1 = 1; cfg.gamma = 5.0 / 3.0; cfg.t_end = 0.5;
            cfg.bcx = BC::Periodic; cfg.bcy = BC::Periodic; break;
        case 4:
            cfg.x0 = 0; cfg.x1 = 1; cfg.y0 = 0; cfg.y1 = 1; cfg.gamma = 5.0 / 3.0; cfg.t_end = 0.18;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Transmissive; break;
        // 1D shock-tube tests from Miyoshi & Kusano (2005).
        // Domain [-0.5, 0.5] × [0, ny/nx] (square cells), γ=5/3, transmissive BC.
        // Typical invocation: ./mhd2d <test> 800 4 1 1
        case 5: // Dai & Woodward (1994) shock tube (Miyoshi Fig. 5), t=0.2
            cfg.x0 = -0.5; cfg.x1 = 0.5;
            cfg.y0 = 0.0;  cfg.y1 = static_cast<double>(ny) / nx;
            cfg.gamma = 5.0 / 3.0; cfg.t_end = 0.2;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Transmissive; cfg.cfl = 0.8; break;
        case 6: // Brio & Wu (1988) shock tube, γ=5/3 (Miyoshi Fig. 8), t=0.1
            cfg.x0 = -0.5; cfg.x1 = 0.5;
            cfg.y0 = 0.0;  cfg.y1 = static_cast<double>(ny) / nx;
            cfg.gamma = 5.0 / 3.0; cfg.t_end = 0.1;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Transmissive; break;
        case 7: // Slow switch-off shock (Falle et al. 1998, Miyoshi Fig. 9), t=0.2
            cfg.x0 = -0.5; cfg.x1 = 0.5;
            cfg.y0 = 0.0;  cfg.y1 = static_cast<double>(ny) / nx;
            cfg.gamma = 5.0 / 3.0; cfg.t_end = 0.2;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Transmissive; break;
        case 8: // Slow switch-off rarefaction (Falle et al. 1998, Miyoshi Fig. 10), t=0.2
            cfg.x0 = -0.5; cfg.x1 = 0.5;
            cfg.y0 = 0.0;  cfg.y1 = static_cast<double>(ny) / nx;
            cfg.gamma = 5.0 / 3.0; cfg.t_end = 0.2;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Transmissive; break;
        case 9: // Super-fast rarefaction Mf=3.0 (Miyoshi Fig. 11), t=0.05
            cfg.x0 = -0.5; cfg.x1 = 0.5;
            cfg.y0 = 0.0;  cfg.y1 = static_cast<double>(ny) / nx;
            cfg.gamma = 5.0 / 3.0; cfg.t_end = 0.05;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Transmissive; break;
        case 10: // Super-fast rarefaction Mf=3.1 (Miyoshi Fig. 11), t=0.05
            cfg.x0 = -0.5; cfg.x1 = 0.5;
            cfg.y0 = 0.0;  cfg.y1 = static_cast<double>(ny) / nx;
            cfg.gamma = 5.0 / 3.0; cfg.t_end = 0.05;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Transmissive; break;
        case 11: { // Harris current sheet — resistive MHD reconnection
            // Parameters and references: see include/my_project/harris_sheet.hpp
            //   Equilibrium:   Harris (1962), Il Nuovo Cimento 23, 115
            //   Domain/perturb: Birn et al. (2001), J. Geophys. Res. 106, 3715
            //   MHD convention: Loureiro et al. (2007), Phys. Plasmas 14, 100703
            const HarrisSheetParams hp;
            cfg.x0       = -0.5 * hp.Lx; cfg.x1 = 0.5 * hp.Lx;  // x ∈ [-2π, 2π]
            cfg.y0       = -0.5 * hp.Ly; cfg.y1 = 0.5 * hp.Ly;  // y ∈ [-π,  π]
            cfg.gamma    = 5.0 / 3.0;
            cfg.t_end    = 20.0;       // ≈ 1.6 Alfvén crossing times (Lx / vA∞)
            cfg.bcx      = BC::Periodic;      // reconnected flux wraps around in x
            cfg.bcy      = BC::Transmissive;  // outflow open boundary in y
            cfg.eta      = hp.eta;            // S = Lx/η ≈ 2500 (Sweet-Parker 1958)
            cfg.output_dt = 1.0;              // write snapshots at t=0,1,...
            break;
        }
        case 12: { // Harris current sheet — Hall MHD reconnection (GEM challenge)
            // Same equilibrium and domain as test 11; adds Hall term d_i/ρ J×B.
            // GEM parameters: Birn et al. (2001), J. Geophys. Res. 106, 3715
            //   d_i = 1.0 (ion inertial length ≈ 2λ; speeds reconnection ~10×)
            const HarrisSheetParams hp;
            cfg.x0       = -0.5 * hp.Lx; cfg.x1 = 0.5 * hp.Lx;
            cfg.y0       = -0.5 * hp.Ly; cfg.y1 = 0.5 * hp.Ly;
            cfg.gamma    = 5.0 / 3.0;
            cfg.t_end    = 12.0;       // Hall reconnection is much faster; 12 is enough
            cfg.bcx      = BC::Periodic;
            cfg.bcy      = BC::Transmissive;
            cfg.eta      = hp.eta;            // same resistivity as resistive MHD run
            cfg.hall_di  = 1.0;               // GEM challenge: d_i = 1.0 (= 2λ)
            cfg.output_dt = 1.0;
            cfg.rho_floor = 0.02;  // prevent density cavitation (10% of background 0.2)
            cfg.p_floor   = 0.005; // pressure floor
            break;
        }
        case 13: { // Circularly polarised Alfvén wave — Tóth (2000) §5.1 convergence test
            // 1D periodic domain [0,1]; va = B0/sqrt(rho0) = 1, so t_end=1 is one
            // Alfvén crossing time and the solution returns to its exact initial state.
            cfg.x0 = 0.0; cfg.x1 = 1.0;
            cfg.y0 = 0.0; cfg.y1 = static_cast<double>(ny) / nx;
            cfg.gamma = 5.0 / 3.0; cfg.t_end = 1.0;
            cfg.bcx = BC::Periodic; cfg.bcy = BC::Transmissive; break;
        }
        default:
            throw std::runtime_error("Unknown test id");
    }
    return cfg;
}

std::unique_ptr<DivergenceController> make_divergence_controller(DivBCleaningKind kind) {
    if (kind == DivBCleaningKind::GLM) return std::make_unique<GLMDivergenceCleaning>();
    if (kind == DivBCleaningKind::CT) return std::make_unique<CTDivergenceControl>();
    return std::make_unique<NoDivBCleaning>();
}

void apply_bc(Grid& w, int nx, int ny, BC bcx, BC bcy) {
    for (int j = 0; j < ny + 4; ++j) {
        if (bcx == BC::Transmissive) {
            w[0][j] = w[2][j]; w[1][j] = w[2][j];
            w[nx + 2][j] = w[nx + 1][j]; w[nx + 3][j] = w[nx + 1][j];
        } else {
            w[0][j] = w[nx][j]; w[1][j] = w[nx + 1][j];
            w[nx + 2][j] = w[2][j]; w[nx + 3][j] = w[3][j];
        }
    }
    for (int i = 0; i < nx + 4; ++i) {
        if (bcy == BC::Transmissive) {
            w[i][0] = w[i][2]; w[i][1] = w[i][2];
            w[i][ny + 2] = w[i][ny + 1]; w[i][ny + 3] = w[i][ny + 1];
        } else {
            w[i][0] = w[i][ny]; w[i][1] = w[i][ny + 1];
            w[i][ny + 2] = w[i][2]; w[i][ny + 3] = w[i][3];
        }
    }
}

void apply_floor(Grid& w, int nx, int ny, const RunConfig& cfg) {
    if (cfg.rho_floor <= 0.0 && cfg.p_floor <= 0.0) return;
    const double gm1 = cfg.gamma - 1.0;
    for (int i = 2; i < nx + 2; ++i)
        for (int j = 2; j < ny + 2; ++j) {
            Vec& p = w[i][j];
            if (cfg.rho_floor > 0.0 && p[0] < cfg.rho_floor) p[0] = cfg.rho_floor;
            if (cfg.p_floor   > 0.0 && p[4] < cfg.p_floor)   p[4] = cfg.p_floor;
        }
}

void sweep_x(Grid& w, int nx, int ny, double dt, double dx, const RunConfig& cfg,
             DivergenceController& divb) {
    const bool ct = divb.uses_face_centered_b();
    #pragma omp parallel
    {
        ScratchBuf sc;
        sc.ensure(nx + 4);
        #pragma omp for schedule(static)
        for (int j = 2; j < ny + 2; ++j) {
            const int j_int = j - 2;
            Row& s = sc.s_row;
            for (int i = 0; i < nx + 4; ++i) s[i] = w[i][j];
            if (ct) {
                divb.fill_face_bn_x(j_int, nx, sc.face_bn_buf.data());
                slic_step(sc, s, nx, dt, dx, cfg, divb,
                          sc.face_bn_buf.data(), sc.emf_buf.data(),
                          sc.cemf_buf.data(), sc.sgn_buf.data());
                // x-direction: F[6] = vx·By − vy·Bx = −Ez, so Ez = −F[6].
                // Apply the same sign flip to both numerical and centered EMF.
                // sgn_buf holds sign(ρvx) which is frame-independent — no flip.
                for (int ii = 1; ii <= nx + 1; ++ii) {
                    sc.emf_buf[ii]  = -sc.emf_buf[ii];
                    sc.cemf_buf[ii] = -sc.cemf_buf[ii];
                }
                divb.store_emf_x(j_int, nx, sc.emf_buf.data(),
                                 sc.cemf_buf.data(), sc.sgn_buf.data());
            } else {
                slic_step(sc, s, nx, dt, dx, cfg, divb);
            }
            for (int i = 2; i < nx + 2; ++i) w[i][j] = s[i];
        }
    }
}

void sweep_y(Grid& w, int nx, int ny, double dt, double dy, const RunConfig& cfg,
             DivergenceController& divb) {
    const bool ct = divb.uses_face_centered_b();
    #pragma omp parallel
    {
        ScratchBuf sc;
        sc.ensure(ny + 4);
        #pragma omp for schedule(static)
        for (int i = 2; i < nx + 2; ++i) {
            const int i_int = i - 2;
            Row& s = sc.s_row;
            for (int j = 0; j < ny + 4; ++j) {
                s[j] = w[i][j];
                std::swap(s[j][1], s[j][2]);
                std::swap(s[j][5], s[j][6]);
            }
            if (ct) {
                // In the rotated y-sweep frame slot-5 holds By_original.
                // fill_face_bn_y returns face_.by[i_int][j-1] = By on y-faces.
                divb.fill_face_bn_y(i_int, ny, sc.face_bn_buf.data());
                slic_step(sc, s, ny, dt, dy, cfg, divb,
                          sc.face_bn_buf.data(), sc.emf_buf.data(),
                          sc.cemf_buf.data(), sc.sgn_buf.data());
                // Rotated F[6] = Bx_orig·vy_orig − vx_orig·By_orig = +Ez; no sign flip.
                // sgn_buf holds sign(ρvy_orig) — correct upwind direction.
                divb.store_emf_y(i_int, ny, sc.emf_buf.data(),
                                 sc.cemf_buf.data(), sc.sgn_buf.data());
            } else {
                slic_step(sc, s, ny, dt, dy, cfg, divb);
            }
            for (int j = 2; j < ny + 2; ++j) {
                std::swap(s[j][1], s[j][2]);
                std::swap(s[j][5], s[j][6]);
                w[i][j] = s[j];
            }
        }
    }
}

double compute_dt(const Grid& w, int nx, int ny, double dx, double dy,
                  const RunConfig& cfg, DivergenceController& divb) {
    double smax = 1e-10, ch_loc = 0.0, max_va2 = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:smax,ch_loc,max_va2) schedule(static)
    for (int i = 2; i < nx + 2; ++i) {
        for (int j = 2; j < ny + 2; ++j) {
            const Vec& p = w[i][j];
            double cfx = calc_cf(p[0], p[4], p[5], p[6], p[7], cfg.gamma);
            double cfy = calc_cf(p[0], p[4], p[6], p[5], p[7], cfg.gamma);
            double sx = std::fabs(p[1]) + cfx;
            double sy = std::fabs(p[2]) + cfy;
            double s = sx / dx + sy / dy;
            smax = std::max(smax, s);
            ch_loc = std::max(ch_loc, std::max(sx, sy));
            // Track max Alfvén speed for Hall CFL: v_A = |B|/sqrt(ρ)
            double B2 = p[5]*p[5] + p[6]*p[6] + p[7]*p[7];
            double rho_va = cfg.hall_di > 0.0 ? std::max(p[0], 0.1) : std::max(p[0], 1e-14);
            max_va2 = std::max(max_va2, B2 / rho_va);
        }
    }
    divb.update_characteristic_speed(ch_loc);
    // Hall whistler CFL (RK4 integration).
    // RK4 is stable for |ω·dt| < 2√2 ≈ 2.83 on the imaginary axis.
    // ω_max = d_i·(π/mincell)²·v_A, so need d_i·π²·v_A/mincell² · dt < 2.83.
    // Express as smax contribution: smax_hall = d_i·π²·v_A / (2.83·mincell²).
    if (cfg.hall_di > 0.0) {
        constexpr double pi = 3.14159265358979323846;
        double mincell = std::min(dx, dy);
        double smax_hall = cfg.hall_di * (pi*pi/2.83) * std::sqrt(max_va2)
                           / (mincell * mincell);
        smax = std::max(smax, smax_hall);
    }
    return cfg.cfl / smax;
}

// When face_field is provided (CT mode) the exact face-difference ∇·B is used,
// which is the quantity preserved to machine precision by the Faraday update.
std::vector<std::vector<double>> compute_divB(const Grid& w, int nx, int ny,
                                              double dx, double dy,
                                              const FaceField2D* face_field) {
    std::vector<std::vector<double>> dB(nx, std::vector<double>(ny, 0.0));
    if (face_field && !face_field->empty()) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                double d = (face_field->bx[i + 1][j] - face_field->bx[i][j]) / dx
                         + (face_field->by[i][j + 1] - face_field->by[i][j]) / dy;
                dB[i][j] = std::fabs(d);
            }
        }
        return dB;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 2; i < nx + 2; ++i) {
        for (int j = 2; j < ny + 2; ++j) {
            double d = (w[i + 1][j][5] - w[i - 1][j][5]) / (2 * dx)
                     + (w[i][j + 1][6] - w[i][j - 1][6]) / (2 * dy);
            dB[i - 2][j - 2] = std::fabs(d);
        }
    }
    return dB;
}

Diagnostics compute_diagnostics(const Grid& w, int nx, int ny,
                                double dx, double dy,
                                const FaceField2D* face_field) {
    Diagnostics d;
    double min_rho = 1e30, min_p = 1e30, max_divB = 0.0, max_psi = 0.0, max_v = 0.0;
    double divB_sum2 = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) \
        reduction(min:min_rho,min_p) reduction(max:max_divB,max_psi,max_v) \
        reduction(+:divB_sum2)
    for (int i = 2; i < nx + 2; ++i) {
        for (int j = 2; j < ny + 2; ++j) {
            const Vec& p = w[i][j];
            min_rho = std::min(min_rho, p[0]);
            min_p = std::min(min_p, p[4]);
            max_psi = std::max(max_psi, std::fabs(p[8]));
            max_v = std::max(max_v, std::sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
            if (!face_field || face_field->empty()) {
                double dBx = (w[i + 1][j][5] - w[i - 1][j][5]) / (2 * dx);
                double dBy = (w[i][j + 1][6] - w[i][j - 1][6]) / (2 * dy);
                double div = dBx + dBy;
                max_divB = std::max(max_divB, std::fabs(div));
                divB_sum2 += div * div;
            }
        }
    }
    if (face_field && !face_field->empty()) {
        #pragma omp parallel for collapse(2) \
            reduction(max:max_divB) reduction(+:divB_sum2) schedule(static)
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                double dBx = (face_field->bx[i + 1][j] - face_field->bx[i][j]) / dx;
                double dBy = (face_field->by[i][j + 1] - face_field->by[i][j]) / dy;
                double div = dBx + dBy;
                max_divB = std::max(max_divB, std::fabs(div));
                divB_sum2 += div * div;
            }
        }
    }
    d.min_rho = min_rho; d.min_p = min_p; d.max_divB = max_divB;
    d.l2_divB = std::sqrt(divB_sum2 * dx * dy);
    d.max_psi = max_psi; d.max_v = max_v;
    return d;
}

void initialize_problem(Grid& w, const RunConfig& cfg) {
    const double dx = (cfg.x1 - cfg.x0) / cfg.nx;
    const double dy = (cfg.y1 - cfg.y0) / cfg.ny;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 2; i < cfg.nx + 2; ++i) {
        for (int j = 2; j < cfg.ny + 2; ++j) {
            double x = cfg.x0 + (i - 1.5) * dx;
            double y = cfg.y0 + (j - 1.5) * dy;
            switch (cfg.test) {
                case 0: {
                    double r = std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
                    w[i][j] = (r <= 0.4) ? Vec{1,0,0,0,1,0,0,0,0} : Vec{0.125,0,0,0,0.1,0,0,0,0};
                    break;
                }
                case 1: {
                    double m = 0.5 * (cfg.x0 + cfg.x1);
                    w[i][j] = (x <= m) ? Vec{1,0,0,0,1,0.75,1,0,0} : Vec{0.125,0,0,0,0.1,0.75,-1,0,0};
                    break;
                }
                case 2: {
                    double m = 0.5 * (cfg.y0 + cfg.y1);
                    w[i][j] = (y <= m) ? Vec{1,0,0,0,1,1,0.75,0,0} : Vec{0.125,0,0,0,0.1,-1,0.75,0,0};
                    break;
                }
                case 3: {
                    constexpr double pi = 3.14159265358979323846;
                    double rho = cfg.gamma * cfg.gamma;
                    w[i][j] = {rho, -std::sin(2*pi*y), std::sin(2*pi*x), 0, cfg.gamma,
                               -std::sin(2*pi*y), std::sin(4*pi*x), 0, 0};
                    break;
                }
                case 4: {
                    constexpr double pi = 3.14159265358979323846;
                    double r0 = 0.1, r1 = 0.115, v0 = 1.0;
                    double r = std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
                    double ur = (0.5 - y) * v0, vr = (x - 0.5) * v0;
                    double Bx0 = 2.5 / std::sqrt(4.0 * pi);
                    double rho, vxv, vyv;
                    if (r < r0) {
                        rho = 10; vxv = ur / r0; vyv = vr / r0;
                    } else if (r < r1) {
                        double fr = (r1 - r) / (r1 - r0);
                        rho = 1 + 9 * fr; vxv = ur * fr / r; vyv = vr * fr / r;
                    } else {
                        rho = 1; vxv = 0; vyv = 0;
                    }
                    w[i][j] = {rho, vxv, vyv, 0, 0.5, Bx0, 0, 0, 0};
                    break;
                }
                // ---------------------------------------------------------------
                // 1D shock-tube tests (Miyoshi & Kusano 2005).
                // All use domain [-0.5,0.5]; discontinuity at x=0.
                // Primitive state: {rho, vx, vy, vz, p, Bx, By, Bz, psi}
                // ---------------------------------------------------------------
                case 5: {
                    // Dai & Woodward (1994) — Miyoshi Fig. 5
                    constexpr double pi = 3.14159265358979323846;
                    const double sqrt4pi = std::sqrt(4.0 * pi);
                    Vec L = {1.08, 1.2,  0.01, 0.5, 0.95,
                             4.0/sqrt4pi, 3.6/sqrt4pi, 2.0/sqrt4pi, 0};
                    Vec R = {1.0,  0.0,  0.0,  0.0, 1.0,
                             4.0/sqrt4pi, 4.0/sqrt4pi, 2.0/sqrt4pi, 0};
                    w[i][j] = (x < 0.0) ? L : R;
                    break;
                }
                case 6: {
                    // Brio & Wu (1988), γ=5/3 — Miyoshi Fig. 8
                    Vec L = {1.0,   0, 0, 0, 1.0,   0.75,  1.0, 0, 0};
                    Vec R = {0.125, 0, 0, 0, 0.1,   0.75, -1.0, 0, 0};
                    w[i][j] = (x < 0.0) ? L : R;
                    break;
                }
                case 7: {
                    // Slow switch-off shock (Falle et al. 1998) — Miyoshi Fig. 9
                    Vec L = {1.368, 0.269, 1.0, 0, 1.769, 1.0, 0.0, 0, 0};
                    Vec R = {1.0,   0.0,   0.0, 0, 1.0,   1.0, 1.0, 0, 0};
                    w[i][j] = (x < 0.0) ? L : R;
                    break;
                }
                case 8: {
                    // Slow switch-off rarefaction (Falle et al. 1998) — Miyoshi Fig. 10
                    Vec L = {1.0,  0.0,   0.0,   0, 2.0,    1.0, 0.0,    0, 0};
                    Vec R = {0.2,  1.186, 2.967, 0, 0.1368, 1.0, 1.6405, 0, 0};
                    w[i][j] = (x < 0.0) ? L : R;
                    break;
                }
                case 9: {
                    // Super-fast expansion Mf=3.0 — Miyoshi Fig. 11
                    Vec L = {1, -3.0, 0, 0, 0.45, 0, 0.5, 0, 0};
                    Vec R = {1,  3.0, 0, 0, 0.45, 0, 0.5, 0, 0};
                    w[i][j] = (x < 0.0) ? L : R;
                    break;
                }
                case 10: {
                    // Super-fast expansion Mf=3.1 — Miyoshi Fig. 11
                    Vec L = {1, -3.1, 0, 0, 0.45, 0, 0.5, 0, 0};
                    Vec R = {1,  3.1, 0, 0, 0.45, 0, 0.5, 0, 0};
                    w[i][j] = (x < 0.0) ? L : R;
                    break;
                }
                case 11: {
                    // Harris current-sheet — resistive MHD
                    // Uniform density (Loureiro et al. 2007 convention): ρ = ρ_bg = 1.
                    w[i][j] = harris_cell_ic(x, y, HarrisSheetParams{});
                    break;
                }
                case 12: {
                    // Harris current-sheet — Hall MHD (GEM challenge)
                    // Use kinetic Harris density n(y) = n_bg + sech²(y/λ) so that
                    // ρ_min = n_bg = 0.2 prevents (d_i/ρ)J×B from diverging at X-line.
                    // Reference: Birn et al. (2001), J. Geophys. Res. 106, 3715.
                    // Pressure p_eq(y) = 0.5·(0.2 + sech²(y/λ)) is already consistent
                    // with this density at uniform temperature T = 0.5 — no change needed.
                    const HarrisSheetParams hp;
                    Vec ic = harris_cell_ic(x, y, hp);
                    const double s = 1.0 / std::cosh(y / hp.lam);
                    ic[0] = 0.2 + s * s;   // GEM: n_bg=0.2, n0=1.0
                    w[i][j] = ic;
                    break;
                }
                case 13: {
                    // Circularly polarised Alfvén wave — convergence test
                    // following Tóth (2000) §5.1
                    constexpr double pi = 3.14159265358979323846;
                    const double k  = 2.0 * pi;          // wavenumber
                    const double B0 = 1.0, rho0 = 1.0;
                    const double va = B0 / std::sqrt(rho0);
                    const double Bperp = 0.1 * B0;       // small-amplitude
                    w[i][j] = { rho0,
                                0.0,
                               -va * Bperp * std::sin(k * x) / B0,
                                va * Bperp * std::cos(k * x) / B0,
                                0.1,                     // thermal pressure
                                B0,
                                Bperp * std::sin(k * x),
                               -Bperp * std::cos(k * x),
                                0.0 };
                    break;
                }
                default:
                    throw std::runtime_error("Unknown test id");
            }
        }
    }
}

OutputData run_simulation(const RunConfig& cfg) {
    const double dx = (cfg.x1 - cfg.x0) / cfg.nx;
    const double dy = (cfg.y1 - cfg.y0) / cfg.ny;
    const double Lx = cfg.x1 - cfg.x0;
    const double Ly = cfg.y1 - cfg.y0;

    Grid w(cfg.nx + 4, std::vector<Vec>(cfg.ny + 4, Vec(NVAR, 0.0)));
    initialize_problem(w, cfg);
    // Fill ghost cells before CT initializes face B from cell averages.
    apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
    auto divb = make_divergence_controller(cfg.divb);
    divb->set_adiabatic_index(cfg.gamma);
    divb->set_boundary_conditions(cfg.bcx, cfg.bcy);
    divb->set_resistivity(cfg.eta);
    divb->set_hall(cfg.hall_di);
    divb->initialize(w, cfg, dx, dy);

    double t = 0.0;
    int step = 0;
    double t_sweepx = 0.0, t_sweepy = 0.0, t_other = 0.0;
    auto T0 = Clock::now();

    if (cfg.rho_floor > 0.0)
        std::cout << "  density floor = " << cfg.rho_floor << ",  p_floor = " << cfg.p_floor << "\n";

    // Snapshot output: write at t=0, then every output_dt.
    const bool do_snaps = cfg.output_dt > 0.0;
    int snap_idx = 0;
    double next_snap_t = cfg.output_dt;
    if (do_snaps) {
        auto dB0 = compute_divB(w, cfg.nx, cfg.ny, dx, dy, divb->face_field());
        write_snapshot_file(w, dB0, cfg, snap_idx++, 0.0);
        std::cout << "Snapshot 0 written (t=0)\n";
    }

    // L2 divB log: one line per diagnostic interval.
    const std::string divb_log_path = "output/test" + std::to_string(cfg.test) + "_"
        + std::to_string(cfg.nx) + "x" + std::to_string(cfg.ny)
        + solver_suffix(cfg.solver) + divb_suffix(cfg.divb) + "_divBl2.log";
    std::ofstream divb_log(divb_log_path);
    divb_log << "# t  l2_divB\n";
    {
        Diagnostics d0 = compute_diagnostics(w, cfg.nx, cfg.ny, dx, dy, divb->face_field());
        divb_log << 0.0 << ' ' << d0.l2_divB << '\n';
    }

    while (t < cfg.t_end) {
        auto ta = Clock::now();
        apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
        double dt = compute_dt(w, cfg.nx, cfg.ny, dx, dy, cfg, *divb);
        if (t + dt > cfg.t_end) dt = cfg.t_end - t;
        if (dt <= 0) break;

        apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
        divb->pre_step(w, cfg.nx, cfg.ny, dt, dx, dy);
        auto tb = Clock::now();

        // Alternating Strang splitting: even steps XYX, odd steps YXY.
        Clock::time_point tc, td, te;
        if (step % 2 == 0) {
            sweep_x(w, cfg.nx, cfg.ny, 0.5 * dt, dx, cfg, *divb);
            apply_floor(w, cfg.nx, cfg.ny, cfg);
            tc = Clock::now();
            apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
            sweep_y(w, cfg.nx, cfg.ny, dt, dy, cfg, *divb);
            apply_floor(w, cfg.nx, cfg.ny, cfg);
            td = Clock::now();
            apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
            sweep_x(w, cfg.nx, cfg.ny, 0.5 * dt, dx, cfg, *divb);
            apply_floor(w, cfg.nx, cfg.ny, cfg);
            te = Clock::now();
            t_sweepx += elapsed(tb, tc) + elapsed(td, te);
            t_sweepy += elapsed(tc, td);
        } else {
            sweep_y(w, cfg.nx, cfg.ny, 0.5 * dt, dy, cfg, *divb);
            apply_floor(w, cfg.nx, cfg.ny, cfg);
            tc = Clock::now();
            apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
            sweep_x(w, cfg.nx, cfg.ny, dt, dx, cfg, *divb);
            apply_floor(w, cfg.nx, cfg.ny, cfg);
            td = Clock::now();
            apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
            sweep_y(w, cfg.nx, cfg.ny, 0.5 * dt, dy, cfg, *divb);
            apply_floor(w, cfg.nx, cfg.ny, cfg);
            te = Clock::now();
            t_sweepy += elapsed(tb, tc) + elapsed(td, te);
            t_sweepx += elapsed(tc, td);
        }

        divb->post_step(w, cfg.nx, cfg.ny, dt, Lx, Ly, dx, dy);
        auto tf = Clock::now();

        t_other += elapsed(ta, tb) + elapsed(te, tf);
        t += dt;
        ++step;

        if (do_snaps && t >= next_snap_t - 1e-12 * cfg.output_dt) {
            apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
            auto dB_snap = compute_divB(w, cfg.nx, cfg.ny, dx, dy, divb->face_field());
            write_snapshot_file(w, dB_snap, cfg, snap_idx++, t);
            next_snap_t += cfg.output_dt;
        }

        if (step % 200 == 0) {
            apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
            Diagnostics diag = compute_diagnostics(w, cfg.nx, cfg.ny, dx, dy,
                                                   divb->face_field());
            std::cout << "Step " << step
                      << "  t=" << t
                      << "  dt=" << dt
                      << "  min_rho=" << diag.min_rho
                      << "  min_p=" << diag.min_p
                      << "  max|divB|=" << diag.max_divB
                      << "  l2|divB|=" << diag.l2_divB
                      << "  max|psi|=" << diag.max_psi
                      << "  max|v|=" << diag.max_v
                      << '\n';
            divb_log << t << ' ' << diag.l2_divB << '\n';
            if (diag.min_rho < 0 || diag.min_p < 0 || !std::isfinite(diag.max_divB)
                    || !std::isfinite(diag.max_v) || dt < 1e-8 * (t > 0 ? t : 1.0)) {
                std::cerr << "*** FATAL: unphysical state at step " << step
                          << " (dt=" << dt << ")\n";
                break;
            }
        }
    }

    TimingStats timing;
    timing.total = elapsed(T0, Clock::now());
    timing.sweep_x = t_sweepx;
    timing.sweep_y = t_sweepy;
    timing.other = t_other;
    timing.steps = step;
    timing.t_final = t;

    apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
    OutputData out;
    out.primitive = std::move(w);
    out.divB = compute_divB(out.primitive, cfg.nx, cfg.ny, dx, dy, divb->face_field());
    out.timing = timing;
    if (const FaceField2D* ff = divb->face_field()) {
        out.has_face_field = true;
        out.face_field = *ff;
    }
    return out;
}

void write_snapshot_file(const Grid& w,
                         const std::vector<std::vector<double>>& divB,
                         const RunConfig& cfg, int snap_idx, double t) {
    const double dx = (cfg.x1 - cfg.x0) / cfg.nx;
    const double dy = (cfg.y1 - cfg.y0) / cfg.ny;
    std::ostringstream ss;
    ss << "output/test" << cfg.test << "_"
       << cfg.nx << "x" << cfg.ny
       << solver_suffix(cfg.solver) << divb_suffix(cfg.divb)
       << "_snap" << std::setw(3) << std::setfill('0') << snap_idx << ".dat";
    std::ofstream file(ss.str());
    // Header has 5 fields (nx ny gamma glm t) so Python can recover simulation time.
    file << cfg.nx << ' ' << cfg.ny << ' ' << cfg.gamma << ' '
         << (cfg.divb == DivBCleaningKind::GLM ? 1 : 0) << ' ' << t << '\n';
    for (int j = 2; j < cfg.ny + 2; ++j) {
        for (int i = 2; i < cfg.nx + 2; ++i) {
            double x = cfg.x0 + (i - 1.5) * dx;
            double y = cfg.y0 + (j - 1.5) * dy;
            const Vec& p = w[i][j];
            double e = p[4] / ((cfg.gamma - 1.0) * p[0]);
            file << x << ' ' << y;
            for (int k = 0; k < NVAR; ++k) file << ' ' << p[k];
            file << ' ' << e << ' ' << divB[i - 2][j - 2] << '\n';
        }
    }
}

void write_output_file(const OutputData& out, const RunConfig& cfg) {
    const double dx = (cfg.x1 - cfg.x0) / cfg.nx;
    const double dy = (cfg.y1 - cfg.y0) / cfg.ny;
    const std::string folder = "output/";
    const std::string filename = folder + "test" + std::to_string(cfg.test) + "_"
        + std::to_string(cfg.nx) + "x" + std::to_string(cfg.ny)
        + solver_suffix(cfg.solver) + divb_suffix(cfg.divb) + ".dat";
    std::ofstream file(filename);
    file << cfg.nx << ' ' << cfg.ny << ' ' << cfg.gamma << ' '
         << (cfg.divb == DivBCleaningKind::GLM ? 1 : 0) << '\n';
    for (int j = 2; j < cfg.ny + 2; ++j) {
        for (int i = 2; i < cfg.nx + 2; ++i) {
            double x = cfg.x0 + (i - 1.5) * dx;
            double y = cfg.y0 + (j - 1.5) * dy;
            const Vec& p = out.primitive[i][j];
            double e = p[4] / ((cfg.gamma - 1.0) * p[0]);
            file << x << ' ' << y;
            for (int k = 0; k < NVAR; ++k) file << ' ' << p[k];
            file << ' ' << e << ' ' << out.divB[i - 2][j - 2] << '\n';
        }
    }
}

} // namespace
