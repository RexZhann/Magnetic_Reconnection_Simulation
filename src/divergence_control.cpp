#include "my_project/divergence_control.hpp"
#include "my_project/perf.hpp"
#include "my_project/harris_sheet.hpp"
#include "my_project/asym_harris_sheet.hpp"
#include "my_project/double_harris.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <omp.h>

namespace my_project {

// ---------------------------------------------------------------------------
// GLM: exponential damping of the divergence-cleaning scalar ψ
// ---------------------------------------------------------------------------
void GLMDivergenceCleaning::post_step(Grid& w, int nx, int ny, double dt, double Lx, double Ly,
                                      double /*dx*/, double /*dy*/) {
    const double cp = 0.18 * std::max(Lx, Ly);
    const double factor = std::exp(-ch_ * dt / cp);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 2; i < nx + 2; ++i) {
        for (int j = 2; j < ny + 2; ++j) {
            w[i][j][8] *= factor;
        }
    }
}

// ---------------------------------------------------------------------------
// CT: initialization
// ---------------------------------------------------------------------------
void CTDivergenceControl::initialize(Grid& w, const RunConfig& cfg, double dx, double dy) {
    nx_ = cfg.nx;
    ny_ = cfg.ny;
    face_.resize(cfg.nx, cfg.ny);
    emf_x_.assign(cfg.nx + 1, std::vector<double>(cfg.ny, 0.0));
    emf_y_.assign(cfg.nx,     std::vector<double>(cfg.ny + 1, 0.0));
    cemf_x_.assign(cfg.nx + 1, std::vector<double>(cfg.ny, 0.0));
    cemf_y_.assign(cfg.nx,     std::vector<double>(cfg.ny + 1, 0.0));
    sgn_x_.assign (cfg.nx + 1, std::vector<double>(cfg.ny, 0.0));
    sgn_y_.assign (cfg.nx,     std::vector<double>(cfg.ny + 1, 0.0));
    rho_floor_ = cfg.rho_floor;
    initialize_faces_from_problem(w, cfg, dx, dy);
    sync_cell_centered_from_faces(w, cfg.nx, cfg.ny);
}

// ---------------------------------------------------------------------------
// CT: pre_step — no-op.
// Face B is maintained solely by the Faraday update in post_step. Overwriting
// boundary faces here would corrupt the face-difference ∇·B invariant.
// ---------------------------------------------------------------------------
void CTDivergenceControl::pre_step(Grid& /*w*/, int /*nx*/, int /*ny*/,
                                   double /*dt*/, double /*dx*/, double /*dy*/) {}

// ---------------------------------------------------------------------------
// CT: post_step
// ---------------------------------------------------------------------------
void CTDivergenceControl::post_step(Grid& w, int nx, int ny,
                                    double dt, double /*Lx*/, double /*Ly*/,
                                    double dx, double dy) {
    // Step A: assemble ideal corner EMF from Riemann-solver interface fluxes.
    double tw = omp_get_wtime();
    compute_corner_emf_from_interface_emfs(nx, ny);
    perf::t_ct_assemble += omp_get_wtime() - tw;
    tw = omp_get_wtime();

    // ------------------------------------------------------------------
    // Determine sub-cycle count for the non-ideal block.
    // N_sub == 1 → original single-step path, bitwise identical.
    // ------------------------------------------------------------------
    int    N_sub   = 1;
    double dt_sub  = dt;
    double dt_eta  = std::numeric_limits<double>::infinity();  // inf = inactive
    double dt_hall = std::numeric_limits<double>::infinity();  // inf = inactive

    if (subcycle_nonideal_ && (eta_ > 0.0 || hall_di_ > 0.0)) {
        constexpr double pi = 3.14159265358979323846;
        double mincell = std::min(dx, dy);

        // Resistive (parabolic) CFL: dt_eta = mincell² / (2 η)
        if (eta_ > 0.0)
            dt_eta = mincell * mincell / (2.0 * eta_);

        // Hall sub-step CFL: HALL_HLL uses |B|/ρ (stricter); others RK4 with vA.
        if (hall_di_ > 0.0) {
            if (hall_stab_ == HallStabKind::HALL_HLL) {
                // c_w·dt/h ≤ 1 with c_w = π·di·|B|/(ρ·h)
                double max_b2_rho2 = 1e-14;
                #pragma omp parallel for collapse(2) reduction(max:max_b2_rho2) schedule(static)
                for (int i = 2; i < nx + 2; ++i)
                    for (int j = 2; j < ny + 2; ++j) {
                        double rho = std::max(w[i][j][0], rho_floor_);
                        double B2  = w[i][j][5]*w[i][j][5]
                                   + w[i][j][6]*w[i][j][6]
                                   + w[i][j][7]*w[i][j][7];
                        max_b2_rho2 = std::max(max_b2_rho2, B2 / (rho * rho));
                    }
                dt_hall = cfl_ * mincell * mincell
                          / (pi * hall_di_ * std::sqrt(max_b2_rho2));
            } else {
                // RK4 stability |ω·dt| < 2√2 ≈ 2.83, with vA = |B|/√ρ
                double max_va2 = 1e-14;
                #pragma omp parallel for collapse(2) reduction(max:max_va2) schedule(static)
                for (int i = 2; i < nx + 2; ++i)
                    for (int j = 2; j < ny + 2; ++j) {
                        double rho = std::max(w[i][j][0], rho_floor_);
                        double B2  = w[i][j][5]*w[i][j][5]
                                   + w[i][j][6]*w[i][j][6]
                                   + w[i][j][7]*w[i][j][7];
                        max_va2 = std::max(max_va2, B2 / rho);
                    }
                dt_hall = cfl_ * (2.83 / (pi*pi)) * mincell * mincell
                          / (hall_di_ * std::sqrt(max_va2));
            }
        }

        dt_sub   = std::min(dt_eta, dt_hall);
        int N_raw = static_cast<int>(std::ceil(dt / dt_sub));
        if (N_raw > n_subcycle_max_) {
            std::fprintf(stderr,
                "[subcycle] WARNING t=%.6g: dt_hyp=%.3e dt_sub=%.3e"
                " uncapped N=%d -> clamped to %d\n",
                current_t_, dt, dt_sub, N_raw, n_subcycle_max_);
            N_raw = n_subcycle_max_;
        }
        N_sub = std::max(1, N_raw);
    }
    last_n_sub_ = N_sub;
    perf::t_subcfl += omp_get_wtime() - tw;
    ++perf::nsub_calls;
    perf::nsub_sum += N_sub;
    perf::nsub_max = std::max(perf::nsub_max, N_sub);
    if (N_sub == n_subcycle_max_) ++perf::nsub_capped;

    // ------------------------------------------------------------------
    // N_sub == 1: original single-step path (bitwise identical to old code).
    // ------------------------------------------------------------------
    // Stabilization dispatch, shared by single-step and sub-cycle paths.
    const auto apply_stab = [&](double dt_loc) {
        const double ts = omp_get_wtime();
        switch (hall_stab_) {
            case HallStabKind::HYPER_RES:
                add_hyper_resistive_correction(nx, ny, dx, dy);
                break;
            case HallStabKind::HALL_HLL:
                add_hall_hll_stabilization(w, nx, ny, dx, dy);
                break;
            case HallStabKind::NONE:
            default:
                break;
        }
        (void)dt_loc;
        perf::t_hyper += omp_get_wtime() - ts;
    };
    const auto timed_hall = [&](double dt_loc) {
        const double ts = omp_get_wtime();
        add_hall_correction(w, nx, ny, dt_loc, dx, dy);
        perf::t_hall += omp_get_wtime() - ts;
    };
    const auto timed_faraday = [&](double dt_loc) {
        const double ts = omp_get_wtime();
        update_faces_from_emf(nx, ny, dt_loc, dx, dy);
        sync_cell_centered_from_faces(w, nx, ny);
        perf::t_ct_faraday += omp_get_wtime() - ts;
    };

    if (N_sub == 1) {
        add_resistive_correction(w, nx, ny, dt, dx, dy);
        apply_stab(dt);
        timed_hall(dt);
        // Driven BC: overwrite y-boundary corner EMFs last so the boundary
        // Ez time integral is exactly driven_ez_·dt.
        if (driven_ez_ != 0.0) override_inflow_ez(nx, ny, driven_ez_);
        timed_faraday(dt);
    } else {
        // ------------------------------------------------------------------
        // N_sub > 1: operator-split sub-cycling.
        //   1. Apply the ideal Faraday update once with the full dt_hyp.
        //   2. Sub-cycle the non-ideal block N_sub times with dt_local.
        // face_.emf_z currently holds the ideal EMF from step A.
        // ------------------------------------------------------------------
        const double dt_local = dt / static_cast<double>(N_sub);

        std::fprintf(stdout,
            "[subcycle] t=%.6g dt_hyp=%.3e  dt_eta=%.3e dt_hall=%.3e"
            "  dt_sub=%.3e  N_sub=%d  dt_local=%.3e\n",
            current_t_, dt, dt_eta, dt_hall, dt_sub, N_sub, dt_local);

        // Step 1: ideal Faraday once with full dt. Driven BC injects the full
        // boundary Ez here; sub-steps zero it so the integral stays driven_ez_·dt.
        if (driven_ez_ != 0.0) override_inflow_ez(nx, ny, driven_ez_);
        timed_faraday(dt);

        // Step 2: non-ideal sub-cycling on the current face B.
        for (int sub = 0; sub < N_sub; ++sub) {
            // Zero corner EMF: each sub-step accumulates non-ideal terms only.
            tw = omp_get_wtime();
            for (int I = 0; I <= nx; ++I)
                std::fill(face_.emf_z[I].begin(), face_.emf_z[I].end(), 0.0);
            perf::t_ct_faraday += omp_get_wtime() - tw;

            add_resistive_correction(w, nx, ny, dt_local, dx, dy);
            apply_stab(dt_local);
            timed_hall(dt_local);
            if (driven_ez_ != 0.0) override_inflow_ez(nx, ny, 0.0);
            timed_faraday(dt_local);
        }
    }

    // ψ is not used by CT.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 2; i < nx + 2; ++i)
        for (int j = 2; j < ny + 2; ++j)
            w[i][j][8] = 0.0;
}

// ---------------------------------------------------------------------------
// CT face-B access
// ---------------------------------------------------------------------------
void CTDivergenceControl::fill_face_bn_x(int j, int n, double* buf) const {
    buf[0] = 0.0;
    for (int i = 1; i <= n + 1; ++i) buf[i] = face_.bx[i - 1][j];
}

void CTDivergenceControl::fill_face_bn_y(int i, int n, double* buf) const {
    buf[0] = 0.0;
    for (int j = 1; j <= n + 1; ++j) buf[j] = face_.by[i][j - 1];
}

// ---------------------------------------------------------------------------
// CT EMF storage
// ---------------------------------------------------------------------------
void CTDivergenceControl::store_emf_x(int j, int n, const double* emf,
                                       const double* cemf, const double* sgn) {
    for (int i = 1; i <= n + 1; ++i) {
        emf_x_[i - 1][j] = emf[i];
        if (cemf) cemf_x_[i - 1][j] = cemf[i];
        if (sgn)  sgn_x_ [i - 1][j] = sgn [i];
    }
}

void CTDivergenceControl::store_emf_y(int i, int n, const double* emf,
                                       const double* cemf, const double* sgn) {
    for (int j = 1; j <= n + 1; ++j) {
        emf_y_[i][j - 1] = emf[j];
        if (cemf) cemf_y_[i][j - 1] = cemf[j];
        if (sgn)  sgn_y_ [i][j - 1] = sgn [j];
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

// Analytically sample face-centered B from the known test-problem field.
// For test problems where the field is not an analytic formula (fallback = NaN),
// bootstrap from cell-centred averages instead.
void CTDivergenceControl::initialize_faces_from_problem(const Grid& w, const RunConfig& cfg,
                                                        double dx, double dy) {
    const auto sample_bx = [&](double x, double y) -> double {
        switch (cfg.test) {
            case 0:  return 0.0;
            case 1:  (void)y; return 0.75;
            case 2: { const double m = 0.5*(cfg.y0+cfg.y1); return (y<=m)?1.0:-1.0; }
            case 3: { constexpr double pi=3.14159265358979323846; (void)x;
                      return -std::sin(2.0*pi*y); }
            case 4: { constexpr double pi=3.14159265358979323846; (void)x;(void)y;
                      return 2.5/std::sqrt(4.0*pi); }
            case 11: case 12:
            case 19: case 20: case 21: case 22: case 23:
                return harris_bx_face(x, y, dy, HarrisSheetParams{});
            case 25: case 26: {
                AsymHarrisParams ap25_bx;
                ap25_bx.psi0 = cfg.psi0;
                return asym_bx_face(x, y, dy, ap25_bx);
            }
            case 27: case 28: case 30: return asym_bx_face(x, y, dy, asym_scan_params(cfg));
            case 29: return dh_bx_face(x, y, dy, dh_params_from_config(cfg));
            case 31: return adh_bx_face(x, y, dy, adh_params_from_config(cfg));
            case 15: {
                // 45° Alfvén wave: bx from Az = (y−x)/√2 + c·cos(2π(x+y)), c = 0.1/(2π√2)
                constexpr double pi = 3.14159265358979323846;
                constexpr double s2 = 1.41421356237309504880;
                const double c = 0.1 / (2.0 * pi * s2);
                return 1.0/s2 - 2.0*c * std::sin(2.0*pi*(x+y)) * std::sin(pi*dy) / dy;
            }
            default: return std::numeric_limits<double>::quiet_NaN();
        }
    };
    const auto sample_by = [&](double x, double y) -> double {
        switch (cfg.test) {
            case 0:  return 0.0;
            case 1: { const double m=0.5*(cfg.x0+cfg.x1); (void)y;
                      return (x<=m)?1.0:-1.0; }
            case 2:  (void)x;(void)y; return 0.75;
            case 3: { constexpr double pi=3.14159265358979323846; (void)y;
                      return std::sin(4.0*pi*x); }
            case 4:  (void)x;(void)y; return 0.0;
            case 11: case 12:
            case 19: case 20: case 21: case 22: case 23:
                return harris_by_face(x, y, dx, HarrisSheetParams{});
            case 25: case 26: {
                AsymHarrisParams ap25_by;
                ap25_by.psi0 = cfg.psi0;
                return asym_by_face(x, y, dx, ap25_by);
            }
            case 27: case 28: case 30: return asym_by_face(x, y, dx, asym_scan_params(cfg));
            case 29: return dh_by_face(x, y, dx, dh_params_from_config(cfg));
            case 31: return adh_by_face(x, y, dx, adh_params_from_config(cfg));
            case 15: {
                // 45° Alfvén wave: by from Az = (y−x)/√2 + c·cos(2π(x+y)), c = 0.1/(2π√2)
                constexpr double pi = 3.14159265358979323846;
                constexpr double s2 = 1.41421356237309504880;
                const double c = 0.1 / (2.0 * pi * s2);
                return 1.0/s2 + 2.0*c * std::sin(2.0*pi*(x+y)) * std::sin(pi*dx) / dx;
            }
            default: return std::numeric_limits<double>::quiet_NaN();
        }
    };

    for (int i = 0; i < nx_ + 1; ++i) {
        for (int j = 0; j < ny_; ++j) {
            double bx = sample_bx(cfg.x0 + i*dx, cfg.y0 + (j+0.5)*dy);
            if (std::isnan(bx)) { fill_faces_from_cell_centered(w, nx_, ny_); return; }
            face_.bx[i][j] = bx;
        }
    }
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_ + 1; ++j) {
            double by = sample_by(cfg.x0 + (i+0.5)*dx, cfg.y0 + j*dy);
            if (std::isnan(by)) { fill_faces_from_cell_centered(w, nx_, ny_); return; }
            face_.by[i][j] = by;
        }
    }
}

// Bootstrap face B from cell-centred values (ghost cells must be valid).
void CTDivergenceControl::fill_faces_from_cell_centered(const Grid& w, int nx, int ny) {
    face_.resize(nx, ny);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx + 1; ++i)
        for (int j = 0; j < ny; ++j)
            face_.bx[i][j] = 0.5*(w[i+1][j+2][5] + w[i+2][j+2][5]);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny + 1; ++j)
            face_.by[i][j] = 0.5*(w[i+2][j+1][6] + w[i+2][j+2][6]);
}

// ---------------------------------------------------------------------------
// Resistive correction: add η·Jz to corner EMF; Ohmic heating (γ−1)·η·Jz²·dt
// to cell pressure (Jz_cell = mean of 4 corners).
// Corner current [Tóth 2000 §5.1]:
//   Jz[I][J] = (By[I][J] − By[I−1][J])/Δx − (Bx[I][J] − Bx[I][J−1])/Δy
// CT + resistivity: Balsara & Spicer 1999, JCP 149, 270 (§4).
// ---------------------------------------------------------------------------
void CTDivergenceControl::add_resistive_correction(Grid& w, int nx, int ny,
                                                   double dt, double dx, double dy) {
    if (eta_ <= 0.0) return;

    // Step 1: Jz at every corner from face-centred B (pre-Faraday).
    ScalarField Jz(nx + 1, std::vector<double>(ny + 1, 0.0));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int J = 0; J <= ny; ++J) {
            // ∂By/∂x: periodic wraps; transmissive clamps (0 at boundary corners).
            double byR, byL;
            if (bcx_ == BC::Periodic) {
                int Ir = (I < nx) ? I      : 0;
                int Il = (I > 0)  ? I - 1  : nx - 1;
                byR = face_.by[Ir][J];
                byL = face_.by[Il][J];
            } else {
                byR = face_.by[(I < nx) ? I     : nx - 1][J];
                byL = face_.by[(I > 0)  ? I - 1 : 0     ][J];
            }

            // ∂Bx/∂y
            double bxU, bxD;
            if (bcy_ == BC::Periodic) {
                int Ju = (J < ny) ? J      : 0;
                int Jd = (J > 0)  ? J - 1  : ny - 1;
                bxU = face_.bx[I][Ju];
                bxD = face_.bx[I][Jd];
            } else {
                bxU = face_.bx[I][(J < ny) ? J     : ny - 1];
                bxD = face_.bx[I][(J > 0)  ? J - 1 : 0     ];
            }

            Jz[I][J] = (byR - byL) / dx - (bxU - bxD) / dy;

            face_.emf_z[I][J] += eta_ * Jz[I][J];
        }
    }

    // Step 2: Ohmic heating — add (γ−1)·η·Jz_cell²·dt to cell pressure.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double Jc = 0.25 * (Jz[i][j] + Jz[i+1][j] + Jz[i][j+1] + Jz[i+1][j+1]);
            w[i + 2][j + 2][4] += (gamma_ - 1.0) * eta_ * Jc * Jc * dt;
        }
    }
}

// Hyper-resistive correction: add −η_H·∇²Jz to every corner EMF, giving
// ∂B/∂t = −η_H·∇⁴B (biharmonic, k⁴-selective grid-noise damping).
// Explicit stability dt ≤ h⁴/(32·η_H) is enforced by compute_dt.
void CTDivergenceControl::add_hyper_resistive_correction(int nx, int ny,
                                                         double dx, double dy) {
    if (eta_H_ <= 0.0) return;

    // Step 1: Jz at every corner (same stencil as add_resistive_correction).
    ScalarField Jz(nx + 1, std::vector<double>(ny + 1, 0.0));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int J = 0; J <= ny; ++J) {
            double byR, byL;
            if (bcx_ == BC::Periodic) {
                int Ir = (I < nx) ? I      : 0;
                int Il = (I > 0)  ? I - 1  : nx - 1;
                byR = face_.by[Ir][J];
                byL = face_.by[Il][J];
            } else {
                byR = face_.by[(I < nx) ? I     : nx - 1][J];
                byL = face_.by[(I > 0)  ? I - 1 : 0     ][J];
            }
            double bxU, bxD;
            if (bcy_ == BC::Periodic) {
                int Ju = (J < ny) ? J      : 0;
                int Jd = (J > 0)  ? J - 1  : ny - 1;
                bxU = face_.bx[I][Ju];
                bxD = face_.bx[I][Jd];
            } else {
                bxU = face_.bx[I][(J < ny) ? J     : ny - 1];
                bxD = face_.bx[I][(J > 0)  ? J - 1 : 0     ];
            }
            Jz[I][J] = (byR - byL) / dx - (bxU - bxD) / dy;
        }
    }

    // Step 2: ∇²Jz via 5-point stencil; add −η_H·∇²Jz to corner EMF.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int J = 0; J <= ny; ++J) {
            int Im, Ip, Jm, Jp;
            if (bcx_ == BC::Periodic) {
                Im = (I > 0)  ? I - 1 : nx - 1;
                Ip = (I < nx) ? I + 1 : 1;
            } else {
                Im = (I > 0)  ? I - 1 : 0;
                Ip = (I < nx) ? I + 1 : nx;
            }
            if (bcy_ == BC::Periodic) {
                Jm = (J > 0)  ? J - 1 : ny - 1;
                Jp = (J < ny) ? J + 1 : 1;
            } else {
                Jm = (J > 0)  ? J - 1 : 0;
                Jp = (J < ny) ? J + 1 : ny;
            }
            double lap = (Jz[Ip][J] - 2.0*Jz[I][J] + Jz[Im][J]) / (dx*dx)
                       + (Jz[I][Jp] - 2.0*Jz[I][J] + Jz[I][Jm]) / (dy*dy);
            face_.emf_z[I][J] -= eta_H_ * lap;
        }
    }
}

// Corner EMF via CT-Contact (Gardiner & Stone 2005 §4, eq. A9-A11):
//   E_CT = E_arithm + (1/8)(s_S−s_N)(δ_W−δ_E) + (1/8)(s_W−s_E)(δ_S−δ_N)
// with δ = E_numerical − E_centered and s = sign(mass flux) at each face.
// Periodic BC: boundary EMFs are averaged first so emf_z[0] == emf_z[nx].
void CTDivergenceControl::compute_corner_emf_from_interface_emfs(int nx, int ny) {
    if (bcx_ == BC::Periodic) {
        for (int j = 0; j < ny; ++j) {
            double avg;
            avg = 0.5*(emf_x_[0][j]  + emf_x_[nx][j]);
            emf_x_[0][j]  = avg;  emf_x_[nx][j]  = avg;
            avg = 0.5*(cemf_x_[0][j] + cemf_x_[nx][j]);
            cemf_x_[0][j] = avg;  cemf_x_[nx][j] = avg;
            sgn_x_[nx][j] = sgn_x_[0][j];
        }
    }
    if (bcy_ == BC::Periodic) {
        for (int i = 0; i < nx; ++i) {
            double avg;
            avg = 0.5*(emf_y_[i][0]  + emf_y_[i][ny]);
            emf_y_[i][0]  = avg;  emf_y_[i][ny]  = avg;
            avg = 0.5*(cemf_y_[i][0] + cemf_y_[i][ny]);
            cemf_y_[i][0] = avg;  cemf_y_[i][ny] = avg;
            sgn_y_[i][ny] = sgn_y_[i][0];
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I < nx + 1; ++I) {
        for (int J = 0; J < ny + 1; ++J) {
            int Jm, Jp, Im, Ip;
            if (bcy_ == BC::Periodic) {
                Jm = (J > 0)  ? J-1 : ny-1;
                Jp = (J < ny) ? J   : 0;
            } else {
                Jm = (J > 0)  ? J-1 : 0;
                Jp = (J < ny) ? J   : ny-1;
            }
            if (bcx_ == BC::Periodic) {
                Im = (I > 0)  ? I-1 : nx-1;
                Ip = (I < nx) ? I   : 0;
            } else {
                Im = (I > 0)  ? I-1 : 0;
                Ip = (I < nx) ? I   : nx-1;
            }

            double E_arithm = 0.25*(emf_x_[I][Jm] + emf_x_[I][Jp]
                                  + emf_y_[Im][J]  + emf_y_[Ip][J]);

            double dS = emf_x_[I][Jm]  - cemf_x_[I][Jm];
            double dN = emf_x_[I][Jp]  - cemf_x_[I][Jp];
            double dW = emf_y_[Im][J]  - cemf_y_[Im][J];
            double dE = emf_y_[Ip][J]  - cemf_y_[Ip][J];

            double sS = sgn_x_[I][Jm];
            double sN = sgn_x_[I][Jp];
            double sW = sgn_y_[Im][J];
            double sE = sgn_y_[Ip][J];

            face_.emf_z[I][J] = E_arithm
                + 0.125*(sS - sN)*(dW - dE)
                + 0.125*(sW - sE)*(dS - dN);
        }
    }
}

// Driven BC: overwrite both y-boundary corner-EMF rows with a constant Ez.
// Ez < 0 gives E×B inflow on both sides at equal flux-supply rate |Ez|.
// CT's discrete ∇·B = 0 is algebraic in the corner EMF, so the overwrite is
// safe; constant Ez along the row means ΔBy = 0 (no tangential pollution).
void CTDivergenceControl::override_inflow_ez(int nx, int ny, double value) {
    for (int I = 0; I <= nx; ++I) {
        face_.emf_z[I][0]  = value;
        face_.emf_z[I][ny] = value;
    }
}

// Advance face B via discrete Faraday:
//   ΔBx[I][J] = -(dt/dy)*(Ez[I][J+1] - Ez[I][J])
//   ΔBy[I][J] =  (dt/dx)*(Ez[I+1][J] - Ez[I][J])
void CTDivergenceControl::update_faces_from_emf(int nx, int ny,
                                                double dt, double dx, double dy) {
    ScalarField new_bx = face_.bx;
    ScalarField new_by = face_.by;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx + 1; ++i)
        for (int j = 0; j < ny; ++j)
            new_bx[i][j] -= (dt/dy)*(face_.emf_z[i][j+1] - face_.emf_z[i][j]);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny + 1; ++j)
            new_by[i][j] += (dt/dx)*(face_.emf_z[i+1][j] - face_.emf_z[i][j]);
    face_.bx.swap(new_bx);
    face_.by.swap(new_by);
}

// Overwrite cell-centred Bx and By with face averages.
// Total energy and pressure are left unchanged: the O(Δt²) inconsistency
// is harmless and avoiding it prevents spurious pressure floors.
void CTDivergenceControl::sync_cell_centered_from_faces(Grid& w, int nx, int ny) const {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            w[i+2][j+2][5] = 0.5*(face_.bx[i][j]   + face_.bx[i+1][j]);
            w[i+2][j+2][6] = 0.5*(face_.by[i][j]   + face_.by[i][j+1]);
        }
    }
}

// Enforce face BC. Called only if external code needs to re-establish BC
// after directly modifying face fields; NOT called from post_step.
void CTDivergenceControl::apply_face_bc(int nx, int ny) {
    if (bcx_ == BC::Periodic) {
        for (int j = 0; j < ny; ++j) {
            double avg = 0.5*(face_.bx[0][j] + face_.bx[nx][j]);
            face_.bx[0][j] = avg;  face_.bx[nx][j] = avg;
        }
    } else {
        for (int j = 0; j < ny; ++j) {
            face_.bx[0][j]  = face_.bx[1][j];
            face_.bx[nx][j] = face_.bx[nx-1][j];
        }
    }
    if (bcy_ == BC::Periodic) {
        for (int i = 0; i < nx; ++i) {
            double avg = 0.5*(face_.by[i][0] + face_.by[i][ny]);
            face_.by[i][0] = avg;  face_.by[i][ny] = avg;
        }
    } else {
        for (int i = 0; i < nx; ++i) {
            face_.by[i][0]  = face_.by[i][1];
            face_.by[i][ny] = face_.by[i][ny-1];
        }
    }
}

// ---------------------------------------------------------------------------
// Hall correction: ∂B/∂t|_Hall = −∇×[(d_i/ρ) J×B], integrated with RK4.
// Forward Euler is unconditionally unstable for whistlers (pure dispersion);
// RK4 is stable for |ω·dt| < 2√2, guaranteed by the Hall CFL in compute_dt.
// Reference: Tóth et al. 2008, JCP 227, 6967.
// ---------------------------------------------------------------------------

// Pure helper: Hall corner EMF (emfz_out) and cell dBz/dt (Bz_rate_out) from
// face B and cell Bz. Density from w (fixed during RK4 stages). No class state.
static void hall_stage(
    const ScalarField& bx_in,   // (nx+1)×ny  — face Bx
    const ScalarField& by_in,   // nx×(ny+1)  — face By
    const ScalarField& Bz_in,   // nx×ny      — cell-centred Bz (0-indexed)
    const Grid&        w,       // density at w[i+2][j+2][0]
    int nx, int ny, double dx, double dy,
    double hall_di, BC bcx, BC bcy, double rho_floor,
    ScalarField& emfz_out,      // (nx+1)×(ny+1) — E_z^Hall
    ScalarField& Bz_rate_out    // nx×ny         — dBz/dt
) {
    // ponytail: static scratch — serial caller only, fully overwritten each call.
    static ScalarField Jx_c, Jy_c, Jz_c, ExH, EyH;
    auto ensure = [](ScalarField& f, int n1, int n2) {
        if ((int)f.size() != n1 || (int)f[0].size() != n2)
            f.assign(n1, std::vector<double>(n2, 0.0));
    };
    ensure(Jx_c, nx+1, ny+1); ensure(Jy_c, nx+1, ny+1); ensure(Jz_c, nx+1, ny+1);
    ensure(ExH, nx, ny+1);    ensure(EyH, nx+1, ny);
    ensure(emfz_out, nx+1, ny+1); ensure(Bz_rate_out, nx, ny);

    // Corner currents and Hall Ez in one pass.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int J = 0; J <= ny; ++J) {
            // Jz = (By_R − By_L)/dx − (Bx_U − Bx_D)/dy
            double byR, byL, bxU, bxD;
            if (bcx == BC::Periodic) {
                byR = by_in[(I < nx) ? I     : 0     ][J];
                byL = by_in[(I > 0)  ? I - 1 : nx - 1][J];
            } else {
                byR = by_in[(I < nx) ? I     : nx-1][J];
                byL = by_in[(I > 0)  ? I - 1 : 0   ][J];
            }
            if (bcy == BC::Periodic) {
                bxU = bx_in[I][(J < ny) ? J     : 0     ];
                bxD = bx_in[I][(J > 0)  ? J - 1 : ny - 1];
            } else {
                bxU = bx_in[I][(J < ny) ? J     : ny-1];
                bxD = bx_in[I][(J > 0)  ? J - 1 : 0   ];
            }
            Jz_c[I][J] = (byR - byL) / dx - (bxU - bxD) / dy;

            // Jx = ∂Bz/∂y, Jy = −∂Bz/∂x  (BC-aware index clamping/wrapping)
            int iL = (bcx == BC::Periodic) ? ((I > 0) ? I-1 : nx-1) : std::max(I-1, 0);
            int iR = (bcx == BC::Periodic) ? (I % nx)                : std::min(I,   nx-1);
            int jD = (bcy == BC::Periodic) ? ((J > 0) ? J-1 : ny-1) : std::max(J-1, 0);
            int jU = (bcy == BC::Periodic) ? (J % ny)                : std::min(J,   ny-1);

            double Bz_up    = 0.5*(Bz_in[iL][jU] + Bz_in[iR][jU]);
            double Bz_down  = 0.5*(Bz_in[iL][jD] + Bz_in[iR][jD]);
            double Bz_right = 0.5*(Bz_in[iR][jD] + Bz_in[iR][jU]);
            double Bz_left  = 0.5*(Bz_in[iL][jD] + Bz_in[iL][jU]);
            Jx_c[I][J] =  (Bz_up   - Bz_down ) / dy;
            Jy_c[I][J] = -(Bz_right - Bz_left) / dx;

            // Hall Ez at this corner
            double rho_c = 0.25*(w[I+1][J+1][0] + w[I+2][J+1][0] +
                                 w[I+1][J+2][0] + w[I+2][J+2][0]);
            rho_c = std::max(rho_c, rho_floor);

            int jD_c = (bcy == BC::Periodic) ? ((J > 0) ? J-1 : ny-1) : std::max(J-1, 0);
            int jU_c = (bcy == BC::Periodic) ? (J % ny)                : std::min(J, ny-1);
            double Bx_c = 0.5*(bx_in[I][jD_c] + bx_in[I][jU_c]);

            int iL_c = (bcx == BC::Periodic) ? ((I > 0) ? I-1 : nx-1) : std::max(I-1, 0);
            int iR_c = (bcx == BC::Periodic) ? (I % nx)                : std::min(I, nx-1);
            double By_c = 0.5*(by_in[iL_c][J] + by_in[iR_c][J]);

            emfz_out[I][J] = (hall_di / rho_c)
                             * (Jx_c[I][J]*By_c - Jy_c[I][J]*Bx_c);
        }
    }

    // ExH at y-faces (i,J) and EyH at x-faces (I,j) → dBz/dt
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int J = 0; J <= ny; ++J) {
            int jD_f = std::max(J-1, 0);
            int jU_f = std::min(J, ny-1);
            double rho_f = 0.5*(w[i+2][jD_f+2][0] + w[i+2][jU_f+2][0]);
            rho_f = std::max(rho_f, rho_floor);
            double Bz_f = 0.5*(Bz_in[i][jD_f] + Bz_in[i][jU_f]);
            double By_f = by_in[i][J];
            double Jy_f = 0.5*(Jy_c[i][J] + Jy_c[i+1][J]);
            double Jz_f = 0.5*(Jz_c[i][J] + Jz_c[i+1][J]);
            ExH[i][J] = (hall_di / rho_f) * (Jy_f*Bz_f - Jz_f*By_f);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int j = 0; j < ny; ++j) {
            int iL_f = (bcx == BC::Periodic) ? ((I > 0) ? I-1 : nx-1) : std::max(I-1, 0);
            int iR_f = (bcx == BC::Periodic) ? (I % nx)                : std::min(I, nx-1);
            double rho_f = 0.5*(w[iL_f+2][j+2][0] + w[iR_f+2][j+2][0]);
            rho_f = std::max(rho_f, rho_floor);
            double Bz_f = 0.5*(Bz_in[iL_f][j] + Bz_in[iR_f][j]);
            double Bx_f = bx_in[I][j];
            double Jx_f = 0.5*(Jx_c[I][j] + Jx_c[I][j+1]);
            double Jz_f = 0.5*(Jz_c[I][j] + Jz_c[I][j+1]);
            EyH[I][j] = (hall_di / rho_f) * (Jz_f*Bx_f - Jx_f*Bz_f);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Bz_rate_out[i][j] = (ExH[i][j+1] - ExH[i][j]) / dy
                               - (EyH[i+1][j] - EyH[i][j]) / dx;
}

// Apply a Faraday advance of bx/by from a Hall EMF array (one RK4 step).
// bx_out[I][j] = bx_in[I][j] - fac * (emfz[I][j+1] - emfz[I][j]) / dy
// by_out[i][J] = by_in[i][J] + fac * (emfz[i+1][J] - emfz[i][J]) / dx
static void faraday_advance(
    const ScalarField& bx_in, const ScalarField& by_in,
    const ScalarField& emfz,
    int nx, int ny, double dx, double dy, double fac,
    ScalarField& bx_out, ScalarField& by_out
) {
    // Writes outputs directly (no copy); same expressions, same rounding.
    if ((int)bx_out.size() != nx+1 || (int)bx_out[0].size() != ny)
        bx_out.assign(nx+1, std::vector<double>(ny, 0.0));
    if ((int)by_out.size() != nx || (int)by_out[0].size() != ny+1)
        by_out.assign(nx, std::vector<double>(ny+1, 0.0));
    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static) nowait
        for (int I = 0; I <= nx; ++I)
            for (int j = 0; j < ny; ++j)
                bx_out[I][j] = bx_in[I][j] - fac * (emfz[I][j+1] - emfz[I][j]) / dy;
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < nx; ++i)
            for (int J = 0; J <= ny; ++J)
                by_out[i][J] = by_in[i][J] + fac * (emfz[i+1][J] - emfz[i][J]) / dx;
    }
}

void CTDivergenceControl::add_hall_correction(Grid& w, int nx, int ny,
                                              double dt, double dx, double dy) {
    if (hall_di_ <= 0.0) return;

    // ponytail: static RK4 scratch — serial caller only, fully overwritten.
    static ScalarField Bz0, emfz1, emfz2, emfz3, emfz4,
                       kBz1, kBz2, kBz3, kBz4, bx_tmp, by_tmp, Bz_tmp;
    if ((int)Bz0.size() != nx || (int)Bz0[0].size() != ny) {
        Bz0.assign(nx, std::vector<double>(ny, 0.0));
        Bz_tmp.assign(nx, std::vector<double>(ny, 0.0));
    }

    // Extract initial cell-centred Bz into a 0-indexed array
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Bz0[i][j] = w[i+2][j+2][7];

    // Stage 1: rates at (bx_0, by_0, Bz_0)
    hall_stage(face_.bx, face_.by, Bz0, w, nx, ny, dx, dy,
               hall_di_, bcx_, bcy_, rho_floor_, emfz1, kBz1);

    // Stage 2: half-step advance from stage 1
    faraday_advance(face_.bx, face_.by, emfz1, nx, ny, dx, dy, dt*0.5, bx_tmp, by_tmp);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Bz_tmp[i][j] = Bz0[i][j] + (dt*0.5) * kBz1[i][j];
    hall_stage(bx_tmp, by_tmp, Bz_tmp, w, nx, ny, dx, dy,
               hall_di_, bcx_, bcy_, rho_floor_, emfz2, kBz2);

    // Stage 3: half-step advance from stage 2
    faraday_advance(face_.bx, face_.by, emfz2, nx, ny, dx, dy, dt*0.5, bx_tmp, by_tmp);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Bz_tmp[i][j] = Bz0[i][j] + (dt*0.5) * kBz2[i][j];
    hall_stage(bx_tmp, by_tmp, Bz_tmp, w, nx, ny, dx, dy,
               hall_di_, bcx_, bcy_, rho_floor_, emfz3, kBz3);

    // Stage 4: full-step advance from stage 3
    faraday_advance(face_.bx, face_.by, emfz3, nx, ny, dx, dy, dt, bx_tmp, by_tmp);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Bz_tmp[i][j] = Bz0[i][j] + dt * kBz3[i][j];
    hall_stage(bx_tmp, by_tmp, Bz_tmp, w, nx, ny, dx, dy,
               hall_di_, bcx_, bcy_, rho_floor_, emfz4, kBz4);

    // Apply RK4 weighted average: add Hall EMF to face_.emf_z (used by
    // update_faces_from_emf to advance face Bx, By via Faraday law).
    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I)
        for (int J = 0; J <= ny; ++J)
            face_.emf_z[I][J] += (emfz1[I][J] + 2.0*emfz2[I][J]
                                 + 2.0*emfz3[I][J] + emfz4[I][J]) / 6.0;

    // Apply RK4 Bz update directly to cell-centred state
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            w[i+2][j+2][7] += dt * (kBz1[i][j] + 2.0*kBz2[i][j]
                                   + 2.0*kBz3[i][j] + kBz4[i][j]) / 6.0;
}

// ---------------------------------------------------------------------------
// Hall-HLL stabilization (Path B): first-order upwind diffusion at whistler
// speed c_w = π·di·|B|/(ρ·mincell), added to the corner EMF:
//   δEz = +(c_stab/2)(By_right − By_left) − (c_stab/2)(Bx_above − Bx_below)
// Approximates whistlers as pure diffusion (overdamped, γ/ω ≈ π/2), per
// Iwasaki & Tomida 2025. Stability c_stab·dt/h ≤ 1 enforced by compute_dt.
// ---------------------------------------------------------------------------
void CTDivergenceControl::add_hall_hll_stabilization(Grid& w, int nx, int ny,
                                                     double dx, double dy) {
    if (hall_di_ <= 0.0) return;
    constexpr double pi   = 3.14159265358979323846;
    const double mincell  = std::min(dx, dy);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int J = 0; J <= ny; ++J) {
            // Corner (I,J) averages the 4 surrounding cells (2 ghost layers).
            double rho_c = 0.25 * (w[I+1][J+1][0] + w[I+2][J+1][0]
                                 + w[I+1][J+2][0] + w[I+2][J+2][0]);
            rho_c = std::max(rho_c, 0.1);  // matches Hall CFL floor

            double Bx_c = 0.25 * (w[I+1][J+1][5] + w[I+2][J+1][5]
                                 + w[I+1][J+2][5] + w[I+2][J+2][5]);
            double By_c = 0.25 * (w[I+1][J+1][6] + w[I+2][J+1][6]
                                 + w[I+1][J+2][6] + w[I+2][J+2][6]);
            double Bz_c = 0.25 * (w[I+1][J+1][7] + w[I+2][J+1][7]
                                 + w[I+1][J+2][7] + w[I+2][J+2][7]);
            double B2_c = Bx_c*Bx_c + By_c*By_c + Bz_c*Bz_c;

            // Fast-speed bound c_f² = (γp + B²)/ρ: physical nonzero floor at
            // magnetic nulls; negligible vs c_w in strong-field regions.
            double p_c = 0.25 * (w[I+1][J+1][4] + w[I+2][J+1][4]
                               + w[I+1][J+2][4] + w[I+2][J+2][4]);
            p_c = std::max(p_c, 0.0);
            const double c_f_c = std::sqrt((gamma_ * p_c + B2_c) / rho_c);

            // Whistler speed; vanishes honestly at magnetic nulls.
            const double c_w = pi * hall_di_ * std::sqrt(B2_c) / (rho_c * mincell);

            const double c_stab = c_f_c + c_w;

            // By faces left/right of corner (I,J)
            int i_right = (bcx_ == BC::Periodic) ? (I < nx ? I : 0)
                                                  : (I < nx ? I : nx-1);
            int i_left  = (bcx_ == BC::Periodic) ? (I > 0 ? I-1 : nx-1)
                                                  : (I > 0 ? I-1 : 0);
            double jump_by = face_.by[i_right][J] - face_.by[i_left][J];

            // Bx faces above/below corner (I,J)
            int j_above = (bcy_ == BC::Periodic) ? (J < ny ? J : 0)
                                                  : (J < ny ? J : ny-1);
            int j_below = (bcy_ == BC::Periodic) ? (J > 0 ? J-1 : ny-1)
                                                  : (J > 0 ? J-1 : 0);
            double jump_bx = face_.bx[I][j_above] - face_.bx[I][j_below];

            face_.emf_z[I][J] += (c_stab * 0.5) * jump_by - (c_stab * 0.5) * jump_bx;
        }
    }
}

} // namespace my_project
