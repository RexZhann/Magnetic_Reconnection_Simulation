#include "my_project/divergence_control.hpp"
#include "my_project/harris_sheet.hpp"

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
    compute_corner_emf_from_interface_emfs(nx, ny);

    // ------------------------------------------------------------------
    // Determine sub-cycle count for the non-ideal block.
    // N_sub == 1 → original single-step path, bitwise identical.
    // ------------------------------------------------------------------
    int    N_sub   = 1;
    double dt_sub  = dt;
    double dt_eta  = std::numeric_limits<double>::max();
    double dt_hall = std::numeric_limits<double>::max();

    if (subcycle_nonideal_ && (eta_ > 0.0 || hall_di_ > 0.0)) {
        constexpr double pi = 3.14159265358979323846;
        double mincell = std::min(dx, dy);

        // Resistive (parabolic) CFL: dt_eta = mincell² / (2 η)
        if (eta_ > 0.0)
            dt_eta = mincell * mincell / (2.0 * eta_);

        // 霍尔稳定化子步 CFL —— 根据方案选择不同公式：
        //   HYPER_RES / NONE : RK4 稳定性，使用 vA = |B|/√ρ
        //   HALL_HLL         : 一阶迎风扩散，使用 |B|/ρ（更严格）
        if (hall_di_ > 0.0) {
            if (hall_stab_ == HallStabKind::HALL_HLL) {
                // HLL 稳定性条件：c_w·dt/h ≤ 1，c_w = π·di·|B|/(ρ·h)
                // → dt = h² / (π·di·max(|B|/ρ))
                double max_b2_rho2 = 1e-14;
                #pragma omp parallel for collapse(2) reduction(max:max_b2_rho2) schedule(static)
                for (int i = 2; i < nx + 2; ++i)
                    for (int j = 2; j < ny + 2; ++j) {
                        double rho = std::max(w[i][j][0], 0.1);
                        double B2  = w[i][j][5]*w[i][j][5]
                                   + w[i][j][6]*w[i][j][6]
                                   + w[i][j][7]*w[i][j][7];
                        max_b2_rho2 = std::max(max_b2_rho2, B2 / (rho * rho));
                    }
                dt_hall = cfl_ * mincell * mincell
                          / (pi * hall_di_ * std::sqrt(max_b2_rho2));
            } else {
                // RK4 稳定性（|ω·dt| < 2√2 ≈ 2.83），使用 vA = |B|/√ρ
                double max_va2 = 1e-14;
                #pragma omp parallel for collapse(2) reduction(max:max_va2) schedule(static)
                for (int i = 2; i < nx + 2; ++i)
                    for (int j = 2; j < ny + 2; ++j) {
                        double rho = std::max(w[i][j][0], 0.1);
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

    // ------------------------------------------------------------------
    // N_sub == 1: original single-step path (bitwise identical to old code).
    // ------------------------------------------------------------------
    // 根据稳定化方案选择调用哪个函数（单步和子循环路径均使用此 lambda）
    const auto apply_stab = [&](double dt_loc) {
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
    };

    if (N_sub == 1) {
        add_resistive_correction(w, nx, ny, dt, dx, dy);
        apply_stab(dt);
        add_hall_correction(w, nx, ny, dt, dx, dy);
        update_faces_from_emf(nx, ny, dt, dx, dy);
        sync_cell_centered_from_faces(w, nx, ny);
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
            "  dt_sub=%.3e  N_sub=%d\n",
            current_t_, dt,
            (eta_     > 0.0) ? dt_eta  : -1.0,
            (hall_di_ > 0.0) ? dt_hall : -1.0,
            dt_sub, N_sub);

        // 步骤 1：理想 Faraday 推进（一次，全局 dt_hyp）
        update_faces_from_emf(nx, ny, dt, dx, dy);
        sync_cell_centered_from_faces(w, nx, ny);

        // 步骤 2：非理想子循环（每步使用当前面心 B）
        for (int sub = 0; sub < N_sub; ++sub) {
            // 清零角点 EMF：每子步仅累积非理想贡献
            for (int I = 0; I <= nx; ++I)
                std::fill(face_.emf_z[I].begin(), face_.emf_z[I].end(), 0.0);

            add_resistive_correction(w, nx, ny, dt_local, dx, dy);
            apply_stab(dt_local);
            add_hall_correction(w, nx, ny, dt_local, dx, dy);
            update_faces_from_emf(nx, ny, dt_local, dx, dy);
            sync_cell_centered_from_faces(w, nx, ny);
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
            case 19: case 20: case 21: case 22: case 23: {
                // Harris 电流片：由矢势线积分解析给出面心 Bx，初始 ∇·B = 0（机器精度）
                return harris_bx_face(x, y, dy, HarrisSheetParams{});
            }
            case 15: {
                // 45° Alfvén wave: Az = (y−x)/√2 + c·cos(2π(x+y)), c = 0.1/(2π√2)
                // bx = [Az(x,y+½Δy) − Az(x,y−½Δy)] / Δy
                //    = 1/√2 − 2c·sin(2π(x+y))·sin(π·Δy)/Δy
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
            case 19: case 20: case 21: case 22: case 23: {
                // Harris 电流片：由矢势线积分解析给出面心 By，初始 ∇·B = 0（机器精度）
                return harris_by_face(x, y, dx, HarrisSheetParams{});
            }
            case 15: {
                // 45° Alfvén wave: Az = (y−x)/√2 + c·cos(2π(x+y)), c = 0.1/(2π√2)
                // by = −[Az(x+½Δx,y) − Az(x−½Δx,y)] / Δx
                //    = 1/√2 + 2c·sin(2π(x+y))·sin(π·Δx)/Δx
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
// Resistive MHD correction: add η·Jz to corner EMF and η·Jz² (Ohmic heating)
// to cell thermal pressure.
//
// Discrete current at corner (I,J)  [Tóth 2000, §5.1]:
//   Jz[I][J] = (By[I][J] − By[I−1][J]) / Δx  −  (Bx[I][J] − Bx[I][J−1]) / Δy
// where By[I][J] = face_.by[I][J] (y-face to the right of corner),
//       Bx[I][J] = face_.bx[I][J] (x-face above the corner).
//
// Resistive EMF (induction equation with resistivity):
//   ∂B/∂t = ∇×(v×B) + η∇²B  ≡  −∇×E_ideal + ∇×(η J)
//   E_z_total = E_z_ideal + η Jz          ← added to face_.emf_z here
//
// Ohmic heating (energy equation):
//   ∂e_th/∂t += η Jz²   →   δp = (γ−1) η Jz_cell² δt
//   where Jz_cell = average of 4 surrounding corner Jz values.
//
// Reference for CT + resistivity:
//   Balsara, D.S., Spicer, D.S. (1999). "A staggered mesh algorithm using
//   high order Godunov fluxes to ensure solenoidal magnetic fields in MHD
//   simulations." J. Comput. Phys. 149, 270–292.  doi:10.1006/jcph.1998.6153
//   (Section 4 — resistive extension of CT)
// ---------------------------------------------------------------------------
void CTDivergenceControl::add_resistive_correction(Grid& w, int nx, int ny,
                                                   double dt, double dx, double dy) {
    if (eta_ <= 0.0) return;

    // -----------------------------------------------------------------------
    // Step 1: Jz at every corner (I,J) from face-centred B (pre-Faraday).
    // -----------------------------------------------------------------------
    ScalarField Jz(nx + 1, std::vector<double>(ny + 1, 0.0));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int J = 0; J <= ny; ++J) {
            // --- ∂By/∂x: right and left y-faces adjacent to corner (I,J) ---
            // face_.by[i][j] is the y-face at x=(i+0.5)dx, y=j·dy; i ∈ [0,nx−1]
            double byR, byL;
            if (bcx_ == BC::Periodic) {
                // Periodic: corner I and corner nx are the same physical point.
                int Ir = (I < nx) ? I      : 0;
                int Il = (I > 0)  ? I - 1  : nx - 1;
                byR = face_.by[Ir][J];
                byL = face_.by[Il][J];
            } else {
                // Transmissive: clamp to interior (gives 0 at boundary corners).
                byR = face_.by[(I < nx) ? I     : nx - 1][J];
                byL = face_.by[(I > 0)  ? I - 1 : 0     ][J];
            }

            // --- ∂Bx/∂y: upper and lower x-faces adjacent to corner (I,J) ---
            // face_.bx[i][j] is the x-face at x=i·dx, y=(j+0.5)dy; j ∈ [0,ny−1]
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

            // Add resistive contribution to the corner EMF that will be used
            // by update_faces_from_emf for the Faraday update.
            face_.emf_z[I][J] += eta_ * Jz[I][J];
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Ohmic heating — add (γ−1) η Jz_cell² dt to cell pressure.
    //   Jz at cell centre (i,j) = mean of 4 surrounding corner values.
    //   The pressure update is consistent with the Ohmic term in the
    //   total-energy equation:  ∂E/∂t += η Jz²  →  ∂p/∂t += (γ−1) η Jz².
    // -----------------------------------------------------------------------
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double Jc = 0.25 * (Jz[i][j] + Jz[i+1][j] + Jz[i][j+1] + Jz[i+1][j+1]);
            w[i + 2][j + 2][4] += (gamma_ - 1.0) * eta_ * Jc * Jc * dt;
        }
    }
}

// Hyper-resistive correction: add −η_H·∇²Jz to every corner EMF.
//
// With E_hyper = −η_H ∇²Jz and CT Faraday (∂B/∂t = −∇×E), the net effect
// on a div-B-free field is ∂B/∂t = −η_H ∇⁴B (4th-order biharmonic damping).
// This suppresses grid-scale magnetic noise with k⁴ selectivity.
//
// Algorithm:
//   1. Compute Jz at every corner using the same face-B stencil as add_resistive_correction.
//   2. Apply the 5-point Laplacian over corners: ∇²Jz ≈ (Jz[I+1,J]−2Jz[I,J]+Jz[I-1,J])/dx²
//      + (Jz[I,J+1]−2Jz[I,J]+Jz[I,J-1])/dy².  Neighbours wrap periodically / clamp.
//   3. face_.emf_z[I][J] −= η_H · ∇²Jz[I][J].
//
// Explicit stability: dt ≤ h⁴/(32·η_H) is enforced by compute_dt.
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

// Corner EMF via CT-Contact formula (Gardiner & Stone 2005, §4; eq. A9-A11).
//
//   E_CT = E_arithm
//        + (1/8)(s_S − s_N)(δ_W − δ_E)   [y-direction contact correction]
//        + (1/8)(s_W − s_E)(δ_S − δ_N)   [x-direction contact correction]
//
// where δ = E_numerical − E_centered (the upwind numerical dissipation),
// and s = sign(mass flux) at the relevant interface (±1).
// This steers the corner EMF toward the upwind side set by the contact wave
// while preserving ∇·B = 0 to machine precision.
//
// Notation for interfaces surrounding corner (I,J):
//   S = south x-face: emf_x_[I][Jm],  N = north x-face: emf_x_[I][Jp]
//   W = west  y-face: emf_y_[Im][J],  E = east  y-face: emf_y_[Ip][J]
//
// For periodic BC, boundary EMFs are averaged first so that
// emf_z[0][J] == emf_z[nx][J], ensuring identical Faraday increments on
// both copies of the periodic boundary face.
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

            // numerical dissipation δ = E_numerical − E_centered at each face
            double dS = emf_x_[I][Jm]  - cemf_x_[I][Jm];
            double dN = emf_x_[I][Jp]  - cemf_x_[I][Jp];
            double dW = emf_y_[Im][J]  - cemf_y_[Im][J];
            double dE = emf_y_[Ip][J]  - cemf_y_[Ip][J];

            // sign of mass flux at each surrounding face
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
// Hall MHD correction: ∂B/∂t|_Hall = -∇ × [(d_i/ρ) J × B]
//
// Integrated with 4-stage Runge-Kutta (RK4) for numerical stability.
//
// Forward Euler applied to Hall whistler waves is unconditionally unstable
// because Hall generates purely dispersive modes (|G| = sqrt(1+(ω·dt)²) > 1
// for any dt > 0).  RK4 is stable for purely imaginary eigenvalues when
// |ω·dt| < 2√2 ≈ 2.83; compute_dt guarantees this via the Hall CFL formula
// with the π² correction factor.
//
// Reference: Tóth et al. (2008) J. Comput. Phys. 227, 6967-6984.
// ---------------------------------------------------------------------------

// Pure helper: compute Hall EMF at corners (emfz_out) and dBz/dt at cells
// (Bz_rate_out) given face B fields (bx_in, by_in) and cell-centred Bz (Bz_in).
// Density is taken from w[i+2][j+2][0] (unchanged during Hall RK4 stages).
// Does NOT modify any class state.
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
    // Corner currents (I=0..nx, J=0..ny)
    ScalarField Jx_c(nx+1, std::vector<double>(ny+1, 0.0));
    ScalarField Jy_c(nx+1, std::vector<double>(ny+1, 0.0));
    ScalarField Jz_c(nx+1, std::vector<double>(ny+1, 0.0));

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
        }
    }

    // E_z^Hall at corners
    emfz_out.assign(nx+1, std::vector<double>(ny+1, 0.0));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int J = 0; J <= ny; ++J) {
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
    ScalarField ExH(nx,     std::vector<double>(ny+1, 0.0));
    ScalarField EyH(nx+1,   std::vector<double>(ny,   0.0));

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

    Bz_rate_out.assign(nx, std::vector<double>(ny, 0.0));
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
    bx_out = bx_in;
    by_out = by_in;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I)
        for (int j = 0; j < ny; ++j)
            bx_out[I][j] -= fac * (emfz[I][j+1] - emfz[I][j]) / dy;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i)
        for (int J = 0; J <= ny; ++J)
            by_out[i][J] += fac * (emfz[i+1][J] - emfz[i][J]) / dx;
}

void CTDivergenceControl::add_hall_correction(Grid& w, int nx, int ny,
                                              double dt, double dx, double dy) {
    if (hall_di_ <= 0.0) return;

    // Extract initial cell-centred Bz into a 0-indexed array
    ScalarField Bz0(nx, std::vector<double>(ny, 0.0));
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Bz0[i][j] = w[i+2][j+2][7];

    // Allocate RK4 stage arrays
    ScalarField emfz1(nx+1, std::vector<double>(ny+1, 0.0));
    ScalarField emfz2(nx+1, std::vector<double>(ny+1, 0.0));
    ScalarField emfz3(nx+1, std::vector<double>(ny+1, 0.0));
    ScalarField emfz4(nx+1, std::vector<double>(ny+1, 0.0));
    ScalarField kBz1(nx, std::vector<double>(ny, 0.0));
    ScalarField kBz2(nx, std::vector<double>(ny, 0.0));
    ScalarField kBz3(nx, std::vector<double>(ny, 0.0));
    ScalarField kBz4(nx, std::vector<double>(ny, 0.0));
    ScalarField bx_tmp(nx+1, std::vector<double>(ny,   0.0));
    ScalarField by_tmp(nx,   std::vector<double>(ny+1, 0.0));
    ScalarField Bz_tmp(nx,   std::vector<double>(ny,   0.0));

    constexpr double rho_floor = 0.1;

    // Stage 1: rates at (bx_0, by_0, Bz_0)
    hall_stage(face_.bx, face_.by, Bz0, w, nx, ny, dx, dy,
               hall_di_, bcx_, bcy_, rho_floor, emfz1, kBz1);

    // Stage 2: half-step advance from stage 1
    faraday_advance(face_.bx, face_.by, emfz1, nx, ny, dx, dy, dt*0.5, bx_tmp, by_tmp);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Bz_tmp[i][j] = Bz0[i][j] + (dt*0.5) * kBz1[i][j];
    hall_stage(bx_tmp, by_tmp, Bz_tmp, w, nx, ny, dx, dy,
               hall_di_, bcx_, bcy_, rho_floor, emfz2, kBz2);

    // Stage 3: half-step advance from stage 2
    faraday_advance(face_.bx, face_.by, emfz2, nx, ny, dx, dy, dt*0.5, bx_tmp, by_tmp);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Bz_tmp[i][j] = Bz0[i][j] + (dt*0.5) * kBz2[i][j];
    hall_stage(bx_tmp, by_tmp, Bz_tmp, w, nx, ny, dx, dy,
               hall_di_, bcx_, bcy_, rho_floor, emfz3, kBz3);

    // Stage 4: full-step advance from stage 3
    faraday_advance(face_.bx, face_.by, emfz3, nx, ny, dx, dy, dt, bx_tmp, by_tmp);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Bz_tmp[i][j] = Bz0[i][j] + dt * kBz3[i][j];
    hall_stage(bx_tmp, by_tmp, Bz_tmp, w, nx, ny, dx, dy,
               hall_di_, bcx_, bcy_, rho_floor, emfz4, kBz4);

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
// Hall-HLL 哨声波速 HLL 1 阶迎风扩散稳定化（Path B）
//
// 推导：哨声波以波速 c_w = π·di·|B|_角/(ρ_角·mincell) 沿磁场方向传播。
// 将该速度作为 HLL 扩散系数，对 By 和 Bx 各施加一阶迎风差分，
// 等价于向角点 EMF 添加：
//   δEz[I][J] = +(c_w/2)·(By_right − By_left)   ← By 在 x 方向正扩散
//              −(c_w/2)·(Bx_above − Bx_below)   ← Bx 在 y 方向正扩散
//
// 对应 ∂By/∂t 贡献：+(c_w·dx/2)·∂²By/∂x²  > 0（稳定）
// 对应 ∂Bx/∂t 贡献：+(c_w·dy/2)·∂²Bx/∂y²  > 0（稳定）
//
// 稳定性条件：c_w·dt/mincell ≤ 1，由 compute_dt 中的 smax_hll 项保证。
// 注意：该方案将哨声波近似为纯扩散（过阻尼，γ/ω ≈ π/2），
// 与 Iwasaki & Tomida (2025) Hall-HLL 描述一致。
// ---------------------------------------------------------------------------
void CTDivergenceControl::add_hall_hll_stabilization(Grid& w, int nx, int ny,
                                                     double dx, double dy) {
    if (hall_di_ <= 0.0) return;
    constexpr double pi   = 3.14159265358979323846;
    const double mincell  = std::min(dx, dy);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int J = 0; J <= ny; ++J) {
            // 角点 (I,J) 对应周围四个单元格：w[I+1][J+1]..w[I+2][J+2]（含 2 层 ghost）
            double rho_c = 0.25 * (w[I+1][J+1][0] + w[I+2][J+1][0]
                                 + w[I+1][J+2][0] + w[I+2][J+2][0]);
            rho_c = std::max(rho_c, 0.1);  // 与 Hall CFL floor 一致

            double Bx_c = 0.25 * (w[I+1][J+1][5] + w[I+2][J+1][5]
                                 + w[I+1][J+2][5] + w[I+2][J+2][5]);
            double By_c = 0.25 * (w[I+1][J+1][6] + w[I+2][J+1][6]
                                 + w[I+1][J+2][6] + w[I+2][J+2][6]);
            double Bz_c = 0.25 * (w[I+1][J+1][7] + w[I+2][J+1][7]
                                 + w[I+1][J+2][7] + w[I+2][J+2][7]);
            double B2_c = Bx_c*Bx_c + By_c*By_c + Bz_c*Bz_c;

            // c_w = π·di·|B|_角/(ρ_角·mincell)
            const double c_w = pi * hall_di_ * std::sqrt(B2_c) / (rho_c * mincell);

            // face_.by[i][J]：y 向面，i = 0..nx-1，J = 0..ny
            // 角点 (I,J) 右侧 By face：i = I（若 I < nx），左侧：i = I-1
            int i_right = (bcx_ == BC::Periodic) ? (I < nx ? I : 0)
                                                  : (I < nx ? I : nx-1);
            int i_left  = (bcx_ == BC::Periodic) ? (I > 0 ? I-1 : nx-1)
                                                  : (I > 0 ? I-1 : 0);
            double jump_by = face_.by[i_right][J] - face_.by[i_left][J];

            // face_.bx[I][j]：x 向面，I = 0..nx，j = 0..ny-1
            // 角点 (I,J) 上方 Bx face：j = J（若 J < ny），下方：j = J-1
            int j_above = (bcy_ == BC::Periodic) ? (J < ny ? J : 0)
                                                  : (J < ny ? J : ny-1);
            int j_below = (bcy_ == BC::Periodic) ? (J > 0 ? J-1 : ny-1)
                                                  : (J > 0 ? J-1 : 0);
            double jump_bx = face_.bx[I][j_above] - face_.bx[I][j_below];

            face_.emf_z[I][J] += (c_w * 0.5) * jump_by - (c_w * 0.5) * jump_bx;
        }
    }
}

} // namespace my_project
