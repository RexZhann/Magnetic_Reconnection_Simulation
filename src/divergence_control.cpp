#include "my_project/divergence_control.hpp"
#include "my_project/harris_sheet.hpp"

#include <algorithm>
#include <cmath>
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
    emf_y_.assign(cfg.nx, std::vector<double>(cfg.ny + 1, 0.0));
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
    compute_corner_emf_from_interface_emfs(nx, ny);
    add_resistive_correction(w, nx, ny, dt, dx, dy);  // no-op when eta_ == 0
    add_hall_correction(w, nx, ny, dt, dx, dy);        // no-op when hall_di_ == 0
    update_faces_from_emf(nx, ny, dt, dx, dy);
    sync_cell_centered_from_faces(w, nx, ny);

    // ψ is not used by CT.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 2; i < nx + 2; ++i) {
        for (int j = 2; j < ny + 2; ++j) {
            w[i][j][8] = 0.0;
        }
    }
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
void CTDivergenceControl::store_emf_x(int j, int n, const double* emf) {
    for (int i = 1; i <= n + 1; ++i) emf_x_[i - 1][j] = emf[i];
}

void CTDivergenceControl::store_emf_y(int i, int n, const double* emf) {
    for (int j = 1; j <= n + 1; ++j) emf_y_[i][j - 1] = emf[j];
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
            case 11: case 12: {
                // Harris sheet: exact face Bx via vector-potential line integral
                // bx = [Az(x, y+Δy/2) − Az(x, y−Δy/2)] / Δy
                // Guarantees ∇·B = 0 to machine precision (Tóth 2000, §4.1).
                return harris_bx_face(x, y, dy, HarrisSheetParams{});
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
            case 11: case 12: {
                // Harris sheet: exact face By via vector-potential line integral
                // by = −[Az(x+Δx/2, y) − Az(x−Δx/2, y)] / Δx
                // Guarantees ∇·B = 0 to machine precision (Tóth 2000, §4.1).
                return harris_by_face(x, y, dx, HarrisSheetParams{});
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

// Corner EMF by arithmetic average of the four surrounding interface EMFs.
// For periodic BC, boundary interface EMFs are averaged first to keep
// emf_z[0][J] == emf_z[nx][J], which ensures identical Faraday increments
// on the two copies of the periodic boundary face.
void CTDivergenceControl::compute_corner_emf_from_interface_emfs(int nx, int ny) {
    if (bcx_ == BC::Periodic) {
        for (int j = 0; j < ny; ++j) {
            double avg = 0.5*(emf_x_[0][j] + emf_x_[nx][j]);
            emf_x_[0][j] = avg;  emf_x_[nx][j] = avg;
        }
    }
    if (bcy_ == BC::Periodic) {
        for (int i = 0; i < nx; ++i) {
            double avg = 0.5*(emf_y_[i][0] + emf_y_[i][ny]);
            emf_y_[i][0] = avg;  emf_y_[i][ny] = avg;
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
            face_.emf_z[I][J] = 0.25*(emf_x_[I][Jm] + emf_x_[I][Jp]
                                     + emf_y_[Im][J] + emf_y_[Ip][J]);
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
// The Hall EMF is E^Hall = (d_i/ρ) J × B.  In 2.5-D (∂/∂z = 0):
//   J_x = ∂Bz/∂y,   J_y = -∂Bz/∂x,   J_z = ∂By/∂x - ∂Bx/∂y
//
// Step 1: Corner E_z^Hall = (d_i/ρ)(J_x B_y - J_y B_x) → added to face_.emf_z
//         so the subsequent CT Faraday step advances Bx and By correctly.
//
// Step 2: Update cell-centred Bz using the finite-volume discretisation
//   ΔBz[i][j] = dt × [(Ex^H[i][j+1] - Ex^H[i][j]) / dy
//                    - (Ey^H[i+1][j] - Ey^H[i][j]) / dx]
//   with
//   Ex^H at y-face (i,J) = (d_i/ρ)(J_y Bz - J_z By)
//   Ey^H at x-face (I,j) = (d_i/ρ)(J_z Bx - J_x Bz)
//
// Stability: Hall whistler CFL dt ≤ min(dx,dy)² / (d_i·v_A) is enforced in
// compute_dt, so no sub-cycling is needed here.
//
// Reference: Tóth et al. (2008) J. Comput. Phys. 227, 6967-6984, §3.
// ---------------------------------------------------------------------------
void CTDivergenceControl::add_hall_correction(Grid& w, int nx, int ny,
                                              double dt, double dx, double dy) {
    if (hall_di_ <= 0.0) return;

    // -----------------------------------------------------------------------
    // Precompute corner current components  (I=0..nx, J=0..ny)
    //
    // Interior cell (i,j) [0-indexed] sits at padded position w[i+2][j+2].
    // Corner (I,J) at x=I·dx, y=J·dy is surrounded by four cells
    //   (I-1,J-1),(I,J-1),(I-1,J),(I,J) → padded w[I+1][J+1], w[I+2][J+1],
    //                                              w[I+1][J+2], w[I+2][J+2]
    // Ghost cells (index 0, 1, nx+2, nx+3 etc.) are already filled by
    // apply_bc before post_step is called, so boundary corners are correct.
    // -----------------------------------------------------------------------
    ScalarField Jx_c(nx + 1, std::vector<double>(ny + 1, 0.0));
    ScalarField Jy_c(nx + 1, std::vector<double>(ny + 1, 0.0));
    ScalarField Jz_c(nx + 1, std::vector<double>(ny + 1, 0.0));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int J = 0; J <= ny; ++J) {
            // Jz from face B (BC-aware, matches add_resistive_correction)
            double byR, byL, bxU, bxD;
            if (bcx_ == BC::Periodic) {
                byR = face_.by[(I < nx) ? I      : 0     ][J];
                byL = face_.by[(I > 0)  ? I - 1  : nx - 1][J];
            } else {
                byR = face_.by[(I < nx) ? I     : nx - 1][J];
                byL = face_.by[(I > 0)  ? I - 1 : 0     ][J];
            }
            if (bcy_ == BC::Periodic) {
                bxU = face_.bx[I][(J < ny) ? J      : 0     ];
                bxD = face_.bx[I][(J > 0)  ? J - 1  : ny - 1];
            } else {
                bxU = face_.bx[I][(J < ny) ? J     : ny - 1];
                bxD = face_.bx[I][(J > 0)  ? J - 1 : 0     ];
            }
            Jz_c[I][J] = (byR - byL) / dx - (bxU - bxD) / dy;

            // Jx = ∂Bz/∂y at corner: average Bz of cells above minus below, / dy
            double Bz_up   = 0.5 * (w[I+1][J+2][7] + w[I+2][J+2][7]);
            double Bz_down = 0.5 * (w[I+1][J+1][7] + w[I+2][J+1][7]);
            Jx_c[I][J] = (Bz_up - Bz_down) / dy;

            // Jy = -∂Bz/∂x at corner: -(Bz_right - Bz_left) / dx
            double Bz_right = 0.5 * (w[I+2][J+1][7] + w[I+2][J+2][7]);
            double Bz_left  = 0.5 * (w[I+1][J+1][7] + w[I+1][J+2][7]);
            Jy_c[I][J] = -(Bz_right - Bz_left) / dx;
        }
    }

    // -----------------------------------------------------------------------
    // Step 1: Add E_z^Hall to corner EMF → drives Bx, By via Faraday
    // -----------------------------------------------------------------------
    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int J = 0; J <= ny; ++J) {
            double rho_c = 0.25 * (w[I+1][J+1][0] + w[I+2][J+1][0] +
                                   w[I+1][J+2][0] + w[I+2][J+2][0]);
            rho_c = std::max(rho_c, 1e-12);

            // Bx at corner: average adjacent x-faces
            double bxD2, bxU2;
            if (bcy_ == BC::Periodic) {
                bxU2 = face_.bx[I][(J < ny) ? J      : 0     ];
                bxD2 = face_.bx[I][(J > 0)  ? J - 1  : ny - 1];
            } else {
                bxU2 = face_.bx[I][(J < ny) ? J     : ny - 1];
                bxD2 = face_.bx[I][(J > 0)  ? J - 1 : 0     ];
            }
            double Bx_c = 0.5 * (bxD2 + bxU2);

            // By at corner: average adjacent y-faces
            double byL2, byR2;
            if (bcx_ == BC::Periodic) {
                byR2 = face_.by[(I < nx) ? I      : 0     ][J];
                byL2 = face_.by[(I > 0)  ? I - 1  : nx - 1][J];
            } else {
                byR2 = face_.by[(I < nx) ? I     : nx - 1][J];
                byL2 = face_.by[(I > 0)  ? I - 1 : 0     ][J];
            }
            double By_c = 0.5 * (byL2 + byR2);

            face_.emf_z[I][J] += (hall_di_ / rho_c)
                                  * (Jx_c[I][J] * By_c - Jy_c[I][J] * Bx_c);
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: Update cell-centred Bz via Hall Ex and Ey
    //
    // Ex^Hall at y-face (i, J): x=(i+0.5)dx, y=J·dy
    //   = (d_i/ρ)(J_y Bz - J_z By)
    //   By = face_.by[i][J]  (exact, stored at this face)
    //   ρ, Bz averaged from cells below (i,J-1) and above (i,J)
    //   J_y, J_z averaged from corners (i,J) and (i+1,J)
    //
    // Ey^Hall at x-face (I, j): x=I·dx, y=(j+0.5)dy
    //   = (d_i/ρ)(J_z Bx - J_x Bz)
    //   Bx = face_.bx[I][j]  (exact, stored at this face)
    //   ρ, Bz averaged from cells left (I-1,j) and right (I,j)
    //   J_x, J_z averaged from corners (I,j) and (I,j+1)
    // -----------------------------------------------------------------------
    ScalarField ExH(nx,     std::vector<double>(ny + 1, 0.0));  // [i][J]  i∈[0,nx-1], J∈[0,ny]
    ScalarField EyH(nx + 1, std::vector<double>(ny,     0.0));  // [I][j]  I∈[0,nx],   j∈[0,ny-1]

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int J = 0; J <= ny; ++J) {
            // cells below (i,J-1) → padded w[i+2][J+1], above (i,J) → w[i+2][J+2]
            double rho_f = 0.5 * (w[i+2][J+1][0] + w[i+2][J+2][0]);
            rho_f = std::max(rho_f, 1e-12);
            double Bz_f  = 0.5 * (w[i+2][J+1][7] + w[i+2][J+2][7]);
            double By_f  = face_.by[i][J];
            double Jy_f  = 0.5 * (Jy_c[i][J] + Jy_c[i+1][J]);
            double Jz_f  = 0.5 * (Jz_c[i][J] + Jz_c[i+1][J]);
            ExH[i][J] = (hall_di_ / rho_f) * (Jy_f * Bz_f - Jz_f * By_f);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I <= nx; ++I) {
        for (int j = 0; j < ny; ++j) {
            // cells left (I-1,j) → padded w[I+1][j+2], right (I,j) → w[I+2][j+2]
            double rho_f = 0.5 * (w[I+1][j+2][0] + w[I+2][j+2][0]);
            rho_f = std::max(rho_f, 1e-12);
            double Bz_f  = 0.5 * (w[I+1][j+2][7] + w[I+2][j+2][7]);
            double Bx_f  = face_.bx[I][j];
            double Jx_f  = 0.5 * (Jx_c[I][j] + Jx_c[I][j+1]);
            double Jz_f  = 0.5 * (Jz_c[I][j] + Jz_c[I][j+1]);
            EyH[I][j] = (hall_di_ / rho_f) * (Jz_f * Bx_f - Jx_f * Bz_f);
        }
    }

    // ΔBz[i][j] = dt·[(ExH[i][j+1] - ExH[i][j])/dy - (EyH[i+1][j] - EyH[i][j])/dx]
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            w[i+2][j+2][7] += dt * ((ExH[i][j+1] - ExH[i][j]) / dy
                                   - (EyH[i+1][j] - EyH[i][j]) / dx);
        }
    }
}

} // namespace my_project
