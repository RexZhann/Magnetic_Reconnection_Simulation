#include "my_project/divergence_control.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace my_project {

namespace {

int clamp_or_wrap(int idx, int n, BC bc) {
    if (bc == BC::Periodic) {
        if (n <= 0) return 0;
        int wrapped = idx % n;
        return (wrapped < 0) ? wrapped + n : wrapped;
    }
    return std::clamp(idx, 0, n - 1);
}

double ideal_ez_from_cell(const Vec& p) {
    return -(p[1] * p[6] - p[2] * p[5]);
}

double total_energy_from_primitive(const Vec& p, double gamma) {
    const double rho = std::max(p[0], 1e-10);
    return p[4] / (gamma - 1.0)
         + 0.5 * rho * (p[1] * p[1] + p[2] * p[2] + p[3] * p[3])
         + 0.5 * (p[5] * p[5] + p[6] * p[6] + p[7] * p[7]);
}

} // namespace

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
void CTDivergenceControl::initialize(Grid& w, int nx, int ny, double /*dx*/, double /*dy*/) {
    nx_ = nx;
    ny_ = ny;
    face_.resize(nx, ny);
    // Interface EMF arrays (same shape as face B arrays).
    emf_x_.assign(nx + 1, std::vector<double>(ny, 0.0));
    emf_y_.assign(nx, std::vector<double>(ny + 1, 0.0));
    // Bootstrap face B from cell-centered initial conditions.
    // apply_bc must have been called before this so ghost cells are valid.
    fill_faces_from_cell_centered(w, nx, ny);
    // Make cell-centered B self-consistent with face B.
    sync_cell_centered_from_faces(w, nx, ny);
}

// ---------------------------------------------------------------------------
// CT: pre_step — no-op for CT.
//
// Face B is maintained solely by the Faraday update in post_step.  Overwriting
// boundary face values here (after apply_bc refreshes ghost cells) would
// corrupt the face-difference ∇·B invariant for boundary cells, because the
// Faraday stencil PRESERVES ∇·B algebraically — it cannot repair errors that
// were introduced before the sweep.
// ---------------------------------------------------------------------------
void CTDivergenceControl::pre_step(Grid& /*w*/, int /*nx*/, int /*ny*/,
                                   double /*dt*/, double /*dx*/, double /*dy*/) {
    // Intentionally empty.
}

// ---------------------------------------------------------------------------
// CT: post_step — compute corner EMFs, advance faces via Faraday, sync cells
// ---------------------------------------------------------------------------
void CTDivergenceControl::post_step(Grid& w, int nx, int ny,
                                    double dt, double /*Lx*/, double /*Ly*/,
                                    double dx, double dy) {
    compute_corner_emf_from_interface_emfs(w, nx, ny, dx, dy);
    update_faces_from_emf(nx, ny, dt, dx, dy);
    // Do NOT call apply_face_bc here: overwriting face B after the Faraday
    // update would corrupt the ∇·B=0 invariant for boundary cells.
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
// CT face-B access: fill contiguous buffer for slic_step
//
// slic_step interface index mapping (n interior cells, 2 ghost layers):
//   Interface i in slice (i=1..n+1) lies between ghost/interior cells i & i+1.
//   For x-sweep of interior row j: interface i ↔ face bx[i-1][j].
//   For y-sweep of interior col i: interface j ↔ face by[i][j-1].
// ---------------------------------------------------------------------------
void CTDivergenceControl::fill_face_bn_x(int j, int n, double* buf) const {
    // buf[i] for i=1..n+1;  buf[0] is unused but zeroed for safety.
    buf[0] = 0.0;
    for (int i = 1; i <= n + 1; ++i) {
        buf[i] = face_.bx[i - 1][j];
    }
}

void CTDivergenceControl::fill_face_bn_y(int i, int n, double* buf) const {
    buf[0] = 0.0;
    for (int j = 1; j <= n + 1; ++j) {
        buf[j] = face_.by[i][j - 1];
    }
}

// ---------------------------------------------------------------------------
// CT EMF storage: called by sweep_x/sweep_y after each slic_step row/col
//
// emf[i] holds Ez at slic_step interface i, for i=1..n+1.
//   x-sweep: Ez = −F[6] (where F[6] = vx·By − vy·Bx_face = −Ez).
//   y-sweep (rotated frame): Ez = +F[6] (By_rot·vx_rot − vy_rot·Bx_rot = Ez).
//
// Strang split has two x half-sweeps.  The second overwriting the first is
// intentional: the final x-sweep is evaluated at the most advanced state and
// gives the best estimate of Ez for the subsequent Faraday update.
// ---------------------------------------------------------------------------
void CTDivergenceControl::store_emf_x(int j, int n, const double* emf) {
    for (int i = 1; i <= n + 1; ++i) {
        emf_x_[i - 1][j] = emf[i];
    }
}

void CTDivergenceControl::store_emf_y(int i, int n, const double* emf) {
    for (int j = 1; j <= n + 1; ++j) {
        emf_y_[i][j - 1] = emf[j];
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

// Average cell-centered B to faces using all available cells.
// apply_bc must be called before this so ghost cells carry correct BC values:
//   - Periodic: ghosts wrap, giving bx[0][j] == bx[nx][j] automatically.
//   - Transmissive: ghosts copy nearest interior, giving zero-gradient boundary faces.
void CTDivergenceControl::fill_faces_from_cell_centered(const Grid& w, int nx, int ny) {
    face_.resize(nx, ny);

    // bx[i][j] lies between w[i+1][j+2] and w[i+2][j+2]  (i = 0..nx).
    // For i=0 the left neighbour is a ghost cell; for i=nx the right neighbour
    // is a ghost cell.  Both are valid because apply_bc was called first.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx + 1; ++i) {
        for (int j = 0; j < ny; ++j) {
            face_.bx[i][j] = 0.5 * (w[i + 1][j + 2][5] + w[i + 2][j + 2][5]);
        }
    }

    // by[i][j] lies between w[i+2][j+1] and w[i+2][j+2]  (j = 0..ny).
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny + 1; ++j) {
            face_.by[i][j] = 0.5 * (w[i + 2][j + 1][6] + w[i + 2][j + 2][6]);
        }
    }
}

// Corner EMF from the four surrounding interface EMFs, with BC-aware index
// wrapping at domain boundaries.
//
// Corner (I, J) is surrounded by:
//   emf_x_[I][J-1]  — x-face bx[I][J-1], centred at (x_I, y_{J-0.5})
//   emf_x_[I][J]    — x-face bx[I][J],   centred at (x_I, y_{J+0.5})
//   emf_y_[I-1][J]  — y-face by[I-1][J], centred at (x_{I-0.5}, y_J)
//   emf_y_[I][J]    — y-face by[I][J],   centred at (x_{I+0.5}, y_J)
//
// Periodic BC: boundary interface EMFs are averaged to enforce the periodicity
// invariant emf_x_[0][j] == emf_x_[nx][j] (same physical face) before the
// corner average.  This guarantees emf_z[0][J] == emf_z[nx][J], so the Faraday
// update gives identical increments to bx[0][j] and bx[nx][j] and preserves
// the bx[0] == bx[nx] invariant established by fill_faces_from_cell_centered.
//
// Transmissive BC: boundary corners use the nearest available interface EMF
// (zero-gradient / one-sided stencil).
void CTDivergenceControl::compute_corner_emf_from_interface_emfs(const Grid& w, int nx, int ny,
                                                                 double dx, double dy) {
    ScalarField ez_cell(nx, std::vector<double>(ny, 0.0));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            ez_cell[i][j] = ideal_ez_from_cell(w[i + 2][j + 2]);
        }
    }

    // Enforce periodic EMF symmetry at domain boundaries.
    if (bcx_ == BC::Periodic) {
        for (int j = 0; j < ny; ++j) {
            double avg = 0.5 * (emf_x_[0][j] + emf_x_[nx][j]);
            emf_x_[0][j]  = avg;
            emf_x_[nx][j] = avg;
        }
    }
    if (bcy_ == BC::Periodic) {
        for (int i = 0; i < nx; ++i) {
            double avg = 0.5 * (emf_y_[i][0] + emf_y_[i][ny]);
            emf_y_[i][0]  = avg;
            emf_y_[i][ny] = avg;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I < nx + 1; ++I) {
        for (int J = 0; J < ny + 1; ++J) {
            int Jm, Jp, Im, Ip;
            if (bcy_ == BC::Periodic) {
                Jm = (J > 0)  ? J - 1 : ny - 1;
                Jp = (J < ny) ? J     : 0;
            } else {
                Jm = (J > 0)  ? J - 1 : 0;
                Jp = (J < ny) ? J     : ny - 1;
            }
            if (bcx_ == BC::Periodic) {
                Im = (I > 0)  ? I - 1 : nx - 1;
                Ip = (I < nx) ? I     : 0;
            } else {
                Im = (I > 0)  ? I - 1 : 0;
                Ip = (I < nx) ? I     : nx - 1;
            }

            auto dedy_on_x_face = [&](int face_i, int face_j) -> double {
                const double vx_face = 0.5 * (w[face_i + 1][face_j + 2][1]
                                            + w[face_i + 2][face_j + 2][1]);
                const int left_col = clamp_or_wrap(face_i - 1, nx, bcx_);
                const int right_col = clamp_or_wrap(face_i, nx, bcx_);
                const int row_lo = clamp_or_wrap(face_j, ny, bcy_);
                const int row_hi = clamp_or_wrap(face_j + 1, ny, bcy_);
                const double left_grad = (ez_cell[left_col][row_hi] - ez_cell[left_col][row_lo]) / dy;
                const double right_grad = (ez_cell[right_col][row_hi] - ez_cell[right_col][row_lo]) / dy;
                if (vx_face > 1e-12) return left_grad;
                if (vx_face < -1e-12) return right_grad;
                return 0.5 * (left_grad + right_grad);
            };

            auto dedx_on_y_face = [&](int face_i, int face_j) -> double {
                const double vy_face = 0.5 * (w[face_i + 2][face_j + 1][2]
                                            + w[face_i + 2][face_j + 2][2]);
                const int lower_row = clamp_or_wrap(face_j - 1, ny, bcy_);
                const int upper_row = clamp_or_wrap(face_j, ny, bcy_);
                const int col_lo = clamp_or_wrap(face_i, nx, bcx_);
                const int col_hi = clamp_or_wrap(face_i + 1, nx, bcx_);
                const double lower_grad = (ez_cell[col_hi][lower_row] - ez_cell[col_lo][lower_row]) / dx;
                const double upper_grad = (ez_cell[col_hi][upper_row] - ez_cell[col_lo][upper_row]) / dx;
                if (vy_face > 1e-12) return lower_grad;
                if (vy_face < -1e-12) return upper_grad;
                return 0.5 * (lower_grad + upper_grad);
            };

            const double edge_avg = 0.25 * (emf_x_[I][Jm] + emf_x_[I][Jp]
                                          + emf_y_[Im][J] + emf_y_[Ip][J]);
            const double dedy_lo = dedy_on_x_face(I, Jm);
            const double dedy_hi = dedy_on_x_face(I, Jp);
            const double dedx_lo = dedx_on_y_face(Im, J);
            const double dedx_hi = dedx_on_y_face(Ip, J);

            face_.emf_z[I][J] = edge_avg
                              + 0.125 * dy * (dedy_lo - dedy_hi)
                              + 0.125 * dx * (dedx_lo - dedx_hi);
        }
    }
}

// Advance face B by one full time step using the discrete Faraday law:
//   ΔBx_{I,J} = −(dt/dy) · (Ez_{I,J+1} − Ez_{I,J})
//   ΔBy_{I,J} =  (dt/dx) · (Ez_{I+1,J} − Ez_{I,J})
void CTDivergenceControl::update_faces_from_emf(int nx, int ny,
                                                double dt, double dx, double dy) {
    ScalarField new_bx = face_.bx;
    ScalarField new_by = face_.by;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx + 1; ++i) {
        for (int j = 0; j < ny; ++j) {
            new_bx[i][j] -= (dt / dy) * (face_.emf_z[i][j + 1] - face_.emf_z[i][j]);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny + 1; ++j) {
            new_by[i][j] += (dt / dx) * (face_.emf_z[i + 1][j] - face_.emf_z[i][j]);
        }
    }

    face_.bx.swap(new_bx);
    face_.by.swap(new_by);
}

// Derive cell-centered B by averaging adjacent face values while preserving
// each cell's total energy. Primitive pressure is adjusted so the synchronized
// magnetic field does not inject or remove conserved energy.
void CTDivergenceControl::sync_cell_centered_from_faces(Grid& w, int nx, int ny) const {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            Vec& p = w[i + 2][j + 2];
            const double total_energy = total_energy_from_primitive(p, gamma_);
            p[5] = 0.5 * (face_.bx[i][j] + face_.bx[i + 1][j]);
            p[6] = 0.5 * (face_.by[i][j] + face_.by[i][j + 1]);
            const double rho = std::max(p[0], 1e-10);
            const double kinetic = 0.5 * rho * (p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
            const double magnetic = 0.5 * (p[5] * p[5] + p[6] * p[6] + p[7] * p[7]);
            p[4] = std::max((gamma_ - 1.0) * (total_energy - kinetic - magnetic), 1e-10);
        }
    }
}

// Enforce face BC at domain boundaries.
// Periodic: average the two boundary faces that represent the same physical
//           face so that bx[0][j] == bx[nx][j] and by[i][0] == by[i][ny].
// Transmissive: zero-gradient extrapolation (boundary face = nearest interior face).
//
// Called only from apply_face_bc (which may be invoked by external code that
// needs to re-establish BC after modifying face fields directly).  It is NOT
// called from post_step, which would corrupt the ∇·B=0 invariant.
void CTDivergenceControl::apply_face_bc(int nx, int ny) {
    if (bcx_ == BC::Periodic) {
        for (int j = 0; j < ny; ++j) {
            double avg = 0.5 * (face_.bx[0][j] + face_.bx[nx][j]);
            face_.bx[0][j]  = avg;
            face_.bx[nx][j] = avg;
        }
    } else {
        for (int j = 0; j < ny; ++j) {
            face_.bx[0][j]  = face_.bx[1][j];
            face_.bx[nx][j] = face_.bx[nx - 1][j];
        }
    }
    if (bcy_ == BC::Periodic) {
        for (int i = 0; i < nx; ++i) {
            double avg = 0.5 * (face_.by[i][0] + face_.by[i][ny]);
            face_.by[i][0]  = avg;
            face_.by[i][ny] = avg;
        }
    } else {
        for (int i = 0; i < nx; ++i) {
            face_.by[i][0]  = face_.by[i][1];
            face_.by[i][ny] = face_.by[i][ny - 1];
        }
    }
}

} // namespace my_project
