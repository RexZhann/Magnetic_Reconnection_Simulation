#include "my_project/divergence_control.hpp"

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

} // namespace my_project
