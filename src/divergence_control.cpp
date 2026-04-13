#include "my_project/divergence_control.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace my_project {

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

void CTDivergenceControl::initialize(Grid& w, int nx, int ny, double /*dx*/, double /*dy*/) {
    face_.resize(nx, ny);
    fill_faces_from_cell_centered(w, nx, ny);
    sync_cell_centered_from_faces(w, nx, ny);
}

void CTDivergenceControl::pre_step(Grid& w, int nx, int ny, double /*dt*/, double /*dx*/, double /*dy*/) {
    // Rebuild face-centered magnetic field from the current cell-centered state
    // before the split sweeps. This keeps the interface between today's cell-
    // centered solver and tomorrow's true CT update explicit.
    fill_faces_from_cell_centered(w, nx, ny);
}

void CTDivergenceControl::post_step(Grid& w, int nx, int ny, double dt, double /*Lx*/, double /*Ly*/, double dx, double dy) {
    // Structural CT scaffold:
    // 1) build corner EMFs from cell-centered velocities and face-centered B,
    // 2) update face-centered B using Faraday's law,
    // 3) sync cell-centered B from faces.
    //
    // This is intentionally a lightweight, buildable bridge from the current
    // GLM codebase toward a full CT implementation. It is not yet a production-
    // grade unsplit CT integrator.
    compute_corner_emf(w, nx, ny);
    update_faces_from_emf(nx, ny, dt, dx, dy);
    enforce_boundary_faces(w, nx, ny);
    sync_cell_centered_from_faces(w, nx, ny);

    // psi is not used by CT.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 2; i < nx + 2; ++i) {
        for (int j = 2; j < ny + 2; ++j) {
            w[i][j][8] = 0.0;
        }
    }
}

void CTDivergenceControl::fill_faces_from_cell_centered(const Grid& w, int nx, int ny) {
    face_.resize(nx, ny);

    // Bx on vertical faces.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx + 1; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (i == 0) {
                face_.bx[i][j] = w[2][j + 2][5];
            } else if (i == nx) {
                face_.bx[i][j] = w[nx + 1][j + 2][5];
            } else {
                face_.bx[i][j] = 0.5 * (w[i + 1][j + 2][5] + w[i + 2][j + 2][5]);
            }
        }
    }

    // By on horizontal faces.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny + 1; ++j) {
            if (j == 0) {
                face_.by[i][j] = w[i + 2][2][6];
            } else if (j == ny) {
                face_.by[i][j] = w[i + 2][ny + 1][6];
            } else {
                face_.by[i][j] = 0.5 * (w[i + 2][j + 1][6] + w[i + 2][j + 2][6]);
            }
        }
    }
}

void CTDivergenceControl::compute_corner_emf(const Grid& w, int nx, int ny) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx + 1; ++i) {
        for (int j = 0; j < ny + 1; ++j) {
            // Corner-centered velocity from surrounding cells.
            double vx = 0.0, vy = 0.0, count = 0.0;
            for (int di = -1; di <= 0; ++di) {
                for (int dj = -1; dj <= 0; ++dj) {
                    int ci = i + di;
                    int cj = j + dj;
                    if (ci >= 0 && ci < nx && cj >= 0 && cj < ny) {
                        const Vec& p = w[ci + 2][cj + 2];
                        vx += p[1];
                        vy += p[2];
                        count += 1.0;
                    }
                }
            }
            if (count > 0.0) {
                vx /= count;
                vy /= count;
            }

            // Corner-centered B from adjacent faces.
            double bx = 0.0, by = 0.0;
            if (j == 0) {
                bx = face_.bx[i][0];
            } else if (j == ny) {
                bx = face_.bx[i][ny - 1];
            } else {
                bx = 0.5 * (face_.bx[i][j - 1] + face_.bx[i][j]);
            }
            if (i == 0) {
                by = face_.by[0][j];
            } else if (i == nx) {
                by = face_.by[nx - 1][j];
            } else {
                by = 0.5 * (face_.by[i - 1][j] + face_.by[i][j]);
            }

            // 2D ideal MHD: Ez = -(v x B)_z = -(vx * By - vy * Bx).
            face_.emf_z[i][j] = -(vx * by - vy * bx);
        }
    }
}

void CTDivergenceControl::update_faces_from_emf(int nx, int ny, double dt, double dx, double dy) {
    ScalarField new_bx = face_.bx;
    ScalarField new_by = face_.by;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx + 1; ++i) {
        for (int j = 0; j < ny; ++j) {
            new_bx[i][j] = face_.bx[i][j] - (dt / dy) * (face_.emf_z[i][j + 1] - face_.emf_z[i][j]);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny + 1; ++j) {
            new_by[i][j] = face_.by[i][j] + (dt / dx) * (face_.emf_z[i + 1][j] - face_.emf_z[i][j]);
        }
    }

    face_.bx.swap(new_bx);
    face_.by.swap(new_by);
}

void CTDivergenceControl::sync_cell_centered_from_faces(Grid& w, int nx, int ny) const {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            w[i + 2][j + 2][5] = 0.5 * (face_.bx[i][j] + face_.bx[i + 1][j]);
            w[i + 2][j + 2][6] = 0.5 * (face_.by[i][j] + face_.by[i][j + 1]);
        }
    }
}

void CTDivergenceControl::enforce_boundary_faces(Grid& w, int nx, int ny) {
    // Keep boundary faces consistent with ghost-cell-updated cell-centered field.
    // This is the narrowest possible bridge to the current split solver.
    for (int j = 0; j < ny; ++j) {
        face_.bx[0][j] = w[2][j + 2][5];
        face_.bx[nx][j] = w[nx + 1][j + 2][5];
    }
    for (int i = 0; i < nx; ++i) {
        face_.by[i][0] = w[i + 2][2][6];
        face_.by[i][ny] = w[i + 2][ny + 1][6];
    }
}

} // namespace my_project
