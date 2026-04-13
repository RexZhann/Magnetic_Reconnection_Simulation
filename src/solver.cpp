#include "my_project/solver.hpp"

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

void slic_step(ScratchBuf& sc, Row& wp, int n, double dt, double dx,
               const RunConfig& cfg, DivergenceController& divb) {
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
            delta[i][k] = 0.5 * (d0[i][k] + d1[i][k]);
        }
    }

    for (int i = 1; i < N - 1; ++i) {
        for (int k = 0; k < NVAR; ++k) {
            double den = d1[i][k];
            double r = (std::fabs(den) < 1e-30) ? 0.0 : d0[i][k] / den;
            double phi = minbee(r);
            xL[i][k] = uc[i][k] - 0.5 * phi * delta[i][k];
            xR[i][k] = uc[i][k] + 0.5 * phi * delta[i][k];
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

    for (int i = 1; i < N - 1; ++i) {
        Vec fL = phys_flux(xL[i], cfg.gamma, divb.glm_enabled(), divb.characteristic_speed());
        Vec fR = phys_flux(xR[i], cfg.gamma, divb.glm_enabled(), divb.characteristic_speed());
        for (int k = 0; k < NVAR; ++k) {
            hL[i][k] = xL[i][k] - 0.5 * (dt / dx) * (fR[k] - fL[k]);
            hR[i][k] = xR[i][k] - 0.5 * (dt / dx) * (fR[k] - fL[k]);
        }
        if (hL[i][0] < 0.0 || hR[i][0] < 0.0 || check_pressure(hL[i]) < 0.0 || check_pressure(hR[i]) < 0.0) {
            hL[i] = uc[i]; hR[i] = uc[i];
        }
    }

    for (int i = 1; i < n + 2; ++i) {
        if (cfg.solver == SolverKind::FORCE) {
            iflx[i] = force_flux(hR[i], hL[i + 1], dt, dx, cfg.gamma,
                                 divb.glm_enabled(), divb.characteristic_speed());
        } else {
            iflx[i] = hlld_flux(hR[i], hL[i + 1], cfg.gamma,
                                divb.glm_enabled(), divb.characteristic_speed());
        }
    }

    for (int i = 2; i < n + 2; ++i) {
        for (int k = 0; k < NVAR; ++k) {
            if (!divb.glm_enabled() && k == 5) continue;
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

void sweep_x(Grid& w, int nx, int ny, double dt, double dx, const RunConfig& cfg,
             DivergenceController& divb) {
    #pragma omp parallel
    {
        ScratchBuf sc;
        sc.ensure(nx + 4);
        #pragma omp for schedule(static)
        for (int j = 2; j < ny + 2; ++j) {
            Row& s = sc.s_row;
            for (int i = 0; i < nx + 4; ++i) s[i] = w[i][j];
            slic_step(sc, s, nx, dt, dx, cfg, divb);
            for (int i = 2; i < nx + 2; ++i) w[i][j] = s[i];
        }
    }
}

void sweep_y(Grid& w, int nx, int ny, double dt, double dy, const RunConfig& cfg,
             DivergenceController& divb) {
    #pragma omp parallel
    {
        ScratchBuf sc;
        sc.ensure(ny + 4);
        #pragma omp for schedule(static)
        for (int i = 2; i < nx + 2; ++i) {
            Row& s = sc.s_row;
            for (int j = 0; j < ny + 4; ++j) {
                s[j] = w[i][j];
                std::swap(s[j][1], s[j][2]);
                std::swap(s[j][5], s[j][6]);
            }
            slic_step(sc, s, ny, dt, dy, cfg, divb);
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
    double smax = 1e-10, ch_loc = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:smax,ch_loc) schedule(static)
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
        }
    }
    divb.update_characteristic_speed(ch_loc);
    return cfg.cfl / smax;
}

std::vector<std::vector<double>> compute_divB(const Grid& w, int nx, int ny,
                                              double dx, double dy) {
    std::vector<std::vector<double>> dB(nx, std::vector<double>(ny, 0.0));
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
                                double dx, double dy) {
    Diagnostics d;
    double min_rho = 1e30, min_p = 1e30, max_divB = 0.0, max_psi = 0.0, max_v = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) \
        reduction(min:min_rho,min_p) reduction(max:max_divB,max_psi,max_v)
    for (int i = 2; i < nx + 2; ++i) {
        for (int j = 2; j < ny + 2; ++j) {
            const Vec& p = w[i][j];
            min_rho = std::min(min_rho, p[0]);
            min_p = std::min(min_p, p[4]);
            max_psi = std::max(max_psi, std::fabs(p[8]));
            max_v = std::max(max_v, std::sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
            double dBx = (w[i + 1][j][5] - w[i - 1][j][5]) / (2 * dx);
            double dBy = (w[i][j + 1][6] - w[i][j - 1][6]) / (2 * dy);
            max_divB = std::max(max_divB, std::fabs(dBx + dBy));
        }
    }
    d.min_rho = min_rho; d.min_p = min_p; d.max_divB = max_divB; d.max_psi = max_psi; d.max_v = max_v;
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
    auto divb = make_divergence_controller(cfg.divb);
    divb->initialize(w, cfg.nx, cfg.ny, dx, dy);

    double t = 0.0;
    int step = 0;
    double t_sweepx = 0.0, t_sweepy = 0.0, t_other = 0.0;
    auto T0 = Clock::now();

    while (t < cfg.t_end) {
        auto ta = Clock::now();
        apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
        double dt = compute_dt(w, cfg.nx, cfg.ny, dx, dy, cfg, *divb);
        if (t + dt > cfg.t_end) dt = cfg.t_end - t;
        if (dt <= 0) break;

        apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
        divb->pre_step(w, cfg.nx, cfg.ny, dt, dx, dy);
        auto tb = Clock::now();
        sweep_x(w, cfg.nx, cfg.ny, 0.5 * dt, dx, cfg, *divb);
        auto tc = Clock::now();
        apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
        sweep_y(w, cfg.nx, cfg.ny, dt, dy, cfg, *divb);
        auto td = Clock::now();
        apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
        sweep_x(w, cfg.nx, cfg.ny, 0.5 * dt, dx, cfg, *divb);
        auto te = Clock::now();
        divb->post_step(w, cfg.nx, cfg.ny, dt, Lx, Ly, dx, dy);
        auto tf = Clock::now();

        t_sweepx += elapsed(tb, tc) + elapsed(td, te);
        t_sweepy += elapsed(tc, td);
        t_other += elapsed(ta, tb) + elapsed(te, tf);
        t += dt;
        ++step;

        if (step % 200 == 0) {
            apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy);
            Diagnostics diag = compute_diagnostics(w, cfg.nx, cfg.ny, dx, dy);
            std::cout << "Step " << step
                      << "  t=" << t
                      << "  dt=" << dt
                      << "  min_rho=" << diag.min_rho
                      << "  min_p=" << diag.min_p
                      << "  max|divB|=" << diag.max_divB
                      << "  max|psi|=" << diag.max_psi
                      << "  max|v|=" << diag.max_v
                      << '\n';
            if (diag.min_rho < 0 || diag.min_p < 0 || !std::isfinite(diag.max_divB)) {
                std::cerr << "*** FATAL: unphysical state at step " << step << '\n';
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
    out.divB = compute_divB(out.primitive, cfg.nx, cfg.ny, dx, dy);
    out.timing = timing;
    if (const FaceField2D* ff = divb->face_field()) {
        out.has_face_field = true;
        out.face_field = *ff;
    }
    return out;
}

void write_output_file(const OutputData& out, const RunConfig& cfg) {
    const double dx = (cfg.x1 - cfg.x0) / cfg.nx;
    const double dy = (cfg.y1 - cfg.y0) / cfg.ny;
    const std::string filename = "test" + std::to_string(cfg.test) + "_"
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

} // namespace my_project
