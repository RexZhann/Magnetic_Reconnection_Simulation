#include "my_project/solver.hpp"

#include "my_project/perf.hpp"
#include "my_project/harris_sheet.hpp"
#include "my_project/asym_harris_sheet.hpp"
#include "my_project/double_harris.hpp"
#include "my_project/riemann.hpp"
#include "my_project/state.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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

// 返回该配置的输出目录（末尾无斜杠）。
// label 非空时为 output/{label}，否则为 output。
std::string cfg_outdir(const RunConfig& cfg) {
    return cfg.label.empty() ? "output" : "output/" + cfg.label;
}

using Sec = std::chrono::duration<double>;

double elapsed(Clock::time_point a, Clock::time_point b) {
    return Sec(b - a).count();
}

// campaign 质量计数器（test 31 L1 列；production run 期望全零）：
// floor 触发次数（apply_floor，串行）与 MUSCL/半步正性回退格数
// （slic_step，OMP 并行 → atomic）。累计值随 L1 行输出并存入 checkpoint。
long long g_floor_rho_count = 0;
long long g_floor_p_count   = 0;
std::atomic<long long> g_fallback_count{0};

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

    double tp0 = omp_get_wtime();
    for (int i = 0; i < N; ++i) uc[i] = pri2con(wp[i], cfg.gamma);
    double tp_conupd = omp_get_wtime() - tp0;
    tp0 = omp_get_wtime();

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
            g_fallback_count.fetch_add(1, std::memory_order_relaxed);
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
            g_fallback_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    const double tp_recon = omp_get_wtime() - tp0;
    tp0 = omp_get_wtime();

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

    const double tp_riemann = omp_get_wtime() - tp0;
    tp0 = omp_get_wtime();

    for (int i = 2; i < n + 2; ++i) {
        for (int k = 0; k < NVAR; ++k) {
            if (!glm && k == 5) continue;  // normal B is handled by CT Faraday update
            uc[i][k] -= (dt / dx) * (iflx[i][k] - iflx[i - 1][k]);
        }
    }
    for (int i = 2; i < n + 2; ++i) wp[i] = con2pri(uc[i], cfg.gamma);
    tp_conupd += omp_get_wtime() - tp0;

    #pragma omp atomic
    perf::t_recon += tp_recon;
    #pragma omp atomic
    perf::t_riemann += tp_riemann;
    #pragma omp atomic
    perf::t_conupd += tp_conupd;
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
        case 14: { // Same wave propagating in y — convergence test for y-sweeps
            // Mirror of test 13: guide field By=B0, perturbations in Bx/Bz,
            // periodic in y, nx=2 (thin strip in x).
            cfg.x0 = 0.0; cfg.x1 = static_cast<double>(nx) / ny;
            cfg.y0 = 0.0; cfg.y1 = 1.0;
            cfg.gamma = 5.0 / 3.0; cfg.t_end = 1.0;
            cfg.bcx = BC::Transmissive; cfg.bcy = BC::Periodic; break;
        }
        case 15: { // 2D circularly polarised Alfvén wave at 45° — Tóth (2000) §5.2
            // Wave vector k = 2π(x̂+ŷ), guide field B0(x̂+ŷ)/√2.
            // Periodic in both directions; domain [0,1]×[0,1]; nx=ny=N.
            // t_end = 1/√2: one Alfvén period → solution returns to initial state.
            cfg.x0 = 0.0; cfg.x1 = 1.0;
            cfg.y0 = 0.0; cfg.y1 = 1.0;
            cfg.gamma = 5.0 / 3.0;
            cfg.t_end = 1.0 / std::sqrt(2.0);
            cfg.bcx = BC::Periodic; cfg.bcy = BC::Periodic; break;
        }
        case 16: { // Hyper-resistivity k⁴ scaling — mode n=1 (k=2π)
            // By=A0·sin(2πx), no background field.  All three modes (16/17/18) share
            // identical η_H and t_end so γ ratios directly reflect the k⁴ operator.
            // Analytic: γ = η_H·k⁴.  Run: ./build/mhd2d 16 64 64 2 1
            cfg.x0 = 0.0; cfg.x1 = 1.0;
            cfg.y0 = 0.0; cfg.y1 = 1.0;
            cfg.gamma = 5.0 / 3.0;
            cfg.t_end = 0.05;
            cfg.bcx = BC::Periodic; cfg.bcy = BC::Periodic;
            cfg.eta_H = 1e-4;
            break;
        }
        case 17: { // Hyper-resistivity k⁴ scaling — mode n=2 (k=4π)
            cfg.x0 = 0.0; cfg.x1 = 1.0;
            cfg.y0 = 0.0; cfg.y1 = 1.0;
            cfg.gamma = 5.0 / 3.0;
            cfg.t_end = 0.05;
            cfg.bcx = BC::Periodic; cfg.bcy = BC::Periodic;
            cfg.eta_H = 1e-4;
            break;
        }
        case 18: { // Hyper-resistivity k⁴ scaling — mode n=4 (k=8π)
            cfg.x0 = 0.0; cfg.x1 = 1.0;
            cfg.y0 = 0.0; cfg.y1 = 1.0;
            cfg.gamma = 5.0 / 3.0;
            cfg.t_end = 0.05;
            cfg.bcx = BC::Periodic; cfg.bcy = BC::Periodic;
            cfg.eta_H = 1e-4;
            break;
        }
        case 19: { // Harris sheet — hyper-resistive MHD reconnection
            // Same equilibrium and domain as test 11 (resistive MHD), but
            // replaces uniform resistivity η with 4th-order hyper-resistivity η_H.
            // η_H = 0.001 gives effective dissipation at grid scale comparable to
            // η = 0.005 at the current-sheet scale λ = 0.5.
            // dt ~ h^4/(32 η_H) ≈ 2.9e-3 for 128×64 → ~7k steps for t_end=20.
            const HarrisSheetParams hp;
            cfg.x0        = -0.5 * hp.Lx; cfg.x1 = 0.5 * hp.Lx;
            cfg.y0        = -0.5 * hp.Ly; cfg.y1 = 0.5 * hp.Ly;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 25.0;
            cfg.bcx       = BC::Periodic;
            cfg.bcy       = BC::Transmissive;
            cfg.eta       = 0.0;          // no uniform resistivity
            cfg.eta_H     = 0.001;        // hyper-resistivity only
            cfg.hall_di   = 0.0;
            cfg.output_dt = 1.0;
            cfg.rho_floor = 0.1;
            cfg.p_floor   = 0.01;
            break;
        }
        case 20: { // Harris 电流片——Hall MHD + 超电阻（GEM + hyper）
            // Hall 项（d_i=1，GEM challenge）+ 超电阻（η_H=0.001）
            // dt 由 Hall RK4 CFL 主导（约 2.6e-4 for 128×64）
            const HarrisSheetParams hp;
            cfg.x0        = -0.5 * hp.Lx; cfg.x1 = 0.5 * hp.Lx;
            cfg.y0        = -0.5 * hp.Ly; cfg.y1 = 0.5 * hp.Ly;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 15.0;
            cfg.bcx       = BC::Periodic;
            cfg.bcy       = BC::Transmissive;
            cfg.eta       = 0.0;
            cfg.eta_H     = 0.001;
            cfg.hall_di   = 1.0;
            cfg.hall_stab = HallStabKind::HYPER_RES;
            cfg.output_dt = 1.0;
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            break;
        }
        case 21: { // test 20 + 非理想子循环（subcycle_nonideal=true）
            // 霍尔 CFL 从全局 dt 移除；每全局步做 N_sub 子步处理哨声波稳定性
            const HarrisSheetParams hp;
            cfg.x0        = -0.5 * hp.Lx; cfg.x1 = 0.5 * hp.Lx;
            cfg.y0        = -0.5 * hp.Ly; cfg.y1 = 0.5 * hp.Ly;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 15.0;
            cfg.bcx       = BC::Periodic;
            cfg.bcy       = BC::Transmissive;
            cfg.eta       = 0.0;
            cfg.eta_H     = 0.001;
            cfg.hall_di   = 1.0;
            cfg.hall_stab = HallStabKind::HYPER_RES;
            cfg.output_dt = 1.0;
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            cfg.subcycle_nonideal = true;
            cfg.n_subcycle_max    = 100;
            break;
        }
        case 22: { // test 21 但 eta_H=0（纯 Hall MHD，无超电阻）
            // 缺乏短波耗散，仿真在 t≈5 因 X 点奇点而崩溃
            const HarrisSheetParams hp;
            cfg.x0        = -0.5 * hp.Lx; cfg.x1 = 0.5 * hp.Lx;
            cfg.y0        = -0.5 * hp.Ly; cfg.y1 = 0.5 * hp.Ly;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 15.0;
            cfg.bcx       = BC::Periodic;
            cfg.bcy       = BC::Transmissive;
            cfg.eta       = 0.0;
            cfg.eta_H     = 0.0;
            cfg.hall_di   = 1.0;
            cfg.hall_stab = HallStabKind::NONE;
            cfg.output_dt = 1.0;
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            cfg.subcycle_nonideal = true;
            cfg.n_subcycle_max    = 100;
            break;
        }
        case 23: { // Harris 电流片——Hall MHD + Hall-HLL 扩散稳定化（Path B）
            // *** Hall-HLL 开发已停止（实验性），保留供参考；主线使用 test 21 (HYPER_RES) ***
            // 用 HLL 型哨声波速耗散替代超电阻，无子循环。
            // dt 由 HLL CFL 主导：约 9e-5 for 128×64（约 2.85× 更严格于 RK4 CFL）
            // 参考：Iwasaki & Tomida 2025（Hall-HLL 弥散近似）
            const HarrisSheetParams hp;
            cfg.x0        = -0.5 * hp.Lx; cfg.x1 = 0.5 * hp.Lx;
            cfg.y0        = -0.5 * hp.Ly; cfg.y1 = 0.5 * hp.Ly;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 15.0;
            cfg.bcx       = BC::Periodic;
            cfg.bcy       = BC::Transmissive;
            cfg.eta       = 0.0;
            cfg.eta_H     = 0.0;
            cfg.hall_di   = 1.0;
            cfg.hall_stab = HallStabKind::HALL_HLL;
            cfg.output_dt = 1.0;
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            break;
        }
        case 24: { // 1D 哨声波色散测试（验证 Hall-HLL 过阻尼特性）
            // *** Hall-HLL 开发已停止（实验性），保留供参考；主线使用 test 21 (HYPER_RES) ***
            // 背景：ρ=1，p=0.1，Bx=B₀=1（沿传播方向），δBy=A·cos(kx)，δBz=A·sin(kx)
            // 解析频率：ω = di·k² = 0.1·(2π)² ≈ 3.95
            // Hall-HLL 理论阻尼率：γ = (π/2)·ω ≈ 6.2（过阻尼，γ > ω）
            // 用途：验证 γ/ω = π/2 与 Iwasaki & Tomida 2025 预测一致
            cfg.x0     = 0.0; cfg.x1 = 1.0;
            cfg.y0     = 0.0; cfg.y1 = static_cast<double>(ny) / nx;
            cfg.gamma  = 5.0 / 3.0;
            cfg.t_end  = 1.0;   // 约 0.63 个哨声波周期；HLL 阻尼在 t≈0.5 完成
            cfg.bcx    = BC::Periodic;
            cfg.bcy    = BC::Periodic;
            cfg.eta    = 0.0;
            cfg.eta_H  = 0.0;
            cfg.hall_di   = 0.1;
            cfg.hall_stab = HallStabKind::HALL_HLL;
            cfg.output_dt = 0.02;
            break;
        }
        case 25: { // 非对称磁场重联（B1=1, B2=2, ρ1=2, ρ2=1；X点偏向弱场侧 y>0）
            const AsymHarrisParams ap;
            cfg.x0        = -0.5 * ap.Lx; cfg.x1 = 0.5 * ap.Lx;
            cfg.y0        = -0.5 * ap.Ly; cfg.y1 = 0.5 * ap.Ly;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 20.0;
            cfg.bcx       = BC::Transmissive;  // x 方向改为出流，防止重联出流周期性绕回
            cfg.bcy       = BC::Transmissive;
            cfg.eta       = 0.0;
            cfg.eta_H     = 0.001;
            cfg.hall_di   = 1.0;
            cfg.hall_stab = HallStabKind::HYPER_RES;
            cfg.output_dt = 1.0;
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            cfg.subcycle_nonideal = true;
            cfg.n_subcycle_max    = 100;
            break;
        }
        case 26: { // 非对称重联 driven 版：test 25 + y 边界恒定 E_z 注入磁通
            // 与 test 25 完全一致，仅 inflow(y) 边界从纯 transmissive 改为
            // CT 边界角点 EMF 指定恒定 E_z = -0.02（第一步试探值）。
            // 符号：本问题重联 E_z < 0（X 点 Jz < 0，峰值时 E_z ≈ -0.05），
            // 负号才对应两侧 E×B 入流（上边界 v_y=Ez/B1<0，下边界 v_y=Ez/(-B2)>0）。
            const AsymHarrisParams ap;
            cfg.x0        = -0.5 * ap.Lx; cfg.x1 = 0.5 * ap.Lx;
            cfg.y0        = -0.5 * ap.Ly; cfg.y1 = 0.5 * ap.Ly;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 20.0;
            cfg.bcx       = BC::Transmissive;  // 出流边界不变
            cfg.bcy       = BC::Transmissive;  // ρ/p/v 零梯度；面 B 由指定 E_z 驱动
            cfg.eta       = 0.0;
            cfg.eta_H     = 0.001;
            cfg.hall_di   = 1.0;
            cfg.hall_stab = HallStabKind::HYPER_RES;
            cfg.output_dt = 1.0;
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            cfg.subcycle_nonideal = true;
            cfg.n_subcycle_max    = 100;
            cfg.driven_ez = -0.02;
            break;
        }
        case 27: { // CS2008 非对称度扫描：single-kick 首峰协议（B2/ρ1 由 CLI 11/12 配置）
            // 域为 test 25 的一半（Lx=4π, Ly=2π），256×128 时 Δx 与旧域 512×256 相同。
            // η_H 按 Δx² 规则统一缩放（锚点 4e-3 @ Δx=8π/64），五档构型同值。
            // ψ0 由 CLI 按 0.75·B_eff 传入；B2/ρ1 在 IC 时经 asym_scan_params 生效。
            constexpr double pi27 = 3.14159265358979323846;
            const double Lx27 = 4.0 * pi27, Ly27 = 2.0 * pi27;
            cfg.x0        = -0.5 * Lx27; cfg.x1 = 0.5 * Lx27;
            cfg.y0        = -0.5 * Ly27; cfg.y1 = 0.5 * Ly27;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 20.0;
            cfg.bcx       = BC::Transmissive;
            cfg.bcy       = BC::Transmissive;
            cfg.eta       = 0.0;
            {
                const double dx27   = Lx27 / nx;
                const double dx_ref = 8.0 * pi27 / 64.0;
                cfg.eta_H = 4.0e-3 * (dx27 / dx_ref) * (dx27 / dx_ref);
            }
            cfg.hall_di   = 1.0;
            cfg.hall_stab = HallStabKind::HYPER_RES;
            cfg.output_dt = 0.5;   // 首峰采样密度（协议要求 Δt_out ≤ 0.5）
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            cfg.subcycle_nonideal = true;
            cfg.n_subcycle_max    = 100;
            break;
        }
        case 28: { // Dirichlet + 大域 driven 侦察（R2 构型固定，Lx=8π, Ly=2π）
            // 核心改动：y 边界 ρ/p ghost 固定为上游渐近值（质量收支闭合实验），
            // E_z 驱动照旧（默认 E0=-0.06，CLI 10 可覆盖）。
            // 注意本 case 固定 R2 构型（B2=2, ρ1=2），bc_* 渐近值与之绑定；
            // 若用 CLI 11/12 改构型，需同步修改下面的 bc_* 值。
            constexpr double pi28 = 3.14159265358979323846;
            const double Lx28 = 8.0 * pi28, Ly28 = 2.0 * pi28;
            cfg.x0        = -0.5 * Lx28; cfg.x1 = 0.5 * Lx28;
            cfg.y0        = -0.5 * Ly28; cfg.y1 = 0.5 * Ly28;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 60.0;
            cfg.bcx       = BC::Transmissive;
            cfg.bcy       = BC::Transmissive;
            cfg.eta       = 0.0;
            {
                const double dx28   = Lx28 / nx;
                const double dx_ref = 8.0 * pi28 / 64.0;
                cfg.eta_H = 4.0e-3 * (dx28 / dx_ref) * (dx28 / dx_ref);
            }
            cfg.hall_di   = 1.0;
            cfg.hall_stab = HallStabKind::HYPER_RES;
            cfg.output_dt = 1.0;
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            cfg.subcycle_nonideal = true;
            cfg.n_subcycle_max    = 100;
            cfg.driven_ez = -0.06;
            // Dirichlet 标量边界：按现有 IC 渐近值（P_total = max(B²)/2+1 = 3.0）
            //   顶 y→+∞（弱场 B1=1）：ρ=ρ1=2, p = 3 − 0.5 = 2.5
            //   底 y→−∞（强场 B2=2）：ρ=ρ2=1, p = 3 − 2.0 = 1.0
            cfg.dirichlet_y_scalars = true;
            cfg.bc_rho_top = 2.0;  cfg.bc_p_top = 2.5;
            cfg.bc_rho_bot = 1.0;  cfg.bc_p_bot = 1.0;
            break;
        }
        case 29: { // CS2008 双周期双 Harris 片（对称档 pilot；Section 3 移植）
            // 域 51.2×25.6 d_i（CS2008 的 1/4 线性尺寸），512×256 → dx=dy=0.1。
            // 双向周期；无驱动、无欧姆电阻；η_H 按 Δx² 规则由代码计算。
            // psi0 复用为扰动三件套的统一 scale（1=CS 幅值；冒烟传 1e-12 关断）。
            cfg.x0        = -25.6; cfg.x1 = 25.6;
            cfg.y0        = -12.8; cfg.y1 = 12.8;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 150.0;
            cfg.bcx       = BC::Periodic;
            cfg.bcy       = BC::Periodic;
            cfg.eta       = 0.0;
            {
                const double dx29   = (cfg.x1 - cfg.x0) / nx;
                const double dx_ref = 8.0 * 3.14159265358979323846 / 64.0;
                cfg.eta_H = 4.0e-3 * (dx29 / dx_ref) * (dx29 / dx_ref);
            }
            cfg.hall_di   = 1.0;
            cfg.hall_stab = HallStabKind::HYPER_RES;
            cfg.output_dt = 1.0;
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            cfg.subcycle_nonideal = true;
            cfg.n_subcycle_max    = 100;
            cfg.psi0      = 1.0;   // 默认全幅扰动；CLI 9 可覆盖
            break;
        }
        case 30: { // 电阻性非对称度扫描：test 27 镜像，耗散机制换为物理电阻
            // 耗散无关性检验（CS2007）：同 R1–R5 构型、同域、同 kick ψ₀=0.375·B_eff，
            // hall_di=0（无 Hall）、η_H=0（无超电阻）、η=0.005（Sweet-Parker 制度）。
            // S_eff ~ v_out·L/η ≈ 10³，低于 plasmoid 阈值 ~10⁴。
            // t_end=60（电阻慢）；Δt_out=0.5 保持首峰采样密度。
            constexpr double pi30 = 3.14159265358979323846;
            const double Lx30 = 4.0 * pi30, Ly30 = 2.0 * pi30;
            cfg.x0        = -0.5 * Lx30; cfg.x1 = 0.5 * Lx30;
            cfg.y0        = -0.5 * Ly30; cfg.y1 = 0.5 * Ly30;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 60.0;
            cfg.bcx       = BC::Transmissive;
            cfg.bcy       = BC::Transmissive;
            cfg.eta       = 0.005;
            cfg.eta_H     = 0.0;
            cfg.hall_di   = 0.0;
            cfg.hall_stab = HallStabKind::HYPER_RES;
            cfg.output_dt = 0.5;
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            cfg.subcycle_nonideal = false;
            break;
        }
        case 31: { // test29 campaign 档：CS2008 双片，双倍域，三层 I/O（本机版）
            // 域由格数定，dx=dy=0.1 固定（10 cpdi）：Lx=0.1·nx, Ly=0.1·ny。
            //   1024×512 → 102.4×51.2（Stage 0 正式档）
            //    512×256 →  51.2×25.6（半尺寸成本校准/续跑实测）
            // 物理与 case 29 pilot 逐项相同：双向周期、η=0、Hall d_i=1、
            // η_H 按 Δx² 规则由实际 dx 代码计算、扰动三件套 scale=psi0。
            // 传统快照关闭（output_dt=0），由 L1/L2/L3 三层 I/O 取代。
            const double Lx31 = 0.1 * nx, Ly31 = 0.1 * ny;
            cfg.x0        = -0.5 * Lx31; cfg.x1 = 0.5 * Lx31;
            cfg.y0        = -0.5 * Ly31; cfg.y1 = 0.5 * Ly31;
            cfg.gamma     = 5.0 / 3.0;
            cfg.t_end     = 400.0;
            cfg.bcx       = BC::Periodic;
            cfg.bcy       = BC::Periodic;
            cfg.eta       = 0.0;
            {
                const double dx31   = Lx31 / nx;
                const double dx_ref = 8.0 * 3.14159265358979323846 / 64.0;
                cfg.eta_H = 4.0e-3 * (dx31 / dx_ref) * (dx31 / dx_ref);
            }
            cfg.hall_di   = 1.0;
            cfg.hall_stab = HallStabKind::HYPER_RES;
            cfg.output_dt = 0.0;
            cfg.rho_floor = 0.02;
            cfg.p_floor   = 0.005;
            cfg.subcycle_nonideal = true;
            cfg.n_subcycle_max    = 100;
            cfg.psi0      = 1.0;
            cfg.campaign_io = true;
            cfg.l1_dt     = 0.5;
            cfg.ckpt_dt   = 10.0;
            break;
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

void apply_bc(Grid& w, int nx, int ny, BC bcx, BC bcy, const RunConfig* cfg) {
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

    // Dirichlet inflow 标量（test 28）：y ghost 的 ρ/p 覆写为固定上游值，
    // 提供不衰减的质量/压强库；速度与 B 保持上面的零梯度填充。
    if (cfg && cfg->dirichlet_y_scalars && bcy == BC::Transmissive) {
        for (int i = 0; i < nx + 4; ++i) {
            w[i][0][0] = cfg->bc_rho_bot;      w[i][1][0] = cfg->bc_rho_bot;
            w[i][0][4] = cfg->bc_p_bot;        w[i][1][4] = cfg->bc_p_bot;
            w[i][ny + 2][0] = cfg->bc_rho_top; w[i][ny + 3][0] = cfg->bc_rho_top;
            w[i][ny + 2][4] = cfg->bc_p_top;   w[i][ny + 3][4] = cfg->bc_p_top;
        }
    }
}

void apply_floor(Grid& w, int nx, int ny, const RunConfig& cfg) {
    if (cfg.rho_floor <= 0.0 && cfg.p_floor <= 0.0) return;
    const double gm1 = cfg.gamma - 1.0;
    for (int i = 2; i < nx + 2; ++i)
        for (int j = 2; j < ny + 2; ++j) {
            Vec& p = w[i][j];
            if (cfg.rho_floor > 0.0 && p[0] < cfg.rho_floor) { p[0] = cfg.rho_floor; ++g_floor_rho_count; }
            if (cfg.p_floor   > 0.0 && p[4] < cfg.p_floor)   { p[4] = cfg.p_floor;   ++g_floor_p_count; }
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
    double smax = 1e-10, ch_loc = 0.0, max_va2 = 0.0, max_b2_rho2 = 0.0, max_cf2_hll = 0.0;
    #pragma omp parallel for collapse(2) \
        reduction(max:smax,ch_loc,max_va2,max_b2_rho2,max_cf2_hll) schedule(static)
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
            double B2 = p[5]*p[5] + p[6]*p[6] + p[7]*p[7];
            // 霍尔 CFL 两种方案分别需要 |B|/√ρ 和 |B|/ρ
            double rho_va = cfg.hall_di > 0.0 ? std::max(p[0], cfg.rho_floor) : std::max(p[0], 1e-14);
            max_va2     = std::max(max_va2,    B2 / rho_va);                        // 用于 RK4 CFL
            max_b2_rho2 = std::max(max_b2_rho2, B2 / (rho_va*rho_va));            // 用于 HLL CFL
            if (cfg.hall_di > 0.0 && cfg.hall_stab == HallStabKind::HALL_HLL)     // 用于 c_stab CFL（仅 HALL_HLL）
                max_cf2_hll = std::max(max_cf2_hll, (cfg.gamma * p[4] + B2) / rho_va);
        }
    }
    divb.update_characteristic_speed(ch_loc);

    if (cfg.hall_di > 0.0 && !cfg.subcycle_nonideal) {
        constexpr double pi = 3.14159265358979323846;
        double mincell = std::min(dx, dy);
        if (cfg.hall_stab == HallStabKind::HALL_HLL) {
            // c_stab = c_f + c_w 稳定性条件：c_stab·dt/h ≤ 1
            // c_f_max = sqrt(max(γp+B²)/ρ)：快磁声速，在磁零点退化为声速（物理底）
            // c_w_max = π·di·max(|B|/ρ)：哨声波速（不加人工常数 floor）
            double c_f_max  = std::sqrt(max_cf2_hll);
            double c_w_max  = cfg.hall_di * pi * std::sqrt(max_b2_rho2);
            double smax_hll = (c_f_max + c_w_max) / mincell;
            smax = std::max(smax, smax_hll);
        } else {
            // RK4 稳定性（|ω·dt| < 2√2 ≈ 2.83），使用 vA = |B|/√ρ
            double smax_hall = cfg.hall_di * (pi*pi/2.83) * std::sqrt(max_va2)
                               / (mincell * mincell);
            smax = std::max(smax, smax_hall);
        }
    }
    double dt = cfg.cfl / smax;
    if (cfg.eta_H > 0.0) {
        double mincell = std::min(dx, dy);
        // 显式 Euler 双调和稳定性：dt ≤ h⁴ / (32·η_H)
        double dt_hyper = std::pow(mincell, 4) / (32.0 * cfg.eta_H);
        dt = std::min(dt, dt_hyper);
    }
    return dt;
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
                case 14: {
                    // Same wave propagating in y — mirror of test 13.
                    // Guide field By = B0; perturbations in Bx/Bz plane.
                    constexpr double pi = 3.14159265358979323846;
                    const double k  = 2.0 * pi;
                    const double B0 = 1.0, rho0 = 1.0;
                    const double va = B0 / std::sqrt(rho0);
                    const double Bperp = 0.1 * B0;
                    w[i][j] = { rho0,
                               -va * Bperp * std::sin(k * y) / B0,
                                0.0,
                                va * Bperp * std::cos(k * y) / B0,
                                0.1,
                                Bperp * std::sin(k * y),
                                B0,
                               -Bperp * std::cos(k * y),
                                0.0 };
                    break;
                }
                case 15: {
                    // 2D circularly polarised Alfvén wave at 45° (Tóth 2000 §5.2).
                    // Guide field: B0 along (x̂+ŷ)/√2.
                    // Perturbation phase: φ = 2π(x+y).
                    constexpr double pi = 3.14159265358979323846;
                    constexpr double s2 = 1.41421356237309504880; // √2
                    const double phi    = 2.0 * pi * (x + y);
                    const double B0 = 1.0, Bperp = 0.1;
                    w[i][j] = { 1.0,
                                 Bperp / s2 * std::sin(phi),   // vx = −δBx (forward: v=−B)
                                -Bperp / s2 * std::sin(phi),   // vy = −δBy
                                -Bperp      * std::cos(phi),   // vz = −Bz  (forward)
                                 0.1,
                                 B0 / s2 - Bperp / s2 * std::sin(phi),  // Bx
                                 B0 / s2 + Bperp / s2 * std::sin(phi),  // By
                                +Bperp      * std::cos(phi),              // Bz (forward: sign +)
                                 0.0 };
                    break;
                }
                case 16: {
                    // k=2π (n=1).  Analytic: A(t) = A0·exp(−η_H·(2π)⁴·t).
                    constexpr double pi = 3.14159265358979323846;
                    const double A0 = 0.1;
                    w[i][j] = { 1.0, 0.0, 0.0, 0.0, 0.1,
                                 0.0,
                                 A0 * std::sin(2.0 * pi * x),
                                 0.0, 0.0 };
                    break;
                }
                case 17: {
                    // k=4π (n=2).  Analytic: A(t) = A0·exp(−η_H·(4π)⁴·t).
                    constexpr double pi = 3.14159265358979323846;
                    const double A0 = 0.1;
                    w[i][j] = { 1.0, 0.0, 0.0, 0.0, 0.1,
                                 0.0,
                                 A0 * std::sin(4.0 * pi * x),
                                 0.0, 0.0 };
                    break;
                }
                case 18: {
                    // k=8π (n=4).  Analytic: A(t) = A0·exp(−η_H·(8π)⁴·t).
                    constexpr double pi = 3.14159265358979323846;
                    const double A0 = 0.1;
                    w[i][j] = { 1.0, 0.0, 0.0, 0.0, 0.1,
                                 0.0,
                                 A0 * std::sin(8.0 * pi * x),
                                 0.0, 0.0 };
                    break;
                }
                case 19: {
                    // Harris sheet — hyper-resistive MHD (same IC as test 11).
                    w[i][j] = harris_cell_ic(x, y, HarrisSheetParams{});
                    break;
                }
                case 20:
                case 21:
                case 22:
                case 23: {
                    // Hall + hyper-res (test 20/21)、纯 Hall (test 22) 或 Hall-HLL (test 23)
                    // 均使用 GEM 动理论密度 IC（与 test 12 相同）
                    const HarrisSheetParams hp;
                    Vec ic = harris_cell_ic(x, y, hp);
                    const double s = 1.0 / std::cosh(y / hp.lam);
                    ic[0] = 0.2 + s * s;   // GEM：n_bg=0.2，n0=1.0
                    w[i][j] = ic;
                    break;
                }
                case 24: {
                    // 1D 哨声波测试：背景场 Bx=1（沿 x），横向扰动 δBy/δBz
                    constexpr double pi = 3.14159265358979323846;
                    const double k  = 2.0 * pi / (cfg.x1 - cfg.x0);
                    const double A  = 0.01;
                    w[i][j] = { 1.0,                    // ρ
                                0.0, 0.0, 0.0,          // vx, vy, vz
                                0.1,                    // p
                                1.0,                    // Bx（背景场，哨声波沿此方向传播）
                                A * std::cos(k * x),    // δBy
                                A * std::sin(k * x),    // δBz（圆偏振）
                                0.0 };                  // ψ
                    break;
                }
                case 25: case 26: {
                    AsymHarrisParams ap25;
                    ap25.psi0 = cfg.psi0;
                    w[i][j] = asym_cell_ic(x, y, ap25);
                    break;
                }
                case 27: case 28: case 30: {
                    w[i][j] = asym_cell_ic(x, y, asym_scan_params(cfg));
                    break;
                }
                case 29: {
                    w[i][j] = dh_cell_ic(x, y, dh_params_from_config(cfg));
                    break;
                }
                case 31: {
                    w[i][j] = adh_cell_ic(x, y, adh_params_from_config(cfg));
                    break;
                }
                default:
                    throw std::runtime_error("Unknown test id");
            }
        }
    }
}

namespace {
// ============================================================================
// test 31 campaign 三层 I/O（L1 schema 冻结于 output/test29_campaign/
// l1_freeze_report.md；算法与 pilot29_diag.py 逐式一致）
// ============================================================================

struct L1Row {
    double min_rho = 0, min_p = 0, max_divb = 0, max_v = 0, max_bz = 0;
    // 索引 0 = 上片 (y=+Ly/4)，1 = 下片 (y=−Ly/4)
    double psi[2] = {0, 0}, xX[2] = {0, 0}, xO[2] = {0, 0};
    double ezB[2] = {0, 0}, wisl[2] = {0, 0}, vout[2] = {0, 0};
    // schema v2（AB/AN/ABN 档物理信号）：片位 y（|Jz| 峰行，随动/漂移信号）
    // 与 X 列上片两侧 ±3.0 处的带符号 vy（入流；随动修正 dyX/dt 离线做）
    double yX[2] = {0, 0}, vinp[2] = {0, 0}, vinm[2] = {0, 0};
};

struct CampaignState {
    std::string dir, l1_path, ckpt_path;
    std::ofstream l1;
    double next_l1 = 0.0, next_ckpt = 0.0, next_flush = 50.0, next_coarse = 50.0;
    int n_l2 = 0;
    int fired_plateau = 0, fired_island = 0, fired_burst = 0, armed_island = 0;
    std::vector<double> pending;   // 平台窗内追加帧的计划时刻
    std::vector<double> ea;        // EA 滚动缓冲（40 行 = Δt 20）
    double prev_psi[2] = {0, 0};
    double prev_t = -1.0;
    std::vector<double> az, jz, lap;   // 工作数组（nx*ny，重复使用）
};

// 通量函数 + 片诊断（cell 中心量；与 pilot 的 numpy 实现同式）。
L1Row campaign_l1_compute(const Grid& w, const RunConfig& cfg,
                          double dx, double dy, CampaignState& cs) {
    const int nx = cfg.nx, ny = cfg.ny;
    const double Ly = cfg.y1 - cfg.y0;
    auto W = [&](int i, int j, int k) -> double { return w[i + 2][j + 2][k]; };
    auto id = [&](int i, int j) { return i * ny + j; };
    cs.az.assign((size_t)nx * ny, 0.0);
    cs.jz.assign((size_t)nx * ny, 0.0);
    cs.lap.assign((size_t)nx * ny, 0.0);
    // 2-leg 线积分通量函数（首列沿 y 积 Bx，再沿 x 积 By 的面均值）
    {
        double a = 0.0;
        for (int j = 0; j < ny; ++j) { a += W(0, j, 5) * dy; cs.az[id(0, j)] = a; }
        for (int i = 1; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                cs.az[id(i, j)] = cs.az[id(i - 1, j)]
                    - 0.5 * (W(i - 1, j, 6) + W(i, j, 6)) * dx;
    }
    L1Row r;
    for (int i = 0; i < nx; ++i) {
        const int ip = (i + 1) % nx, im = (i + nx - 1) % nx;
        for (int j = 0; j < ny; ++j) {
            const int jp = (j + 1) % ny, jm = (j + ny - 1) % ny;
            cs.jz[id(i, j)] = (W(ip, j, 6) - W(im, j, 6)) / (2.0 * dx)
                            - (W(i, jp, 5) - W(i, jm, 5)) / (2.0 * dy);
            r.max_bz = std::max(r.max_bz, std::fabs(W(i, j, 7)));
        }
    }
    for (int i = 0; i < nx; ++i) {
        const int ip = (i + 1) % nx, im = (i + nx - 1) % nx;
        for (int j = 0; j < ny; ++j) {
            const int jp = (j + 1) % ny, jm = (j + ny - 1) % ny;
            cs.lap[id(i, j)] =
                (cs.jz[id(ip, j)] - 2.0 * cs.jz[id(i, j)] + cs.jz[id(im, j)]) / (dx * dx)
              + (cs.jz[id(i, jp)] - 2.0 * cs.jz[id(i, j)] + cs.jz[id(i, jm)]) / (dy * dy);
        }
    }
    const double ys[2] = { +0.25 * Ly, -0.25 * Ly };
    for (int s = 0; s < 2; ++s) {
        // 片行 = 带内 |y−ys|≤5.0 中 x 平均 |Jz| 最大的行
        // （v2 起带宽 2.5→5.0：非对称档片沿入流方向向强场侧漂移，需跟踪）
        int jrow = -1; double best = -1.0;
        for (int j = 0; j < ny; ++j) {
            const double yc = cfg.y0 + (j + 0.5) * dy;
            if (std::fabs(yc - ys[s]) > 5.0) continue;
            double m = 0.0;
            for (int i = 0; i < nx; ++i) m += std::fabs(cs.jz[id(i, j)]);
            if (m > best) { best = m; jrow = j; }
        }
        r.yX[s] = cfg.y0 + (jrow + 0.5) * dy;
        int iO = 0, iX = 0;
        for (int i = 1; i < nx; ++i) {
            if (cs.az[id(i, jrow)] > cs.az[id(iO, jrow)]) iO = i;
            if (cs.az[id(i, jrow)] < cs.az[id(iX, jrow)]) iX = i;
        }
        r.psi[s] = cs.az[id(iO, jrow)] - cs.az[id(iX, jrow)];
        // X = |Jz| 更大的极值点（电流片在 X 处收缩）
        const int iXpt = (std::fabs(cs.jz[id(iX, jrow)]) >= std::fabs(cs.jz[id(iO, jrow)]))
                         ? iX : iO;
        const int iOpt = (iXpt == iX) ? iO : iX;
        r.xX[s] = cfg.x0 + (iXpt + 0.5) * dx;
        r.xO[s] = cfg.x0 + (iOpt + 0.5) * dx;
        // Method B：X 点完整 Ez = −(v×B)z + Hall + hyper
        {
            const int i = iXpt, j = jrow;
            const int ip = (i + 1) % nx, im = (i + nx - 1) % nx;
            const int jp = (j + 1) % ny, jm = (j + ny - 1) % ny;
            const double rho = std::max(W(i, j, 0), std::max(cfg.rho_floor, 1e-10));
            const double vx = W(i, j, 1), vy = W(i, j, 2);
            const double Bx = W(i, j, 5), By = W(i, j, 6);
            const double Jx = (W(i, jp, 7) - W(i, jm, 7)) / (2.0 * dy);
            const double Jy = -(W(ip, j, 7) - W(im, j, 7)) / (2.0 * dx);
            r.ezB[s] = -(vx * By - vy * Bx)
                       + (cfg.hall_di / rho) * (Jx * By - Jy * Bx)
                       - cfg.eta_H * cs.lap[id(i, j)];
        }
        // v2：X 列上片两侧 ±3.0 处的带符号 vy（入流诊断原料；周期折返安全）
        {
            const int joff = int(3.0 / dy + 0.5);
            r.vinp[s] = W(iXpt, (jrow + joff) % ny, 2);
            r.vinm[s] = W(iXpt, (jrow - joff + ny) % ny, 2);
        }
        // 岛宽：O 列上过分离线值的等值段（v2：锚定当前片行 jrow 而非初始
        // 片位，漂移后不失锚；无岛时退化为全高——消费端按 armed 规则处理）
        {
            const double sep = cs.az[id(iXpt, jrow)];
            const int j0 = jrow;
            const double sgn = (cs.az[id(iOpt, j0)] - sep >= 0.0) ? 1.0 : -1.0;
            int jlo = j0, jhi = j0;
            while (jlo > 0 && sgn * (cs.az[id(iOpt, jlo - 1)] - sep) > 0.0) --jlo;
            while (jhi < ny - 1 && sgn * (cs.az[id(iOpt, jhi + 1)] - sep) > 0.0) ++jhi;
            r.wisl[s] = (jhi - jlo + 1) * dy;
        }
        // 片带出流：|y−ys|≤1 内 max|vx|
        for (int j = 0; j < ny; ++j) {
            const double yc = cfg.y0 + (j + 0.5) * dy;
            if (std::fabs(yc - ys[s]) > 1.0) continue;
            for (int i = 0; i < nx; ++i)
                r.vout[s] = std::max(r.vout[s], std::fabs(W(i, j, 1)));
        }
    }
    return r;
}

// L2 事件快照：小端二进制，头 6 个 double {31, nx_out, ny_out, stride, t, nvar}
// 后接 float32 数据（k 外层、j 中层、i 内层；stride 为格点抽样步长）。
void write_l2_frame(const Grid& w, const RunConfig& cfg, double t,
                    const std::string& tag, int stride, CampaignState& cs) {
    // L2_STRIDE 环境变量：所有 L2 帧的抽样步长统一乘以该系数（默认 1 =
    // 行为不变）。纯 I/O 体积开关，供大箱（2048×1024）在 1.2GB quota 下
    // 使用（L2_STRIDE=4 → 单帧 75.5MB→4.7MB）。帧头 stride 字段如实记录，
    // 读取端无需改动。
    static const int l2_mult = [] {
        const char* s = std::getenv("L2_STRIDE");
        const int v = s ? std::atoi(s) : 1;
        if (v > 1) std::cout << "[campaign] L2_STRIDE=" << v
                             << " (all L2 frames coarsened)\n";
        return std::max(1, v);
    }();
    stride *= l2_mult;
    if (cs.n_l2 >= 15) {
        std::cout << "[campaign] L2 cap (15) reached, frame '" << tag
                  << "' at t=" << t << " skipped\n";
        return;
    }
    char name[96];
    std::snprintf(name, sizeof name, "l2_t%07.1f_%s.f32", t, tag.c_str());
    const int nxo = cfg.nx / stride, nyo = cfg.ny / stride;
    std::ofstream f(cs.dir + "/" + name, std::ios::binary);
    const double hdr[6] = { 31.0, double(nxo), double(nyo), double(stride),
                            t, double(NVAR) };
    f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
    std::vector<float> buf(nxo);
    for (int k = 0; k < NVAR; ++k)
        for (int j = 0; j < nyo; ++j) {
            for (int i = 0; i < nxo; ++i)
                buf[i] = float(w[i * stride + 2][j * stride + 2][k]);
            f.write(reinterpret_cast<const char*>(buf.data()),
                    std::streamsize(buf.size() * sizeof(float)));
        }
    ++cs.n_l2;
    std::cout << "[campaign] L2 frame " << cs.n_l2 << "/15: " << name << "\n";
}

// L3 滚动 checkpoint：写 .tmp 后原子替换，保留最近 1 份。
void write_ckpt(const CampaignState& cs, const Grid& w, const FaceField2D* ff,
                const RunConfig& cfg, double t, int step) {
    const std::string tmp = cs.ckpt_path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        const int32_t magic = 31337, nx = cfg.nx, ny = cfg.ny;
        auto W32 = [&](int32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
        auto W64 = [&](int64_t v) { f.write(reinterpret_cast<const char*>(&v), 8); };
        auto WD  = [&](double v)  { f.write(reinterpret_cast<const char*>(&v), 8); };
        W32(magic); W32(nx); W32(ny);
        WD(t); W64(step);
        W64(g_floor_rho_count); W64(g_floor_p_count);
        W64(g_fallback_count.load());
        WD(cs.next_l1); WD(cs.next_ckpt); WD(cs.next_flush); WD(cs.next_coarse);
        W32(cs.n_l2); W32(cs.fired_plateau); W32(cs.fired_island);
        W32(cs.fired_burst); W32(cs.armed_island);
        W32(int32_t(cs.pending.size()));
        for (double v : cs.pending) WD(v);
        W32(int32_t(cs.ea.size()));
        for (double v : cs.ea) WD(v);
        WD(cs.prev_psi[0]); WD(cs.prev_psi[1]); WD(cs.prev_t);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                f.write(reinterpret_cast<const char*>(w[i + 2][j + 2].data()),
                        NVAR * 8);
        for (int i = 0; i <= nx; ++i)
            f.write(reinterpret_cast<const char*>(ff->bx[i].data()),
                    std::streamsize(ny * 8));
        for (int i = 0; i < nx; ++i)
            f.write(reinterpret_cast<const char*>(ff->by[i].data()),
                    std::streamsize((ny + 1) * 8));
    }
    std::remove(cs.ckpt_path.c_str());
    if (std::rename(tmp.c_str(), cs.ckpt_path.c_str()) != 0)
        std::cerr << "[campaign] WARNING: checkpoint rename failed\n";
}

bool load_ckpt(CampaignState& cs, Grid& w, FaceField2D* ff,
               const RunConfig& cfg, double& t, int& step) {
    std::ifstream f(cs.ckpt_path, std::ios::binary);
    if (!f) return false;
    int32_t magic = 0, nx = 0, ny = 0;
    auto R32 = [&]() { int32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v; };
    auto R64 = [&]() { int64_t v; f.read(reinterpret_cast<char*>(&v), 8); return v; };
    auto RD  = [&]() { double v;  f.read(reinterpret_cast<char*>(&v), 8); return v; };
    magic = R32(); nx = R32(); ny = R32();
    if (magic != 31337 || nx != cfg.nx || ny != cfg.ny) {
        std::cerr << "[campaign] checkpoint mismatch (magic/nx/ny), fresh start\n";
        return false;
    }
    t = RD(); step = int(R64());
    g_floor_rho_count = R64(); g_floor_p_count = R64();
    g_fallback_count.store(R64());
    cs.next_l1 = RD(); cs.next_ckpt = RD(); cs.next_flush = RD(); cs.next_coarse = RD();
    cs.n_l2 = R32(); cs.fired_plateau = R32(); cs.fired_island = R32();
    cs.fired_burst = R32(); cs.armed_island = R32();
    cs.pending.assign(size_t(R32()), 0.0);
    for (double& v : cs.pending) v = RD();
    cs.ea.assign(size_t(R32()), 0.0);
    for (double& v : cs.ea) v = RD();
    cs.prev_psi[0] = RD(); cs.prev_psi[1] = RD(); cs.prev_t = RD();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            f.read(reinterpret_cast<char*>(w[i + 2][j + 2].data()), NVAR * 8);
    for (int i = 0; i <= nx; ++i)
        f.read(reinterpret_cast<char*>(ff->bx[i].data()), std::streamsize(ny * 8));
    for (int i = 0; i < nx; ++i)
        f.read(reinterpret_cast<char*>(ff->by[i].data()), std::streamsize((ny + 1) * 8));
    return bool(f);
}

// L1 行 + L2 触发器。返回 false 表示状态非物理（调用方应终止主循环）。
bool campaign_l1_tick(CampaignState& cs, const Grid& w, const RunConfig& cfg,
                      DivergenceController& divb, double dx, double dy,
                      double t, double dt, int step) {
    const double Ly = cfg.y1 - cfg.y0;
    Diagnostics dg = compute_diagnostics(w, cfg.nx, cfg.ny, dx, dy, divb.face_field());
    L1Row r = campaign_l1_compute(w, cfg, dx, dy, cs);
    const int nsub = cfg.subcycle_nonideal ? divb.last_n_sub() : 0;
    char buf[1024];
    std::snprintf(buf, sizeof buf,
        "%.4f,%.6e,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%lld,%lld,%lld,"
        "%.6e,%.4f,%.4f,%.6e,%.4f,%.6e,"
        "%.6e,%.4f,%.4f,%.6e,%.4f,%.6e,"
        "%.4f,%.6e,%.6e,%.4f,%.6e,%.6e\n",
        t, dt, step, nsub, dg.min_rho, dg.min_p, dg.max_divB, dg.max_v,
        r.max_bz, g_floor_rho_count, g_floor_p_count, g_fallback_count.load(),
        r.psi[0], r.xX[0], r.xO[0], r.ezB[0], r.wisl[0], r.vout[0],
        r.psi[1], r.xX[1], r.xO[1], r.ezB[1], r.wisl[1], r.vout[1],
        r.yX[0], r.vinp[0], r.vinm[0], r.yX[1], r.vinp[1], r.vinm[1]);
    cs.l1 << buf;
    if (t >= cs.next_flush - 1e-9) { cs.l1.flush(); cs.next_flush += 50.0; }

    // --- L2 触发器 ---
    // EA 滚动缓冲（两片平均的 Method A 速率）
    if (cs.prev_t >= 0.0 && t > cs.prev_t) {
        const double ea = 0.5 * (std::fabs(r.psi[0] - cs.prev_psi[0])
                               + std::fabs(r.psi[1] - cs.prev_psi[1])) / (t - cs.prev_t);
        cs.ea.push_back(ea);
        if (cs.ea.size() > 40) cs.ea.erase(cs.ea.begin());
    }
    cs.prev_psi[0] = r.psi[0]; cs.prev_psi[1] = r.psi[1]; cs.prev_t = t;
    // 平台起点：40 行（Δt=20）全部落在中位数 ±20% 内且中位数 > 1e-3
    if (!cs.fired_plateau && cs.ea.size() >= 40) {
        std::vector<double> tmp(cs.ea);
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        const double med = tmp[tmp.size() / 2];
        bool flat = med > 1e-3;
        for (double e : cs.ea) if (std::fabs(e - med) >= 0.2 * med) { flat = false; break; }
        if (flat) {
            cs.fired_plateau = 1;
            write_l2_frame(w, cfg, t, "plateau0", 1, cs);
            cs.pending = { t + 5.0, t + 10.0, t + 15.0 };
        }
    }
    // 平台窗内追加帧
    for (size_t k = 0; k < cs.pending.size();) {
        if (t >= cs.pending[k] - 1e-9) {
            char tag[16]; std::snprintf(tag, sizeof tag, "plateau%d",
                                        int(4 - cs.pending.size() + 1));
            write_l2_frame(w, cfg, t, tag, 1, cs);
            cs.pending.erase(cs.pending.begin() + long(k));
        } else ++k;
    }
    // 岛宽过 Ly/2（armed 规则 + 2026-07-18 修复×2，经用户批准）：
    //  1) 退化排除：wisl==Ly 是等值段扫描失败的带内哨兵（非测量值），
    //     曾致 AB2 t=4.5 / AB1big t=8 误燃烧帧位；有效帧须 < Ly-0.5dy。
    //  2) ψ 门槛随域高缩放（0.15 标定于 Ly=51.2；固定值在大箱被
    //     kick 初始通量 ψ0=0.01Ly/π 直接越过而失效）。Ly=51.2 档行为不变。
    const bool cross0 = r.wisl[0] >= 0.5 * Ly && r.wisl[0] < Ly - 0.5 * dy;
    const bool cross1 = r.wisl[1] >= 0.5 * Ly && r.wisl[1] < Ly - 0.5 * dy;
    const double wmax = std::max(r.wisl[0], r.wisl[1]);
    if (!cs.armed_island && wmax < 0.25 * Ly) cs.armed_island = 1;
    if (cs.armed_island && !cs.fired_island && (cross0 || cross1)
        && std::max(r.psi[0], r.psi[1]) > 0.15 * (Ly / 51.2)) {
        cs.fired_island = 1;
        write_l2_frame(w, cfg, t, "islandLy2", 1, cs);
    }
    // 暴发：max|v| ≥ 2 cA0
    if (!cs.fired_burst && dg.max_v >= 2.0) {
        cs.fired_burst = 1;
        write_l2_frame(w, cfg, t, "burst", 1, cs);
    }
    // 每 t=50 粗网帧（stride 2；预留 3 帧给事件 → n_l2 < 12 才写）
    if (t >= cs.next_coarse - 1e-9) {
        if (cs.n_l2 < 12 && t < cfg.t_end - 1.0)
            write_l2_frame(w, cfg, t, "coarse", 2, cs);
        while (cs.next_coarse <= t + 1e-9) cs.next_coarse += 50.0;
    }

    return std::isfinite(dg.max_v) && std::isfinite(r.psi[0]) && std::isfinite(r.psi[1])
        && std::isfinite(dg.max_divB) && dg.min_rho > 0.0 && dg.min_p > 0.0;
}

// 初始化（含 RESUME=1 续跑）。返回是否续跑。
bool campaign_init(CampaignState& cs, Grid& w, DivergenceController& divb,
                   const RunConfig& cfg, double dx, double dy,
                   double& t, int& step) {
    cs.dir = cfg_outdir(cfg);
    cs.l1_path = cs.dir + "/l1.csv";
    cs.ckpt_path = cs.dir + "/ckpt.bin";
    cs.next_ckpt = cfg.ckpt_dt;
    g_floor_rho_count = 0; g_floor_p_count = 0; g_fallback_count.store(0);

    const char* rs = std::getenv("RESUME");
    const bool want_resume = (rs && rs[0] == '1');
    if (want_resume) {
        FaceField2D* ff = divb.mutable_face_field();
        if (ff && load_ckpt(cs, w, ff, cfg, t, step)) {
            // 截掉 checkpoint 之后的 L1 行（避免重复区间）
            std::vector<std::string> keep;
            {
                std::ifstream in(cs.l1_path);
                std::string line;
                while (std::getline(in, line)) {
                    if (line.empty()) continue;
                    if (line[0] == '#' || line[0] == 't') { keep.push_back(line); continue; }
                    if (std::atof(line.c_str()) <= t + 1e-6) keep.push_back(line);
                }
            }
            std::ofstream outf(cs.l1_path);
            for (const auto& s : keep) outf << s << '\n';
            cs.l1.open(cs.l1_path, std::ios::app);
            std::cout << "[campaign] RESUMED from checkpoint: t=" << t
                      << " step=" << step << " (L1 rows kept: "
                      << keep.size() << ")\n";
            return true;
        }
        std::cout << "[campaign] RESUME=1 but no usable checkpoint, fresh start\n";
    }
    // 全新启动：表头（含 t=0 压平衡 RMS 硬检查值）+ t=0 行 + L2 t0 帧
    double pt_mean = 0.0, pt_rms = 0.0;
    {
        const long long n = (long long)cfg.nx * cfg.ny;
        for (int i = 0; i < cfg.nx; ++i)
            for (int j = 0; j < cfg.ny; ++j) {
                const Vec& p = w[i + 2][j + 2];
                pt_mean += p[4] + 0.5 * (p[5] * p[5] + p[6] * p[6] + p[7] * p[7]);
            }
        pt_mean /= double(n);
        for (int i = 0; i < cfg.nx; ++i)
            for (int j = 0; j < cfg.ny; ++j) {
                const Vec& p = w[i + 2][j + 2];
                const double d = p[4] + 0.5 * (p[5] * p[5] + p[6] * p[6] + p[7] * p[7]) - pt_mean;
                pt_rms += d * d;
            }
        pt_rms = std::sqrt(pt_rms / double(n)) / pt_mean;
    }
    cs.l1.open(cs.l1_path);
    cs.l1 << "# tier = " << cfg.label << " (test 31 campaign; schema v2: l1_freeze_report.md)\n"
          << "# Lx = " << (cfg.x1 - cfg.x0) << ", Ly = " << (cfg.y1 - cfg.y0)
          << ", nx = " << cfg.nx << ", ny = " << cfg.ny << "\n"
          << "# eta_H = " << cfg.eta_H << ", hall_di = " << cfg.hall_di
          << ", scale = " << cfg.psi0 << ", w0 = 2.0, beta_min = 4.0\n"
          << "# B01 = " << cfg.dh_B01 << ", B02 = " << cfg.dh_B02
          << ", rho01 = " << cfg.dh_rho01 << ", rho02 = " << cfg.dh_rho02 << "\n"
          << "# ptot_rms0 = " << std::scientific << pt_rms << std::defaultfloat << "\n"
          << "t,dt,step,nsub,min_rho,min_p,max_divb,max_v,max_bz,"
             "floor_rho,floor_p,fallback,"
             "up_psi,up_xX,up_xO,up_ezB,up_wisl,up_vout,"
             "lo_psi,lo_xX,lo_xO,lo_ezB,lo_wisl,lo_vout,"
             "up_yX,up_vinp,up_vinm,lo_yX,lo_vinp,lo_vinm\n";
    std::cout << "[campaign] ptot_rms0 = " << pt_rms << " (hard check #1)\n";

    // 广义双片 IC 静态自检（test 31）：两片上下游 B/ρ 采样应互为镜像，
    // 逐档参数按 β_min 规则生成并打印，供与 CS2008 Table 1 对照。
    if (cfg.test == 31) {
        const double Ly31 = cfg.y1 - cfg.y0, q = 0.25 * Ly31;
        const double dyc = Ly31 / cfg.ny;
        auto samp = [&](double ysamp, int k) {
            int j = int((ysamp - cfg.y0) / dyc - 0.5 + 0.5);
            j = std::max(0, std::min(cfg.ny - 1, j));
            return w[2][j + 2][k];   // x = 第一列（扰动为小量）
        };
        const double off = std::min(5.0, 0.4 * q);
        struct { const char* name; double y_in, y_out; } sh[2] = {
            { "upper", +q - off, +q + off }, { "lower", -q + off, -q - off } };
        double v[2][4];
        for (int s = 0; s < 2; ++s) {
            v[s][0] = samp(sh[s].y_in, 5);  v[s][1] = samp(sh[s].y_in, 0);
            v[s][2] = samp(sh[s].y_out, 5); v[s][3] = samp(sh[s].y_out, 0);
            std::cout << "[campaign] IC " << sh[s].name
                      << " sheet: inner(B02-side) B=" << v[s][0] << " rho=" << v[s][1]
                      << " | outer(B01-side) B=" << v[s][2] << " rho=" << v[s][3] << "\n";
        }
        std::cout << "[campaign] IC mirror diff: inner dB=" << std::fabs(v[0][0] - v[1][0])
                  << " drho=" << std::fabs(v[0][1] - v[1][1])
                  << " | outer dB=" << std::fabs(v[0][2] - v[1][2])
                  << " drho=" << std::fabs(v[0][3] - v[1][3]) << "\n";
        const double Bmx = std::max(cfg.dh_B01, cfg.dh_B02);
        const double Pt = 0.5 * Bmx * Bmx * (1.0 + 4.0);
        const double P1 = Pt - 0.5 * cfg.dh_B01 * cfg.dh_B01;
        const double P2 = Pt - 0.5 * cfg.dh_B02 * cfg.dh_B02;
        std::cout << "[campaign] tier params (check vs CS2008 Table 1): B01=" << cfg.dh_B01
                  << " B02=" << cfg.dh_B02 << " rho01=" << cfg.dh_rho01
                  << " rho02=" << cfg.dh_rho02 << " | P_total=" << Pt
                  << " P1=" << P1 << " P2=" << P2
                  << " T1=" << P1 / cfg.dh_rho01 << " T2=" << P2 / cfg.dh_rho02
                  << " beta01=" << P1 / (0.5 * cfg.dh_B01 * cfg.dh_B01)
                  << " beta02=" << P2 / (0.5 * cfg.dh_B02 * cfg.dh_B02) << "\n";

        // kick 生效自检（2026-07-13 修复：扰动幅值 0.01·B0 不乘 Bmax）。
        // 直接采样编译进本二进制的 az_coh/az_total —— 若二进制仍是误乘
        // Bmax 的旧码，此处 psi0_coh 与 dBrand 会按 Bmax 放大（AB2 为 3×）。
        {
            const auto pp = adh_params_from_config(cfg);
            const double Lxk = cfg.x1 - cfg.x0, Lyk = cfg.y1 - cfg.y0;
            const double qk = 0.25 * Lyk;
            const double psi0_coh = pp.scale
                * (pp.az_coh(0.25 * Lxk, qk) - pp.az_coh(-0.25 * Lxk, qk));
            const double psi0_expect = pp.scale * 0.02 * Lyk
                / (2.0 * 3.14159265358979323846);
            double db_half = 0.0;   // max amp_br·|hash| ≈ amp_br/2（沿片行采样）
            for (int i = 0; i < cfg.nx; ++i) {
                const double xs = cfg.x0 + (i + 0.5) * pp.hgrid;
                const double a = pp.az_total(xs, qk) - pp.az_eq(qk)
                               - pp.scale * pp.az_coh(xs, qk);
                db_half = std::max(db_half, std::fabs(a) / (pp.scale * pp.hgrid));
            }
            std::cout << std::scientific
                      << "[campaign] IC kick check: psi0_coh=" << psi0_coh
                      << " (expect " << psi0_expect << " = 0.01*B0*Ly/pi, B0=1)"
                      << "  dBrand_amp=" << 2.0 * db_half
                      << " (expect ~" << pp.amp_br << ")"
                      << std::defaultfloat << "\n";
        }
    }
    campaign_l1_tick(cs, w, cfg, divb, dx, dy, 0.0, 0.0, 0);
    cs.next_l1 = cfg.l1_dt;
    write_l2_frame(w, cfg, 0.0, "t0", 1, cs);
    return false;
}
} // namespace

OutputData run_simulation(const RunConfig& cfg) {
    const double dx = (cfg.x1 - cfg.x0) / cfg.nx;
    const double dy = (cfg.y1 - cfg.y0) / cfg.ny;
    const double Lx = cfg.x1 - cfg.x0;
    const double Ly = cfg.y1 - cfg.y0;

    Grid w(cfg.nx + 4, std::vector<Vec>(cfg.ny + 4, Vec(NVAR, 0.0)));
    initialize_problem(w, cfg);
    // Fill ghost cells before CT initializes face B from cell averages.
    apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy, &cfg);
    auto divb = make_divergence_controller(cfg.divb);
    divb->set_adiabatic_index(cfg.gamma);
    divb->set_boundary_conditions(cfg.bcx, cfg.bcy);
    divb->set_resistivity(cfg.eta);
    divb->set_hall(cfg.hall_di);
    divb->set_hyper_resistivity(cfg.eta_H);
    divb->set_subcycle_options(cfg.subcycle_nonideal, cfg.n_subcycle_max);
    divb->set_cfl(cfg.cfl);
    divb->set_hall_stab(cfg.hall_stab);
    divb->set_driven_ez(cfg.driven_ez);
    divb->initialize(w, cfg, dx, dy);

    double t = 0.0;
    int step = 0;
    double t_sweepx = 0.0, t_sweepy = 0.0, t_other = 0.0;
    auto T0 = Clock::now();

    if (cfg.rho_floor > 0.0)
        std::cout << "  density floor = " << cfg.rho_floor << ",  p_floor = " << cfg.p_floor << "\n";

    // 确保输出目录存在（label 非空时为 output/{label}，否则为 output）
    // 必须在写 snap000 之前创建，否则首帧快照会因目录不存在而静默失败
    std::filesystem::create_directories(cfg_outdir(cfg));

    // test 31 campaign 三层 I/O 初始化（含 RESUME=1 续跑）
    CampaignState cs;
    bool campaign_resumed = false;
    if (cfg.campaign_io) {
        if (!divb->uses_face_centered_b())
            throw std::runtime_error("campaign_io requires CT divergence control");
        campaign_resumed = campaign_init(cs, w, *divb, cfg, dx, dy, t, step);
        if (campaign_resumed) apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy, &cfg);
    }

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
    const std::string divb_log_path = cfg_outdir(cfg) + "/test" + std::to_string(cfg.test) + "_"
        + std::to_string(cfg.nx) + "x" + std::to_string(cfg.ny)
        + solver_suffix(cfg.solver) + divb_suffix(cfg.divb) + "_divBl2.log";
    std::ofstream divb_log(divb_log_path,
                           campaign_resumed ? (std::ios::out | std::ios::app)
                                            : std::ios::out);
    if (!campaign_resumed) {
        divb_log << "# t  l2_divB\n";
        Diagnostics d0 = compute_diagnostics(w, cfg.nx, cfg.ny, dx, dy, divb->face_field());
        divb_log << 0.0 << ' ' << d0.l2_divB << '\n';
    }

    while (t < cfg.t_end) {
        auto ta = Clock::now();
        double tw = omp_get_wtime();
        apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy, &cfg);
        perf::t_bc += omp_get_wtime() - tw; tw = omp_get_wtime();
        double dt = compute_dt(w, cfg.nx, cfg.ny, dx, dy, cfg, *divb);
        perf::t_dt += omp_get_wtime() - tw;
        if (t + dt > cfg.t_end) dt = cfg.t_end - t;
        if (dt <= 0) break;

        tw = omp_get_wtime();
        apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy, &cfg);
        perf::t_bc += omp_get_wtime() - tw;
        divb->pre_step(w, cfg.nx, cfg.ny, dt, dx, dy);
        auto tb = Clock::now();

        // Alternating Strang splitting: even steps XYX, odd steps YXY.
        auto timed_bc = [&] {
            double s = omp_get_wtime();
            apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy, &cfg);
            perf::t_bc += omp_get_wtime() - s;
        };
        auto timed_floor = [&] {
            double s = omp_get_wtime();
            apply_floor(w, cfg.nx, cfg.ny, cfg);
            perf::t_floor += omp_get_wtime() - s;
        };
        Clock::time_point tc, td, te;
        if (step % 2 == 0) {
            sweep_x(w, cfg.nx, cfg.ny, 0.5 * dt, dx, cfg, *divb);
            timed_floor();
            tc = Clock::now();
            timed_bc();
            sweep_y(w, cfg.nx, cfg.ny, dt, dy, cfg, *divb);
            timed_floor();
            td = Clock::now();
            timed_bc();
            sweep_x(w, cfg.nx, cfg.ny, 0.5 * dt, dx, cfg, *divb);
            timed_floor();
            te = Clock::now();
            t_sweepx += elapsed(tb, tc) + elapsed(td, te);
            t_sweepy += elapsed(tc, td);
        } else {
            sweep_y(w, cfg.nx, cfg.ny, 0.5 * dt, dy, cfg, *divb);
            timed_floor();
            tc = Clock::now();
            timed_bc();
            sweep_x(w, cfg.nx, cfg.ny, dt, dx, cfg, *divb);
            timed_floor();
            td = Clock::now();
            timed_bc();
            sweep_y(w, cfg.nx, cfg.ny, 0.5 * dt, dy, cfg, *divb);
            timed_floor();
            te = Clock::now();
            t_sweepy += elapsed(tb, tc) + elapsed(td, te);
            t_sweepx += elapsed(tc, td);
        }

        divb->set_current_time(t);
        divb->post_step(w, cfg.nx, cfg.ny, dt, Lx, Ly, dx, dy);
        auto tf = Clock::now();

        t_other += elapsed(ta, tb) + elapsed(te, tf);
        t += dt;
        ++step;

        if (do_snaps && t >= next_snap_t - 1e-12 * cfg.output_dt) {
            apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy, &cfg);
            auto dB_snap = compute_divB(w, cfg.nx, cfg.ny, dx, dy, divb->face_field());
            write_snapshot_file(w, dB_snap, cfg, snap_idx++, t);
            next_snap_t += cfg.output_dt;
        }

        // campaign 三层 I/O：L1 行（每 l1_dt）→ L2 触发器 → L3 checkpoint
        if (cfg.campaign_io) {
            const double tdg = omp_get_wtime();
            bool ok = true;
            if (t >= cs.next_l1 - 1e-9) {
                ok = campaign_l1_tick(cs, w, cfg, *divb, dx, dy, t, dt, step);
                while (cs.next_l1 <= t + 1e-9) cs.next_l1 += cfg.l1_dt;
            }
            if (cfg.ckpt_dt > 0.0 && t >= cs.next_ckpt - 1e-9) {
                write_ckpt(cs, w, divb->face_field(), cfg, t, step);
                while (cs.next_ckpt <= t + 1e-9) cs.next_ckpt += cfg.ckpt_dt;
            }
            perf::t_diag += omp_get_wtime() - tdg;
            if (!ok) {
                cs.l1.flush();
                std::cerr << "*** FATAL: campaign L1 detected unphysical state at t="
                          << t << " (checkpoint retained for postmortem)\n";
                break;
            }
        }

        if (step % 200 == 0) {
            const double tdg2 = omp_get_wtime();
            apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy, &cfg);
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
                      << "  max|v|=" << diag.max_v;
            if (cfg.subcycle_nonideal) {
                int nsub = divb->last_n_sub();
                std::cout << "  N_sub=" << nsub;
            }
            std::cout << '\n';
            divb_log << t << ' ' << diag.l2_divB << '\n';
            perf::t_diag += omp_get_wtime() - tdg2;
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

    apply_bc(w, cfg.nx, cfg.ny, cfg.bcx, cfg.bcy, &cfg);

    if (cfg.campaign_io) {
        write_l2_frame(w, cfg, t, "final", 1, cs);
        write_ckpt(cs, w, divb->face_field(), cfg, t, step);
        cs.l1.flush();
        std::cout << "[campaign] floor activations: rho=" << g_floor_rho_count
                  << " p=" << g_floor_p_count
                  << "  first-order fallback cells: " << g_fallback_count.load()
                  << "  L2 frames: " << cs.n_l2 << "/15\n";
    }

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
    ss << cfg_outdir(cfg) << "/test" << cfg.test << "_"
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
    const std::string filename = cfg_outdir(cfg) + "/test" + std::to_string(cfg.test) + "_"
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
