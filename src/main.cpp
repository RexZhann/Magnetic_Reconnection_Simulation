#include "my_project/solver.hpp"
#include "my_project/perf.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>

using namespace my_project;

namespace {
std::string divb_name(DivBCleaningKind d) {
    switch (d) {
        case DivBCleaningKind::GLM: return "GLM";
        case DivBCleaningKind::CT: return "CT";
        default: return "NONE";
    }
}
}

int main(int argc, char* argv[]) {
    try {
        int test = 0, nx = 200, ny = 200;
        DivBCleaningKind divb = DivBCleaningKind::GLM;
        SolverKind solver = SolverKind::HLLD;

        if (argc > 1) test = std::atoi(argv[1]);
        if (argc > 2) nx = std::atoi(argv[2]);
        if (argc > 3) ny = std::atoi(argv[3]);
        if (argc > 4) {
            int v = std::atoi(argv[4]);
            if (v == 2) divb = DivBCleaningKind::CT;
            else if (v == 1) divb = DivBCleaningKind::GLM;
            else divb = DivBCleaningKind::None;
        }
        if (argc > 5) solver = (std::atoi(argv[5]) == 0) ? SolverKind::FORCE : SolverKind::HLLD;

        RunConfig cfg = make_config_for_test(test, nx, ny, divb, solver);

        // CLI 参数 6–8：覆盖 η_H、t_end 和输出标签（均可选）
        if (argc > 6) { double v = std::atof(argv[6]); if (v >= 0.0) cfg.eta_H = v; }
        if (argc > 7) { double v = std::atof(argv[7]); if (v >  0.0) cfg.t_end = v; }
        if (argc > 8) cfg.label = std::string(argv[8]);
        if (argc > 9) { double v = std::atof(argv[9]); if (v > 0.0) cfg.psi0 = v; }
        // CLI 参数 10：覆盖 driven_ez（test 26 驱动强度；注意本问题符号应为负）
        if (argc > 10) cfg.driven_ez = std::atof(argv[10]);
        // CLI 参数 11/12：test 27 非对称度扫描的 B2 与 ρ1（B1=1、ρ2=1 固定）
        if (argc > 11) { double v = std::atof(argv[11]); if (v > 0.0) cfg.asym_B2   = v; }
        if (argc > 12) { double v = std::atof(argv[12]); if (v > 0.0) cfg.asym_rho1 = v; }
        // CLI 参数 13-16：test 31 广义双片 B01/B02/ρ01/ρ02（CS2008 Table 1 九档）
        if (argc > 13) { double v = std::atof(argv[13]); if (v > 0.0) cfg.dh_B01   = v; }
        if (argc > 14) { double v = std::atof(argv[14]); if (v > 0.0) cfg.dh_B02   = v; }
        if (argc > 15) { double v = std::atof(argv[15]); if (v > 0.0) cfg.dh_rho01 = v; }
        if (argc > 16) { double v = std::atof(argv[16]); if (v > 0.0) cfg.dh_rho02 = v; }

        // IC_CHECK=1：t=0 自检模式——初始化+打印 IC 检查（含 kick 幅值）后
        // 不进主循环即收尾。用于服务器端验证"实际运行的二进制"是新码。
        if (const char* icc = std::getenv("IC_CHECK"); icc && icc[0] == '1') {
            cfg.t_end = 0.0;
            std::cout << "[IC_CHECK] t=0 self-check mode: no time stepping\n";
        }

        std::cout << "Test " << cfg.test << ", " << cfg.nx << "x" << cfg.ny
                  << ", " << (cfg.solver == SolverKind::FORCE ? "FORCE" : "HLLD")
                  << ", divB=" << divb_name(cfg.divb)
                  << ", Threads=" << omp_get_max_threads() << '\n';

        OutputData out = run_simulation(cfg);
        std::cout << "Finished: " << out.timing.steps << " steps, t=" << out.timing.t_final << "\n\n";
        std::cout << "=== Timing Summary ===\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Total wall time : " << out.timing.total << " s\n";
        std::cout << "  sweep_x (total) : " << out.timing.sweep_x << " s  ("
                  << 100.0 * out.timing.sweep_x / std::max(out.timing.total, 1e-30) << "%)\n";
        std::cout << "  sweep_y (total) : " << out.timing.sweep_y << " s  ("
                  << 100.0 * out.timing.sweep_y / std::max(out.timing.total, 1e-30) << "%)\n";
        std::cout << "  other   (bc/dt) : " << out.timing.other << " s  ("
                  << 100.0 * out.timing.other / std::max(out.timing.total, 1e-30) << "%)\n";
        if (out.timing.steps > 0) {
            std::cout << "  per step avg    : " << 1000.0 * out.timing.total / out.timing.steps << " ms/step\n";
            std::cout << "  Mcell-steps/s   : " << 1e-6 * static_cast<long long>(out.timing.steps) * cfg.nx * cfg.ny / std::max(out.timing.total, 1e-30) << "\n";
        }
        std::cout << "======================\n";

        {
            // 模块占比表：sweep 内三段是线程时间和，按 sweep 墙钟等比折算。
            namespace pf = my_project::perf;
            const double total = std::max(out.timing.total, 1e-30);
            const double sweep_wall = out.timing.sweep_x + out.timing.sweep_y;
            const double thr_sum = pf::t_recon + pf::t_riemann + pf::t_conupd;
            const double f = (thr_sum > 0.0) ? sweep_wall / thr_sum : 0.0;
            const double w_recon = pf::t_recon * f, w_riem = pf::t_riemann * f,
                         w_conupd = pf::t_conupd * f;
            const double known = w_recon + w_riem + w_conupd + pf::t_ct_assemble
                + pf::t_ct_faraday + pf::t_hall + pf::t_hyper + pf::t_subcfl
                + pf::t_bc + pf::t_dt + pf::t_floor + pf::t_diag;
            auto row = [&](const char* name, double v) {
                std::cout << "  " << std::left << std::setw(26) << name
                          << std::right << std::setw(9) << v << " s  ("
                          << std::setw(5) << 100.0 * v / total << "%)\n";
            };
            std::cout << "=== Perf profile (wall-clock, scaled) ===\n";
            row("MUSCL-Hancock recon", w_recon);
            row("Riemann (HLLD) flux", w_riem);
            row("cons update+pri<->con", w_conupd);
            row("CT corner-EMF assemble", pf::t_ct_assemble);
            row("CT Faraday+sync+zero", pf::t_ct_faraday);
            row("Hall RK4", pf::t_hall);
            row("hyper-resistivity", pf::t_hyper);
            row("subcycle CFL scan", pf::t_subcfl);
            row("boundary conditions", pf::t_bc);
            row("compute_dt", pf::t_dt);
            row("apply_floor", pf::t_floor);
            row("diagnostics/IO", pf::t_diag);
            row("rest (unaccounted)", total - known);
            if (pf::nsub_calls > 0) {
                std::cout << "  N_sub: mean=" << double(pf::nsub_sum) / double(pf::nsub_calls)
                          << " max=" << pf::nsub_max
                          << " capped_steps=" << pf::nsub_capped
                          << "/" << pf::nsub_calls << "\n";
            }
            std::cout << "  OMP threads: " << omp_get_max_threads() << "\n";
            std::cout << "=========================================\n";
        }

        double maxDB = 0.0;
        for (const auto& row : out.divB) for (double v : row) maxDB = std::max(maxDB, v);
        std::cout << "Max |div B| = " << maxDB << '\n';
        if (out.has_face_field) {
            std::cout << "CT face-centered magnetic field stored in memory for post-processing.\n";
        }
        write_output_file(out, cfg);
        std::cout << "Output file written." << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << '\n';
        return 1;
    }
}
