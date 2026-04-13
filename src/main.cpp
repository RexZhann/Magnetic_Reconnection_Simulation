#include "my_project/solver.hpp"

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
