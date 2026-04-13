#include "my_project/solver.hpp"
#include "my_project/state.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace my_project;

namespace {
void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}
}

int main() {
    // --- primitive <-> conservative roundtrip ---
    {
        Vec w{1.0, 0.1, -0.2, 0.3, 1.0, 0.5, -0.4, 0.2, 0.0};
        Vec u = pri2con(w, 1.4);
        Vec w2 = con2pri(u, 1.4);
        require(std::fabs(w2[0] - w[0]) < 1e-12, "rho roundtrip failed");
        require(std::fabs(w2[4] - w[4]) < 1e-12, "pressure roundtrip failed");
    }

    // --- minbee limiter branches ---
    {
        require(minbee(-1.0) == 0.0,                   "minbee negative branch failed");
        require(std::fabs(minbee(0.5) - 0.5) < 1e-14, "minbee linear branch failed");
        require(std::fabs(minbee(2.0) - 1.0) < 1e-14, "minbee saturated branch failed");
    }

    // --- GLM simulation sanity check ---
    {
        RunConfig cfg = make_config_for_test(0, 20, 20, DivBCleaningKind::GLM, SolverKind::HLLD);
        OutputData out = run_simulation(cfg);
        require(out.timing.steps > 0, "GLM simulation did not advance");
        double max_db = 0.0;
        for (const auto& row : out.divB) for (double v : row) max_db = std::max(max_db, v);
        require(std::isfinite(max_db), "GLM divB is not finite");
    }

    // --- CT initialization should start from a discrete divergence-free face field ---
    {
        RunConfig cfg = make_config_for_test(3, 32, 32, DivBCleaningKind::CT, SolverKind::HLLD);
        cfg.t_end = 0.0;
        OutputData out = run_simulation(cfg);

        require(out.has_face_field, "CT init did not expose face-centered field");
        require(out.timing.steps == 0, "CT init-only run unexpectedly advanced");

        double max_db = 0.0;
        for (const auto& row : out.divB) for (double v : row) max_db = std::max(max_db, v);
        require(std::isfinite(max_db), "CT init divB is not finite");
        require(max_db < 1e-12, "CT init face field is not discretely divergence-free");
    }

    // --- CT simulation: face fields, physics, and divergence-free constraint ---
    {
        RunConfig cfg = make_config_for_test(0, 32, 32, DivBCleaningKind::CT, SolverKind::HLLD);
        cfg.t_end = 0.05;
        OutputData out = run_simulation(cfg);

        require(out.timing.steps > 0, "CT simulation did not advance");
        require(out.has_face_field,   "CT did not expose face-centered field");
        require(static_cast<int>(out.face_field.bx.size()) == cfg.nx + 1,
                "CT bx face x-size mismatch");
        require(static_cast<int>(out.face_field.by.size()) == cfg.nx,
                "CT by face x-size mismatch");
        require(static_cast<int>(out.face_field.by[0].size()) == cfg.ny + 1,
                "CT by face y-size mismatch");

        // divB is computed from the face-centred fields (exact Faraday stencil).
        // The Faraday update preserves the face-difference divergence to machine
        // precision, so max|divB| must be near floating-point round-off.
        double max_db = 0.0;
        for (const auto& row : out.divB) for (double v : row) max_db = std::max(max_db, v);
        require(std::isfinite(max_db), "CT divB is not finite");
        require(max_db < 1e-10,        "CT divB exceeds machine-precision threshold");

        // Physical sanity: density must remain positive throughout.
        double min_rho = 1e30;
        for (int i = 2; i < cfg.nx + 2; ++i)
            for (int j = 2; j < cfg.ny + 2; ++j)
                min_rho = std::min(min_rho, out.primitive[i][j][0]);
        require(min_rho > 0.0, "CT simulation produced non-positive density");
    }

    std::cout << "All tests passed.\n";
    return 0;
}
