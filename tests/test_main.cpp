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
    {
        Vec w{1.0, 0.1, -0.2, 0.3, 1.0, 0.5, -0.4, 0.2, 0.0};
        Vec u = pri2con(w, 1.4);
        Vec w2 = con2pri(u, 1.4);
        require(std::fabs(w2[0] - w[0]) < 1e-12, "rho roundtrip failed");
        require(std::fabs(w2[4] - w[4]) < 1e-12, "pressure roundtrip failed");
    }
    {
        require(minbee(-1.0) == 0.0, "minbee negative branch failed");
        require(std::fabs(minbee(0.5) - 0.5) < 1e-14, "minbee linear branch failed");
        require(std::fabs(minbee(2.0) - 1.0) < 1e-14, "minbee saturated branch failed");
    }
    {
        RunConfig cfg = make_config_for_test(0, 20, 20, DivBCleaningKind::GLM, SolverKind::HLLD);
        OutputData out = run_simulation(cfg);
        require(out.timing.steps > 0, "GLM simulation did not advance");
        double max_db = 0.0;
        for (const auto& row : out.divB) for (double v : row) max_db = std::max(max_db, v);
        require(std::isfinite(max_db), "GLM divB is not finite");
    }
    {
        RunConfig cfg = make_config_for_test(0, 16, 16, DivBCleaningKind::CT, SolverKind::HLLD);
        cfg.t_end = 0.01;
        OutputData out = run_simulation(cfg);
        require(out.timing.steps > 0, "CT scaffold simulation did not advance");
        require(out.has_face_field, "CT scaffold did not expose face-centered field");
        require(static_cast<int>(out.face_field.bx.size()) == cfg.nx + 1, "CT bx face size mismatch");
        require(static_cast<int>(out.face_field.by.size()) == cfg.nx, "CT by face size mismatch");
    }
    std::cout << "All tests passed.\n";
    return 0;
}
