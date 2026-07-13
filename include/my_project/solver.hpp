#pragma once

#include <memory>
#include "my_project/divergence_control.hpp"
#include "my_project/types.hpp"

namespace my_project {

struct ScratchBuf {
    int cap = 0;
    Row uc, d0, d1, delta, xL, xR, hL, hR, iflx, s_row;
    // CT work buffers: face-centered normal B, interface EMF, centered EMF,
    // and mass-flux sign. Sized to N = n+4 (matching the 1D slice length).
    std::vector<double> face_bn_buf;
    std::vector<double> emf_buf;
    std::vector<double> cemf_buf;   // centered (dissipation-free) EMF
    std::vector<double> sgn_buf;    // sign of mass flux (+1 or -1)
    void ensure(int N);
};

RunConfig make_config_for_test(int test, int nx, int ny,
                               DivBCleaningKind divb,
                               SolverKind solver);
// cfg 非空且 cfg->dirichlet_y_scalars 时，y 方向 ghost 的 ρ/p 在零梯度填充后
// 被覆写为固定上游值（test 28 Dirichlet inflow 标量边界）；其余行为不变。
void apply_bc(Grid& w, int nx, int ny, BC bcx, BC bcy,
              const RunConfig* cfg = nullptr);
void sweep_x(Grid& w, int nx, int ny, double dt, double dx, const RunConfig& cfg,
             DivergenceController& divb);
void sweep_y(Grid& w, int nx, int ny, double dt, double dy, const RunConfig& cfg,
             DivergenceController& divb);
double compute_dt(const Grid& w, int nx, int ny, double dx, double dy,
                  const RunConfig& cfg, DivergenceController& divb);
// When face_field is provided (CT mode) the exact face-difference divergence is used.
std::vector<std::vector<double>> compute_divB(const Grid& w, int nx, int ny,
                                              double dx, double dy,
                                              const FaceField2D* face_field = nullptr);
Diagnostics compute_diagnostics(const Grid& w, int nx, int ny,
                                double dx, double dy,
                                const FaceField2D* face_field = nullptr);
void initialize_problem(Grid& w, const RunConfig& cfg);
OutputData run_simulation(const RunConfig& cfg);
void write_output_file(const OutputData& out, const RunConfig& cfg);
// Write a single time snapshot (used internally by run_simulation when
// cfg.output_dt > 0).  The file header contains an extra 't' field so that
// the Python loader can recover the simulation time from each file.
void write_snapshot_file(const Grid& w,
                         const std::vector<std::vector<double>>& divB,
                         const RunConfig& cfg, int snap_idx, double t);
std::unique_ptr<DivergenceController> make_divergence_controller(DivBCleaningKind kind);

} // namespace my_project
