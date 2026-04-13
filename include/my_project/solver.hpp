#pragma once

#include <memory>
#include "my_project/divergence_control.hpp"
#include "my_project/types.hpp"

namespace my_project {

struct ScratchBuf {
    int cap = 0;
    Row uc, d0, d1, delta, xL, xR, hL, hR, iflx, s_row;
    void ensure(int N);
};

RunConfig make_config_for_test(int test, int nx, int ny,
                               DivBCleaningKind divb,
                               SolverKind solver);
void apply_bc(Grid& w, int nx, int ny, BC bcx, BC bcy);
void sweep_x(Grid& w, int nx, int ny, double dt, double dx, const RunConfig& cfg,
             DivergenceController& divb);
void sweep_y(Grid& w, int nx, int ny, double dt, double dy, const RunConfig& cfg,
             DivergenceController& divb);
double compute_dt(const Grid& w, int nx, int ny, double dx, double dy,
                  const RunConfig& cfg, DivergenceController& divb);
std::vector<std::vector<double>> compute_divB(const Grid& w, int nx, int ny,
                                              double dx, double dy);
Diagnostics compute_diagnostics(const Grid& w, int nx, int ny,
                                double dx, double dy);
void initialize_problem(Grid& w, const RunConfig& cfg);
OutputData run_simulation(const RunConfig& cfg);
void write_output_file(const OutputData& out, const RunConfig& cfg);
std::unique_ptr<DivergenceController> make_divergence_controller(DivBCleaningKind kind);

} // namespace my_project
