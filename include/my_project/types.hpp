#pragma once

#include <vector>

namespace my_project {

inline constexpr int NVAR = 9;
using Vec  = std::vector<double>;
using Row  = std::vector<Vec>;
using Grid = std::vector<std::vector<Vec>>;
using ScalarField = std::vector<std::vector<double>>;

struct FaceField2D {
    // Bx stored on x-faces: size (nx+1) x ny
    ScalarField bx;
    // By stored on y-faces: size nx x (ny+1)
    ScalarField by;
    // Ez stored on cell corners / edges normal to z: size (nx+1) x (ny+1)
    ScalarField emf_z;

    void resize(int nx, int ny) {
        bx.assign(nx + 1, std::vector<double>(ny, 0.0));
        by.assign(nx, std::vector<double>(ny + 1, 0.0));
        emf_z.assign(nx + 1, std::vector<double>(ny + 1, 0.0));
    }
    bool empty() const { return bx.empty() || by.empty(); }
};

enum class BC { Transmissive, Periodic };
enum class SolverKind { FORCE = 0, HLLD = 1 };
enum class DivBCleaningKind { None = 0, GLM = 1, CT = 2 };

struct RunConfig {
    int test = 0;
    int nx = 200;
    int ny = 200;
    double gamma = 1.4;
    double cfl = 0.3;
    double x0 = 0.0, x1 = 1.0;
    double y0 = 0.0, y1 = 1.0;
    double t_end = 0.25;
    BC bcx = BC::Transmissive;
    BC bcy = BC::Transmissive;
    SolverKind solver = SolverKind::HLLD;
    DivBCleaningKind divb = DivBCleaningKind::GLM;
    // Uniform resistivity η for resistive MHD (0 = ideal MHD).
    // Only active for CT divergence control; explicit stability requires
    // η ≤ min(dx,dy)² / (2 dt), which is automatically satisfied when
    // the ideal-MHD CFL condition is the tighter constraint.
    double eta = 0.0;
};

struct Diagnostics {
    double min_rho = 0.0;
    double min_p = 0.0;
    double max_divB = 0.0;
    double max_psi = 0.0;
    double max_v = 0.0;
};

struct TimingStats {
    double total = 0.0;
    double sweep_x = 0.0;
    double sweep_y = 0.0;
    double other = 0.0;
    int steps = 0;
    double t_final = 0.0;
};

struct OutputData {
    Grid primitive;
    ScalarField divB;
    TimingStats timing;
    bool has_face_field = false;
    FaceField2D face_field;
};

} // namespace my_project
