#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <string>
#include <vector>

namespace my_project {

inline constexpr int NVAR = 9;

// Fixed-size stack state vector (was std::vector<double>): removes heap
// allocation from hot paths. Constructors keep the old vector signatures.
struct Vec : std::array<double, NVAR> {
    Vec() : std::array<double, NVAR>{} {}          // zero-initialised
    explicit Vec(int) : Vec() {}
    Vec(int, double v) { fill(v); }
    Vec(std::initializer_list<double> il) : Vec() {
        std::copy(il.begin(), il.end(), begin());
    }
};
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

// Hall short-wave stabilization (CT mode only).
// NONE: RK4 Hall CFL alone; HYPER_RES: −η_H·∇²Jz (mainline, test 20/21);
// HALL_HLL: whistler-speed upwind diffusion (Path B, test 23, experimental).
enum class HallStabKind { NONE = 0, HYPER_RES = 1, HALL_HLL = 2 };

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
    // Uniform resistivity η (0 = ideal MHD). CT mode only.
    double eta = 0.0;

    // Ion inertial length d_i for the Hall term (0 = off). CT mode only;
    // the Hall CFL is enforced by compute_dt.
    double hall_di = 0.0;

    // 4th-order hyper-resistivity η_H (0 = off). CT mode only;
    // CFL dt ≤ h⁴/(32 η_H) is enforced by compute_dt.
    double eta_H = 0.0;

    // Snapshot interval (0 = final state only; >0 writes snapNNN.dat from t=0).
    double output_dt = 0.0;

    // Density/pressure floors applied after each sweep (0 = disabled).
    double rho_floor = 0.0;
    double p_floor   = 0.0;

    // Non-ideal sub-cycling: removes Hall/resistive CFL from the global dt,
    // sub-cycling the non-ideal block N_sub times per step instead.
    bool subcycle_nonideal = false;
    int  n_subcycle_max    = 100;   // safety cap; warning printed if hit

    // Hall stabilization scheme (see HallStabKind).
    HallStabKind hall_stab = HallStabKind::NONE;

    // Output label: non-empty writes to output/{label}/ (CLI arg 8).
    std::string label = "";

    // test 25 perturbation amplitude ψ0 (CLI arg 9)
    double psi0 = 0.2;

    // test 27 asymmetry scan: B2 and rho1 configurable, B1 = rho2 = 1 fixed
    // (CLI 11/12; defaults = R2 configuration).
    double asym_B2   = 2.0;
    double asym_rho1 = 2.0;

    // test 28 Dirichlet inflow scalars: fix y-ghost rho/p to upstream values
    // so long driven runs keep a mass reservoir (v/B unchanged).
    bool   dirichlet_y_scalars = false;
    double bc_rho_top = 0.0, bc_p_top = 0.0;   // y=y1 side (weak field B1)
    double bc_rho_bot = 0.0, bc_p_bot = 0.0;   // y=y0 side (strong field B2)

    // test 26 driven reconnection: constant boundary Ez (0 = off), applied to
    // both y-boundary corner-EMF rows. CT + transmissive bcy only.
    // Must be negative here — reconnection Ez < 0, so only a negative drive
    // gives inflow on both sides.
    double driven_ez = 0.0;

    // test 31 campaign three-layer I/O (L1 schema frozen in
    // output/test29_campaign/l1_freeze_report.md): L1 CSV rows every l1_dt,
    // L2 event-triggered float32 snapshots (cap 15), L3 rolling checkpoint
    // every ckpt_dt (RESUME=1 continues).
    bool   campaign_io = false;
    double l1_dt   = 0.5;
    double ckpt_dt = 10.0;

    // test 31 double sheet (CS2008 Eq. 3-4): 01 = outer, 02 = inner.
    // CLI 13-16; all 1 = symmetric tier.
    double dh_B01 = 1.0, dh_B02 = 1.0, dh_rho01 = 1.0, dh_rho02 = 1.0;
};

struct Diagnostics {
    double min_rho = 0.0;
    double min_p = 0.0;
    double max_divB = 0.0;
    double l2_divB = 0.0;
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
