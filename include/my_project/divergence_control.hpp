#pragma once

#include "my_project/types.hpp"

namespace my_project {

class DivergenceController {
public:
    virtual ~DivergenceController() = default;
    virtual bool glm_enabled() const = 0;
    virtual bool uses_face_centered_b() const { return false; }
    virtual void update_characteristic_speed(double value) = 0;
    virtual double characteristic_speed() const = 0;
    virtual void initialize(Grid& w, const RunConfig& cfg, double dx, double dy) = 0;
    virtual void pre_step(Grid&, int, int, double, double, double) {}
    virtual void post_step(Grid& w, int nx, int ny, double dt, double Lx, double Ly,
                           double dx, double dy) = 0;
    virtual const FaceField2D* face_field() const { return nullptr; }
    // 非 const 访问：L3 checkpoint 续跑时把存档的面心 B 写回（仅 CT 有效）。
    virtual FaceField2D* mutable_face_field() { return nullptr; }
    virtual void set_adiabatic_index(double /*gamma*/) {}

    // Set the boundary condition types so that the controller can apply the
    // correct face-B BC and periodic EMF wrapping.
    virtual void set_boundary_conditions(BC /*bcx*/, BC /*bcy*/) {}

    // Set uniform resistivity η for resistive MHD.  Default 0 (ideal MHD).
    // Only the CT controller uses this; all others silently ignore it.
    virtual void set_resistivity(double /*eta*/) {}

    // Set ion inertial length d_i for Hall MHD.  Default 0 (no Hall term).
    // Only the CT controller uses this; all others silently ignore it.
    virtual void set_hall(double /*di*/) {}

    // Set 4th-order hyper-resistivity coefficient η_H.  Default 0 (off).
    // Only the CT controller uses this; all others silently ignore it.
    virtual void set_hyper_resistivity(double /*eta_H*/) {}

    // Enable non-ideal sub-cycling with a safety cap on N_sub.
    // Only the CT controller uses this; all others silently ignore it.
    virtual void set_subcycle_options(bool /*enable*/, int /*nmax*/) {}

    // Inform the controller of the current simulation time (used for diagnostic logging).
    virtual void set_current_time(double /*t*/) {}

    // Pass the CFL number so the sub-cycler can apply the same safety margin.
    virtual void set_cfl(double /*cfl*/) {}

    // 设置霍尔短波稳定化方案。仅 CT 控制器使用，其他静默忽略。
    virtual void set_hall_stab(HallStabKind /*hs*/) {}

    // driven reconnection：y 边界指定恒定 E_z（0 = 关闭）。仅 CT 控制器使用。
    virtual void set_driven_ez(double /*ez*/) {}

    // Query the sub-cycle count used in the most recent post_step call.
    // Returns 1 when sub-cycling was not active.
    virtual int  last_n_sub() const { return 1; }

    // CT interface: fill a contiguous buffer with the face-centered normal B
    // for a 1D sweep row/column. buf must have size >= n+2.
    // For x-sweep of interior row j (0-indexed): buf[i] = Bx_face at interface i (i=1..n+1).
    // For y-sweep of interior col i (0-indexed): buf[j] = By_face at interface j (j=1..n+1).
    virtual void fill_face_bn_x(int /*j*/, int /*n*/, double* /*buf*/) const {}
    virtual void fill_face_bn_y(int /*i*/, int /*n*/, double* /*buf*/) const {}

    // CT interface: store interface EMFs produced by a 1D sweep.
    // emf[i] (i=1..n+1) is Ez at interface i of the slice.
    // cemf[i] is the centered (dissipation-free) EMF; sgn[i] is sign(mass flux).
    // x-sweep: Ez = -F[6]; y-sweep (rotated frame): Ez = F[6].
    virtual void store_emf_x(int /*j*/, int /*n*/, const double* /*emf*/,
                              const double* /*cemf*/ = nullptr,
                              const double* /*sgn*/  = nullptr) {}
    virtual void store_emf_y(int /*i*/, int /*n*/, const double* /*emf*/,
                              const double* /*cemf*/ = nullptr,
                              const double* /*sgn*/  = nullptr) {}
};

class NoDivBCleaning final : public DivergenceController {
public:
    bool glm_enabled() const override { return false; }
    void update_characteristic_speed(double) override {}
    double characteristic_speed() const override { return 0.0; }
    void initialize(Grid&, const RunConfig&, double, double) override {}
    void post_step(Grid&, int, int, double, double, double, double, double) override {}
};

class GLMDivergenceCleaning final : public DivergenceController {
public:
    bool glm_enabled() const override { return true; }
    void update_characteristic_speed(double value) override { ch_ = value; }
    double characteristic_speed() const override { return ch_; }
    void initialize(Grid&, const RunConfig&, double, double) override {}
    void post_step(Grid& w, int nx, int ny, double dt, double Lx, double Ly,
                   double dx, double dy) override;
private:
    double ch_ = 0.0;
};

// Constrained Transport divergence control.
//
// Primary B storage: face-centered bx[i][j] (x-faces, size (nx+1)×ny)
//                    and by[i][j] (y-faces, size nx×(ny+1)).
// Cell-centered B is derived from faces by averaging and is never used
// as the authoritative source after initialization.
//
// Each directional sweep contributes interface EMFs (Ez) at the faces
// normal to that sweep direction.  After all sweeps, corner EMFs are
// formed by arithmetic averaging of the four surrounding interface EMFs
// (with periodic wrapping for periodic BC), and face B is advanced via
// the discrete Faraday law.  By stencil algebra this preserves ∇·B to
// machine precision every step for all interior cells.
class CTDivergenceControl final : public DivergenceController {
public:
    bool glm_enabled() const override { return false; }
    bool uses_face_centered_b() const override { return true; }
    void update_characteristic_speed(double value) override { ch_like_ = value; }
    double characteristic_speed() const override { return ch_like_; }
    void set_adiabatic_index(double gamma) override { gamma_ = gamma; }
    void set_boundary_conditions(BC bcx, BC bcy) override { bcx_ = bcx; bcy_ = bcy; }
    void set_resistivity(double eta) override { eta_ = eta; }
    void set_hall(double di) override { hall_di_ = di; }
    void set_hyper_resistivity(double eta_H) override { eta_H_ = eta_H; }
    void set_subcycle_options(bool enable, int nmax) override {
        subcycle_nonideal_ = enable;
        n_subcycle_max_    = nmax;
    }
    void set_current_time(double t) override { current_t_ = t; }
    void set_cfl(double cfl) override { cfl_ = cfl; }
    void set_hall_stab(HallStabKind hs) override { hall_stab_ = hs; }
    void set_driven_ez(double ez) override { driven_ez_ = ez; }
    int  last_n_sub() const override { return last_n_sub_; }
    void initialize(Grid& w, const RunConfig& cfg, double dx, double dy) override;
    void pre_step(Grid& w, int nx, int ny, double dt, double dx, double dy) override;
    void post_step(Grid& w, int nx, int ny, double dt, double Lx, double Ly,
                   double dx, double dy) override;
    const FaceField2D* face_field() const override { return &face_; }
    FaceField2D* mutable_face_field() override { return &face_; }

    void fill_face_bn_x(int j, int n, double* buf) const override;
    void fill_face_bn_y(int i, int n, double* buf) const override;
    void store_emf_x(int j, int n, const double* emf,
                     const double* cemf = nullptr, const double* sgn = nullptr) override;
    void store_emf_y(int i, int n, const double* emf,
                     const double* cemf = nullptr, const double* sgn = nullptr) override;

private:
    FaceField2D face_;
    // Interface EMFs collected during sweeps:
    //   emf_x_[k][j] = Ez at x-face bx[k][j]  (k=0..nx, j=0..ny-1)
    //   emf_y_[i][k] = Ez at y-face by[i][k]   (i=0..nx-1, k=0..ny)
    ScalarField emf_x_;
    ScalarField emf_y_;
    // CT-Contact: centered (dissipation-free) EMF and sign(mass flux).
    ScalarField cemf_x_, cemf_y_;
    ScalarField sgn_x_,  sgn_y_;
    int nx_ = 0, ny_ = 0;
    double ch_like_ = 0.0;
    double gamma_   = 1.4;
    double eta_     = 0.0;
    double hall_di_ = 0.0;
    double eta_H_   = 0.0;
    BC bcx_ = BC::Transmissive;
    BC bcy_ = BC::Transmissive;
    // Sub-cycling state
    bool   subcycle_nonideal_ = false;
    int    n_subcycle_max_    = 100;
    double current_t_         = 0.0;
    double cfl_               = 0.3;  // CFL safety factor (matches compute_dt)
    int    last_n_sub_        = 1;    // set each post_step; readable via last_n_sub()
    HallStabKind hall_stab_  = HallStabKind::NONE;
    double rho_floor_        = 0.02;  // 与 cfg.rho_floor 保持一致，由 initialize 写入
    double driven_ez_        = 0.0;   // driven reconnection：y 边界恒定 E_z（0 = 关闭）

    void initialize_faces_from_problem(const Grid& w, const RunConfig& cfg, double dx, double dy);
    void fill_faces_from_cell_centered(const Grid& w, int nx, int ny);
    void compute_corner_emf_from_interface_emfs(int nx, int ny);
    // Add η·Jz to every corner EMF and η·Jz² (as Ohmic heating) to cell pressure.
    // Called after compute_corner_emf_from_interface_emfs, before update_faces_from_emf.
    // No-op when eta_ == 0 (ideal MHD).
    void add_resistive_correction(Grid& w, int nx, int ny,
                                  double dt, double dx, double dy);

    // Add −η_H·∇²Jz to every corner EMF, giving ∂B/∂t = −η_H·∇⁴B (biharmonic).
    // Called after add_resistive_correction, before update_faces_from_emf.
    // No-op when eta_H_ == 0.
    void add_hyper_resistive_correction(int nx, int ny, double dx, double dy);

    // Add Hall term (d_i/ρ) J × B to the induction equation:
    //   1. Adds E_z^Hall = (d_i/ρ)(J_x B_y - J_y B_x) to corner EMF (→ drives Bx, By).
    //   2. Updates cell-centred Bz via ∂Bz/∂t = ∂E_x^Hall/∂y - ∂E_y^Hall/∂x.
    // Must be called after add_resistive_correction, before update_faces_from_emf.
    // No-op when hall_di_ == 0.
    void add_hall_correction(Grid& w, int nx, int ny,
                             double dt, double dx, double dy);
    // driven BC：把 J=0（下）和 J=ny（上）两排角点 EMF 直接设为 value。
    // 任意角点 EMF 值都不破坏 CT 的离散 div-B=0（旋度的散度恒为零）。
    // 常数 E_z 沿边界排 → 边界 By face 增量为零，只向边界 Bx face 注入通量。
    void override_inflow_ez(int nx, int ny, double value);
    void update_faces_from_emf(int nx, int ny, double dt, double dx, double dy);
    void sync_cell_centered_from_faces(Grid& w, int nx, int ny) const;
    void apply_face_bc(int nx, int ny);

    // HLL 型哨声波速耗散（Path B）。
    // 向角点 EMF 叠加一阶迎风扩散：δEz[I][J] += (c_w/2)·ΔBy_x − (c_w/2)·ΔBx_y
    // 稳定性条件：c_w·dt/mincell ≤ 1（由 compute_dt 中的 smax_hll 项保证）
    // 参考：Iwasaki & Tomida 2025（Hall-HLL 弥散近似）
    void add_hall_hll_stabilization(Grid& w, int nx, int ny, double dx, double dy); // experimental, deprioritized
};

} // namespace my_project
