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
    virtual void set_adiabatic_index(double /*gamma*/) {}

    // Set the boundary condition types so that the controller can apply the
    // correct face-B BC and periodic EMF wrapping.
    virtual void set_boundary_conditions(BC /*bcx*/, BC /*bcy*/) {}

    // CT interface: fill a contiguous buffer with the face-centered normal B
    // for a 1D sweep row/column. buf must have size >= n+2.
    // For x-sweep of interior row j (0-indexed): buf[i] = Bx_face at interface i (i=1..n+1).
    // For y-sweep of interior col i (0-indexed): buf[j] = By_face at interface j (j=1..n+1).
    virtual void fill_face_bn_x(int /*j*/, int /*n*/, double* /*buf*/) const {}
    virtual void fill_face_bn_y(int /*i*/, int /*n*/, double* /*buf*/) const {}

    // CT interface: store interface EMFs produced by a 1D sweep.
    // emf[i] (i=1..n+1) is Ez at interface i of the slice.
    // x-sweep: Ez = -F[6]; y-sweep (rotated frame): Ez = F[6].
    virtual void store_emf_x(int /*j*/, int /*n*/, const double* /*emf*/) {}
    virtual void store_emf_y(int /*i*/, int /*n*/, const double* /*emf*/) {}
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
    void initialize(Grid& w, const RunConfig& cfg, double dx, double dy) override;
    void pre_step(Grid& w, int nx, int ny, double dt, double dx, double dy) override;
    void post_step(Grid& w, int nx, int ny, double dt, double Lx, double Ly,
                   double dx, double dy) override;
    const FaceField2D* face_field() const override { return &face_; }

    void fill_face_bn_x(int j, int n, double* buf) const override;
    void fill_face_bn_y(int i, int n, double* buf) const override;
    void store_emf_x(int j, int n, const double* emf) override;
    void store_emf_y(int i, int n, const double* emf) override;

private:
    FaceField2D face_;
    // Interface EMFs collected during sweeps:
    //   emf_x_[k][j] = Ez at x-face bx[k][j]  (k=0..nx, j=0..ny-1)
    //   emf_y_[i][k] = Ez at y-face by[i][k]   (i=0..nx-1, k=0..ny)
    ScalarField emf_x_;
    ScalarField emf_y_;
    int nx_ = 0, ny_ = 0;
    double ch_like_ = 0.0;
    double gamma_ = 1.4;
    BC bcx_ = BC::Transmissive;
    BC bcy_ = BC::Transmissive;

    void initialize_faces_from_problem(const Grid& w, const RunConfig& cfg, double dx, double dy);
    void fill_faces_from_cell_centered(const Grid& w, int nx, int ny);
    void compute_corner_emf_from_interface_emfs(int nx, int ny);
    void update_faces_from_emf(int nx, int ny, double dt, double dx, double dy);
    void sync_cell_centered_from_faces(Grid& w, int nx, int ny) const;
    void apply_face_bc(int nx, int ny);
};

} // namespace my_project
