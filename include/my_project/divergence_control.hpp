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
    virtual void initialize(Grid& w, int nx, int ny, double dx, double dy) = 0;
    virtual void pre_step(Grid&, int, int, double, double, double) {}
    virtual void post_step(Grid& w, int nx, int ny, double dt, double Lx, double Ly,
                           double dx, double dy) = 0;
    virtual const FaceField2D* face_field() const { return nullptr; }
};

class NoDivBCleaning final : public DivergenceController {
public:
    bool glm_enabled() const override { return false; }
    void update_characteristic_speed(double) override {}
    double characteristic_speed() const override { return 0.0; }
    void initialize(Grid&, int, int, double, double) override {}
    void post_step(Grid&, int, int, double, double, double, double, double) override {}
};

class GLMDivergenceCleaning final : public DivergenceController {
public:
    bool glm_enabled() const override { return true; }
    void update_characteristic_speed(double value) override { ch_ = value; }
    double characteristic_speed() const override { return ch_; }
    void initialize(Grid&, int, int, double, double) override {}
    void post_step(Grid& w, int nx, int ny, double dt, double Lx, double Ly,
                   double dx, double dy) override;
private:
    double ch_ = 0.0;
};

class CTDivergenceControl final : public DivergenceController {
public:
    bool glm_enabled() const override { return false; }
    bool uses_face_centered_b() const override { return true; }
    void update_characteristic_speed(double value) override { ch_like_ = value; }
    double characteristic_speed() const override { return ch_like_; }
    void initialize(Grid& w, int nx, int ny, double dx, double dy) override;
    void pre_step(Grid& w, int nx, int ny, double dt, double dx, double dy) override;
    void post_step(Grid& w, int nx, int ny, double dt, double Lx, double Ly,
                   double dx, double dy) override;
    const FaceField2D* face_field() const override { return &face_; }
private:
    FaceField2D face_;
    double ch_like_ = 0.0;

    void fill_faces_from_cell_centered(const Grid& w, int nx, int ny);
    void compute_corner_emf(const Grid& w, int nx, int ny);
    void update_faces_from_emf(int nx, int ny, double dt, double dx, double dy);
    void sync_cell_centered_from_faces(Grid& w, int nx, int ny) const;
    void enforce_boundary_faces(Grid& w, int nx, int ny);
};

} // namespace my_project
