#include "my_project/state.hpp"

#include <algorithm>
#include <cmath>

namespace my_project {

double minbee(double r) {
    if (r <= 0.0) return 0.0;
    if (r <= 1.0) return r;
    return 1.0;
}

Vec pri2con(const Vec& w, double gamma) {
    Vec u(NVAR);
    const double rho = w[0], vx = w[1], vy = w[2], vz = w[3], p = w[4];
    const double Bx = w[5], By = w[6], Bz = w[7];
    u[0] = rho;
    u[1] = rho * vx;
    u[2] = rho * vy;
    u[3] = rho * vz;
    u[4] = p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)
         + 0.5 * (Bx * Bx + By * By + Bz * Bz);
    u[5] = Bx;
    u[6] = By;
    u[7] = Bz;
    u[8] = w[8];
    return u;
}

Vec con2pri(const Vec& u, double gamma) {
    Vec w(NVAR);
    const double rho = std::max(u[0], 1e-10);
    const double vx = u[1] / rho, vy = u[2] / rho, vz = u[3] / rho;
    const double Bx = u[5], By = u[6], Bz = u[7];
    double p = (gamma - 1.0) * (u[4] - 0.5 * rho * (vx * vx + vy * vy + vz * vz)
              - 0.5 * (Bx * Bx + By * By + Bz * Bz));
    p = std::max(p, 1e-10);
    w[0] = rho; w[1] = vx; w[2] = vy; w[3] = vz; w[4] = p;
    w[5] = Bx;  w[6] = By; w[7] = Bz; w[8] = u[8];
    return w;
}

Vec phys_flux(const Vec& u, double gamma, bool glm_on, double ch) {
    const double rho = std::max(u[0], 1e-10);
    const double vx = u[1] / rho, vy = u[2] / rho, vz = u[3] / rho;
    const double Bx = u[5], By = u[6], Bz = u[7], psi = u[8];
    const double B2 = Bx * Bx + By * By + Bz * Bz;
    double p = (gamma - 1.0) * (u[4] - 0.5 * rho * (vx * vx + vy * vy + vz * vz) - 0.5 * B2);
    p = std::max(p, 1e-10);
    const double pt = p + 0.5 * B2;
    const double vB = vx * Bx + vy * By + vz * Bz;

    Vec F(NVAR, 0.0);
    F[0] = rho * vx;
    F[1] = rho * vx * vx + pt - Bx * Bx;
    F[2] = rho * vy * vx - By * Bx;
    F[3] = rho * vz * vx - Bz * Bx;
    F[4] = (u[4] + pt) * vx - vB * Bx;
    F[6] = By * vx - vy * Bx;
    F[7] = Bz * vx - vz * Bx;
    if (glm_on) {
        F[5] = psi;
        F[8] = ch * ch * Bx;
    }
    return F;
}

double calc_cf(double rho, double p, double Bx, double By, double Bz, double gamma) {
    rho = std::max(rho, 1e-10);
    p = std::max(p, 1e-10);
    const double cs2 = gamma * p / rho;
    const double B2 = Bx * Bx + By * By + Bz * Bz;
    const double va2 = B2 / rho;
    const double ca2 = Bx * Bx / rho;
    double disc = (cs2 + va2) * (cs2 + va2) - 4.0 * cs2 * ca2;
    disc = std::max(disc, 0.0);
    return std::sqrt(0.5 * (cs2 + va2 + std::sqrt(disc)));
}

} // namespace my_project
