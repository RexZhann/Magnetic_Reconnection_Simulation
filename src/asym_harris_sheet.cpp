#include "my_project/asym_harris_sheet.hpp"

namespace my_project {

Vec asym_cell_ic(double x, double y, const AsymHarrisParams& ap) {
    // Kick-start perturbation: analytic derivatives of Az
    const double yn = y - ap.y_NL();
    const double Bx = ap.Bx_eq(y)
                    - ap.psi0 * ap.ky() * std::cos(ap.kx()*x) * std::sin(ap.ky()*yn);
    const double By =   ap.psi0 * ap.kx() * std::sin(ap.kx()*x) * std::cos(ap.ky()*yn);
    // Pressure balance with perturbed field: p + B²/2 = P_total
    const double p = ap.P_total - 0.5 * (Bx*Bx + By*By);
    return {
        ap.rho_eq(y),  // rho (asymmetric profile)
        0.0,           // vx
        0.0,           // vy
        0.0,           // vz
        p,
        Bx,
        By,
        0.0,           // Bz
        0.0            // psi (GLM scalar)
    };
}

// Face-centred B from Az line integral: exact discrete ∇·B = 0
double asym_bx_face(double x, double y, double dy, const AsymHarrisParams& ap) {
    return (ap.Az(x, y + 0.5*dy) - ap.Az(x, y - 0.5*dy)) / dy;
}

double asym_by_face(double x, double y, double dx, const AsymHarrisParams& ap) {
    return -(ap.Az(x + 0.5*dx, y) - ap.Az(x - 0.5*dx, y)) / dx;
}

} // namespace my_project
