#include "my_project/asym_harris_sheet.hpp"

namespace my_project {

Vec asym_cell_ic(double x, double y, const AsymHarrisParams& ap) {
    // kick-start 扰动导出的 δB（Az 的解析偏导数）
    const double yn = y - ap.y_NL();
    const double Bx = ap.Bx_eq(y)
                    - ap.psi0 * ap.ky() * std::cos(ap.kx()*x) * std::sin(ap.ky()*yn);
    const double By =   ap.psi0 * ap.kx() * std::sin(ap.kx()*x) * std::cos(ap.ky()*yn);
    // 压力配平：用含扰动磁场的总磁压补偿，保证 p + B²/2 = P_total
    const double p = ap.P_total - 0.5 * (Bx*Bx + By*By);
    return {
        ap.rho_eq(y),  // ρ  — 非对称密度剖面
        0.0,           // vx
        0.0,           // vy
        0.0,           // vz
        p,             // p  — 配平后：P_total - (Bx²+By²)/2
        Bx,            // Bx — 平衡 + kick-start 扰动
        By,            // By — 仅扰动
        0.0,           // Bz
        0.0            // ψ  (GLM 散度清洗量，t=0 时为零)
    };
}

// CT 面心磁场：由 Az 矢势线积分得到，∇·B = 0 精确成立
double asym_bx_face(double x, double y, double dy, const AsymHarrisParams& ap) {
    return (ap.Az(x, y + 0.5*dy) - ap.Az(x, y - 0.5*dy)) / dy;
}

double asym_by_face(double x, double y, double dx, const AsymHarrisParams& ap) {
    return -(ap.Az(x + 0.5*dx, y) - ap.Az(x - 0.5*dx, y)) / dx;
}

} // namespace my_project
