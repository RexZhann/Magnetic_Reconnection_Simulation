#include "my_project/riemann.hpp"

#include "my_project/state.hpp"

#include <algorithm>
#include <cmath>

namespace my_project {

Vec force_flux(const Vec& uL, const Vec& uR, double dt, double dx,
               double gamma, bool glm_on, double ch) {
    Vec fL = phys_flux(uL, gamma, glm_on, ch);
    Vec fR = phys_flux(uR, gamma, glm_on, ch);
    Vec FLF(NVAR), uh(NVAR), F(NVAR);
    for (int k = 0; k < NVAR; ++k) {
        FLF[k] = 0.5 * (dx / dt) * (uL[k] - uR[k]) + 0.5 * (fL[k] + fR[k]);
        uh[k] = 0.5 * (uL[k] + uR[k]) - 0.5 * (dt / dx) * (fR[k] - fL[k]);
    }
    Vec FRI = phys_flux(uh, gamma, glm_on, ch);
    for (int k = 0; k < NVAR; ++k) F[k] = 0.5 * (FLF[k] + FRI[k]);
    if (glm_on) {
        F[5] = 0.5 * (uL[8] + uR[8]) - 0.5 * ch * (uR[5] - uL[5]);
        F[8] = 0.5 * ch * ch * (uL[5] + uR[5]) - 0.5 * ch * (uR[8] - uL[8]);
    } else {
        F[5] = 0.0;
        F[8] = 0.0;
    }
    return F;
}

Vec hlld_flux(const Vec& uL, const Vec& uR, double g,
              bool glm_on, double ch) {
    double rL = std::max(uL[0], 1e-10);
    double vxL = uL[1] / rL, vyL = uL[2] / rL, vzL = uL[3] / rL;
    double EL = uL[4], BxL = uL[5], ByL = uL[6], BzL = uL[7];
    double pL = (g - 1) * (EL - 0.5 * rL * (vxL * vxL + vyL * vyL + vzL * vzL)
               - 0.5 * (BxL * BxL + ByL * ByL + BzL * BzL));
    pL = std::max(pL, 1e-10);

    double rR = std::max(uR[0], 1e-10);
    double vxR = uR[1] / rR, vyR = uR[2] / rR, vzR = uR[3] / rR;
    double ER = uR[4], BxR = uR[5], ByR = uR[6], BzR = uR[7];
    double pR = (g - 1) * (ER - 0.5 * rR * (vxR * vxR + vyR * vyR + vzR * vzR)
               - 0.5 * (BxR * BxR + ByR * ByR + BzR * BzR));
    pR = std::max(pR, 1e-10);

    double Bx = 0.5 * (BxL + BxR);
    double ptL = pL + 0.5 * (Bx * Bx + ByL * ByL + BzL * BzL);
    double ptR = pR + 0.5 * (Bx * Bx + ByR * ByR + BzR * BzR);
    double cfL = calc_cf(rL, pL, Bx, ByL, BzL, g);
    double cfR = calc_cf(rR, pR, Bx, ByR, BzR, g);
    double SL = std::min(vxL - cfL, vxR - cfR);
    double SR = std::max(vxL + cfL, vxR + cfR);

    Vec FL = phys_flux(uL, g, glm_on, ch), FR = phys_flux(uR, g, glm_on, ch);
    auto add_glm = [&](Vec& F) {
        if (glm_on) {
            F[5] = 0.5 * (uL[8] + uR[8]) - 0.5 * ch * (BxR - BxL);
            F[8] = 0.5 * ch * ch * (BxL + BxR) - 0.5 * ch * (uR[8] - uL[8]);
        } else {
            F[5] = 0.0;
            F[8] = 0.0;
        }
    };
    if (SL >= 0) { Vec F = FL; add_glm(F); return F; }
    if (SR <= 0) { Vec F = FR; add_glm(F); return F; }

    auto hll = [&]() -> Vec {
        Vec F(NVAR);
        for (int k = 0; k < NVAR; ++k) {
            F[k] = (SR * FL[k] - SL * FR[k] + SL * SR * (uR[k] - uL[k])) / (SR - SL);
        }
        add_glm(F);
        return F;
    };

    double dSL = SL - vxL, dSR = SR - vxR;
    double denom_SM = dSR * rR - dSL * rL;
    if (std::fabs(denom_SM) < 1e-20) return hll();
    double SM = (dSR * rR * vxR - dSL * rL * vxL - ptR + ptL) / denom_SM;
    double ptS = (dSR * rR * ptL - dSL * rL * ptR + rL * rR * dSR * dSL * (vxR - vxL)) / denom_SM;
    ptS = std::max(ptS, 0.0);

    auto compute_ustar = [&](double rho, double vx, double vy, double vz,
                             double E, double By, double Bz, double pt,
                             double S, bool& ok) -> Vec {
        double dS = S - vx, dSM = S - SM;
        if (std::fabs(dSM) < 1e-14) { ok = false; return Vec(NVAR, 0.0); }
        double rho_s = rho * dS / dSM;
        if (rho_s < 1e-10) { ok = false; return Vec(NVAR, 0.0); }
        double dd = rho * dS * dSM - Bx * Bx;
        double vy_s, vz_s, By_s, Bz_s;
        if (std::fabs(dd) < 1e-12 * (std::fabs(rho * dS * dSM) + Bx * Bx + 1e-30)) {
            vy_s = vy; vz_s = vz; By_s = By; Bz_s = Bz;
        } else {
            double f1 = Bx * (dS - dSM) / dd;
            vy_s = vy - By * f1;
            vz_s = vz - Bz * f1;
            double f2 = (rho * dS * dS - Bx * Bx) / dd;
            By_s = By * f2;
            Bz_s = Bz * f2;
        }
        double vB = vx * Bx + vy * By + vz * Bz;
        double vBs = SM * Bx + vy_s * By_s + vz_s * Bz_s;
        double E_s = (dS * E - pt * vx + ptS * SM + Bx * (vB - vBs)) / dSM;
        Vec us(NVAR, 0.0);
        us[0] = rho_s; us[1] = rho_s * SM; us[2] = rho_s * vy_s; us[3] = rho_s * vz_s;
        us[4] = E_s; us[5] = Bx; us[6] = By_s; us[7] = Bz_s;
        ok = true;
        return us;
    };

    bool okL = false, okR = false;
    Vec usL = compute_ustar(rL, vxL, vyL, vzL, EL, ByL, BzL, ptL, SL, okL);
    Vec usR = compute_ustar(rR, vxR, vyR, vzR, ER, ByR, BzR, ptR, SR, okR);
    if (!okL || !okR) return hll();

    double srL = std::sqrt(usL[0]), srR = std::sqrt(usR[0]);
    double sign_Bx = (Bx >= 0) ? 1.0 : -1.0;
    double SLS = SM - std::fabs(Bx) / srL;
    double SRS = SM + std::fabs(Bx) / srR;

    double dsr = srL + srR;
    double vy_ss, vz_ss, By_ss, Bz_ss;
    if (dsr < 1e-14) {
        vy_ss = vz_ss = By_ss = Bz_ss = 0.0;
    } else {
        double vyLs = usL[2] / usL[0], vzLs = usL[3] / usL[0];
        double vyRs = usR[2] / usR[0], vzRs = usR[3] / usR[0];
        vy_ss = (srL * vyLs + srR * vyRs + (usR[6] - usL[6]) * sign_Bx) / dsr;
        vz_ss = (srL * vzLs + srR * vzRs + (usR[7] - usL[7]) * sign_Bx) / dsr;
        By_ss = (srL * usR[6] + srR * usL[6] + srL * srR * (vyRs - vyLs) * sign_Bx) / dsr;
        Bz_ss = (srL * usR[7] + srR * usL[7] + srL * srR * (vzRs - vzLs) * sign_Bx) / dsr;
    }

    double vBss = SM * Bx + vy_ss * By_ss + vz_ss * Bz_ss;
    auto make_uss = [&](const Vec& us, double esign) -> Vec {
        Vec uss(NVAR, 0.0);
        uss[0] = us[0]; uss[1] = us[0] * SM;
        uss[2] = us[0] * vy_ss; uss[3] = us[0] * vz_ss;
        uss[5] = Bx; uss[6] = By_ss; uss[7] = Bz_ss;
        double vBs = SM * Bx + (us[2] / us[0]) * us[6] + (us[3] / us[0]) * us[7];
        uss[4] = us[4] + esign * std::sqrt(us[0]) * sign_Bx * (vBs - vBss);
        return uss;
    };

    Vec ussL = make_uss(usL, -1.0);
    Vec ussR = make_uss(usR, +1.0);
    for (int k = 0; k < NVAR; ++k) {
        if (!std::isfinite(ussL[k]) || !std::isfinite(ussR[k])) return hll();
    }
    if (ussL[0] < 0.0 || ussR[0] < 0.0) return hll();

    Vec F(NVAR, 0.0);
    if (SLS >= 0) {
        for (int k = 0; k < NVAR; ++k) F[k] = FL[k] + SL * (usL[k] - uL[k]);
    } else if (SM >= 0) {
        for (int k = 0; k < NVAR; ++k) F[k] = FL[k] + SLS * ussL[k] - (SLS - SL) * usL[k] - SL * uL[k];
    } else if (SRS >= 0) {
        for (int k = 0; k < NVAR; ++k) F[k] = FR[k] + SRS * ussR[k] - (SRS - SR) * usR[k] - SR * uR[k];
    } else {
        for (int k = 0; k < NVAR; ++k) F[k] = FR[k] + SR * (usR[k] - uR[k]);
    }
    for (int k = 0; k < NVAR; ++k) if (!std::isfinite(F[k])) return hll();
    add_glm(F);
    return F;
}

} // namespace my_project
