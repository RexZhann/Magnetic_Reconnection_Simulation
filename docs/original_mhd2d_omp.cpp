#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
#include <chrono>
#include <iomanip>

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;
static double elapsed(Clock::time_point a, Clock::time_point b){
    return Sec(b-a).count();
}
#include <math.h>

#define M_PI 3.1415926535897
/*
 * 2D Ideal MHD Solver: SLIC + HLLD + GLM Divergence Cleaning
 * 9 components: rho, rho*vx, rho*vy, rho*vz, E, Bx, By, Bz, psi
 *
 * Usage: ./mhd2d <test> <nx> <ny> [glm] [solver]
 *   test:   0=CylExplosion 1=BrioWu-x 2=BrioWu-y 3=OrszagTang 4=Rotor
 *   glm:    0=off, 1=on (default=1)
 *   solver: 0=FORCE, 1=HLLD (default=1)
 *
 * Bug fixes vs. original:
 *   [FIX 1] Minbee limiter: for r>1 now correctly returns 1.0 (was 2/(1+r))
 *   [FIX 2] HLLD make_uss: right double-star energy now uses +sign (was -sign for both)
 *            Derivation: R-H across right Alfven wave gives
 *              E**_R = E*_R + sqrt(rho*_R)*sgn(Bx)*(v*_R.B*_R - v**.B**)
 *            while the left state uses a minus sign.
 */

const int NVAR = 9;
using Vec = std::vector<double>;
using Row = std::vector<Vec>;
using Grid = std::vector<std::vector<Vec>>;


bool   g_glm_on = true;
double g_ch     = 0.0;

// ======================== Limiter ========================
// Standard MinBee (= MinMod) limiter.
// [FIX 1] For r > 1 return 1.0, NOT 2/(1+r).
// The old formula made the scheme unnecessarily diffusive in smooth regions.
double Minbee(double r) {
    if (r <= 0) return 0.0;
    if (r <= 1) return r;
    return 1.0;           // FIX 1: was  std::min(1.0, 2.0/(1.0+r))
}

// ======================== Conversions ========================
Vec pri2con(const Vec& w, double g) {
    Vec u(NVAR);
    double rho=w[0],vx=w[1],vy=w[2],vz=w[3],p=w[4];
    double Bx=w[5],By=w[6],Bz=w[7];
    u[0]=rho; u[1]=rho*vx; u[2]=rho*vy; u[3]=rho*vz;
    u[4]=p/(g-1)+0.5*rho*(vx*vx+vy*vy+vz*vz)+0.5*(Bx*Bx+By*By+Bz*Bz);
    u[5]=Bx; u[6]=By; u[7]=Bz; u[8]=w[8];
    return u;
}

Vec con2pri(const Vec& u, double g) {
    Vec w(NVAR);
    double rho=std::max(u[0],1e-10);
    double vx=u[1]/rho,vy=u[2]/rho,vz=u[3]/rho;
    double Bx=u[5],By=u[6],Bz=u[7];
    double p=(g-1)*(u[4]-0.5*rho*(vx*vx+vy*vy+vz*vz)-0.5*(Bx*Bx+By*By+Bz*Bz));
    if(p<1e-10) p=1e-10;
    w[0]=rho;w[1]=vx;w[2]=vy;w[3]=vz;w[4]=p;
    w[5]=Bx;w[6]=By;w[7]=Bz;w[8]=u[8];
    return w;
}

// ======================== Physical flux (x-dir) ========================
Vec phys_flux(const Vec& u, double g) {
    double rho=std::max(u[0],1e-10);
    double vx=u[1]/rho,vy=u[2]/rho,vz=u[3]/rho;
    double Bx=u[5],By=u[6],Bz=u[7],psi=u[8];
    double B2=Bx*Bx+By*By+Bz*Bz;
    double p=(g-1)*(u[4]-0.5*rho*(vx*vx+vy*vy+vz*vz)-0.5*B2);
    if(p<1e-10)p=1e-10;
    double pt=p+0.5*B2, vB=vx*Bx+vy*By+vz*Bz;
    Vec F(NVAR);
    F[0]=rho*vx;
    F[1]=rho*vx*vx+pt-Bx*Bx;
    F[2]=rho*vy*vx-By*Bx;
    F[3]=rho*vz*vx-Bz*Bx;
    F[4]=(u[4]+pt)*vx-vB*Bx;
    F[6]=By*vx-vy*Bx;
    F[7]=Bz*vx-vz*Bx;
    // GLM: F_Bx = psi,  F_psi = ch^2 * Bx  (Dedner 2002)
    if(g_glm_on){ F[5]=psi; F[8]=g_ch*g_ch*Bx; }
    else         { F[5]=0;   F[8]=0; }
    return F;
}

// ======================== Fast magnetosonic speed ========================
double calc_cf(double rho, double p, double Bx, double By, double Bz, double g) {
    if(rho<1e-10)rho=1e-10; if(p<1e-10)p=1e-10;
    double cs2=g*p/rho;
    double B2=Bx*Bx+By*By+Bz*Bz;
    double va2=B2/rho, ca2=Bx*Bx/rho;
    double disc=(cs2+va2)*(cs2+va2)-4.0*cs2*ca2;
    if(disc<0)disc=0;
    return std::sqrt(0.5*(cs2+va2+std::sqrt(disc)));
}

// ======================== Solver selection ========================
int g_solver = 1;  // 0=FORCE, 1=HLLD

// ======================== FORCE flux ========================
Vec force_flux(const Vec& uL, const Vec& uR, double dt, double dx, double g) {
    Vec fL=phys_flux(uL,g), fR=phys_flux(uR,g);
    Vec FLF(NVAR),uh(NVAR),F(NVAR);
    for(int k=0;k<NVAR;k++)
        FLF[k]=0.5*(dx/dt)*(uL[k]-uR[k])+0.5*(fL[k]+fR[k]);
    for(int k=0;k<NVAR;k++)
        uh[k]=0.5*(uL[k]+uR[k])-0.5*(dt/dx)*(fR[k]-fL[k]);
    Vec FRI=phys_flux(uh,g);
    for(int k=0;k<NVAR;k++) F[k]=0.5*(FLF[k]+FRI[k]);
    // GLM: overwrite Bn and psi fluxes with Roe-upwind GLM flux
    if(g_glm_on){
        F[5]=0.5*(uL[8]+uR[8])-0.5*g_ch*(uR[5]-uL[5]);
        F[8]=0.5*g_ch*g_ch*(uL[5]+uR[5])-0.5*g_ch*(uR[8]-uL[8]);
    } else { F[5]=0; F[8]=0; }
    return F;
}

// ======================== HLLD Riemann solver ========================
// Miyoshi & Kusano (2005), Journal of Computational Physics 208, 315-344.
Vec hlld_flux(const Vec& uL, const Vec& uR, double g) {
    // --- Primitives ---
    double rL=std::max(uL[0],1e-10);
    double vxL=uL[1]/rL,vyL=uL[2]/rL,vzL=uL[3]/rL;
    double EL=uL[4],BxL=uL[5],ByL=uL[6],BzL=uL[7];
    double pL=(g-1)*(EL-0.5*rL*(vxL*vxL+vyL*vyL+vzL*vzL)
              -0.5*(BxL*BxL+ByL*ByL+BzL*BzL));
    if(pL<1e-10)pL=1e-10;

    double rR=std::max(uR[0],1e-10);
    double vxR=uR[1]/rR,vyR=uR[2]/rR,vzR=uR[3]/rR;
    double ER=uR[4],BxR=uR[5],ByR=uR[6],BzR=uR[7];
    double pR=(g-1)*(ER-0.5*rR*(vxR*vxR+vyR*vyR+vzR*vzR)
              -0.5*(BxR*BxR+ByR*ByR+BzR*BzR));
    if(pR<1e-10)pR=1e-10;

    // Shared normal field (average across interface)
    double Bx=0.5*(BxL+BxR);

    // Total pressures using shared Bx (M&K Eq. 21)
    double ptL=pL+0.5*(Bx*Bx+ByL*ByL+BzL*BzL);
    double ptR=pR+0.5*(Bx*Bx+ByR*ByR+BzR*BzR);

    // --- Wave speed estimates (Davis bounds) ---
    double cfL=calc_cf(rL,pL,Bx,ByL,BzL,g);
    double cfR=calc_cf(rR,pR,Bx,ByR,BzR,g);
    double SL=std::min(vxL-cfL, vxR-cfR);
    double SR=std::max(vxL+cfL, vxR+cfR);

    // --- Physical fluxes from original left/right states ---
    Vec FL=phys_flux(uL,g), FR=phys_flux(uR,g);

    // --- GLM Roe flux (overwrites Bx and psi components on all exit paths) ---
    // F_Bx  = 0.5*(psiL+psiR) - 0.5*ch*(BxR-BxL)
    // F_psi = 0.5*ch^2*(BxL+BxR) - 0.5*ch*(psiR-psiL)
    auto add_glm=[&](Vec& F){
        if(g_glm_on){
            F[5]=0.5*(uL[8]+uR[8])-0.5*g_ch*(BxR-BxL);
            F[8]=0.5*g_ch*g_ch*(BxL+BxR)-0.5*g_ch*(uR[8]-uL[8]);
        } else { F[5]=0; F[8]=0; }
    };

    // --- Trivial cases: supersonic flow ---
    if(SL>=0){ Vec F=FL; add_glm(F); return F; }
    if(SR<=0){ Vec F=FR; add_glm(F); return F; }

    // --- HLL flux (fallback for degenerate cases) ---
    auto hll=[&]() -> Vec {
        Vec F(NVAR);
        for(int k=0;k<NVAR;k++)
            F[k]=(SR*FL[k]-SL*FR[k]+SL*SR*(uR[k]-uL[k]))/(SR-SL);
        add_glm(F);
        return F;
    };

    // --- Contact speed SM (M&K Eq. 38) ---
    double dSL=SL-vxL, dSR=SR-vxR;
    double denom_SM=dSR*rR-dSL*rL;
    if(std::fabs(denom_SM)<1e-20) return hll();
    double SM=(dSR*rR*vxR-dSL*rL*vxL-ptR+ptL)/denom_SM;
    // Star total pressure (M&K Eq. 41)
    double ptS=(dSR*rR*ptL-dSL*rL*ptR+rL*rR*dSR*dSL*(vxR-vxL))/denom_SM;
    if(ptS<0) ptS=0;

    // --- U* states (M&K Eqs. 43-45) ---
    auto compute_ustar=[&](double rho,double vx,double vy,double vz,
                           double E,double By,double Bz,double pt,
                           double S,bool& ok) -> Vec {
        double dS=S-vx, dSM=S-SM;
        if(std::fabs(dSM)<1e-14){ok=false; return Vec(NVAR,0);}
        double rho_s=rho*dS/dSM;
        if(rho_s<1e-10){ok=false; return Vec(NVAR,0);}

        double dd=rho*dS*dSM-Bx*Bx;
        double vy_s,vz_s,By_s,Bz_s;
        if(std::fabs(dd)<1e-12*(std::fabs(rho*dS*dSM)+Bx*Bx+1e-30)){
            // Near Bx->0 or S->SM: degenerate, keep transverse components
            vy_s=vy; vz_s=vz; By_s=By; Bz_s=Bz;
        } else {
            double f1=Bx*(dS-dSM)/dd;   // = Bx*(SM-vx)/dd
            vy_s=vy-By*f1; vz_s=vz-Bz*f1;
            double f2=(rho*dS*dS-Bx*Bx)/dd;
            By_s=By*f2; Bz_s=Bz*f2;
        }
        double vB =vx *Bx+vy  *By  +vz  *Bz;
        double vBs=SM *Bx+vy_s*By_s+vz_s*Bz_s;
        double E_s=(dS*E-pt*vx+ptS*SM+Bx*(vB-vBs))/dSM;

        Vec us(NVAR,0);
        us[0]=rho_s;us[1]=rho_s*SM;us[2]=rho_s*vy_s;us[3]=rho_s*vz_s;
        us[4]=E_s;us[5]=Bx;us[6]=By_s;us[7]=Bz_s;
        ok=true;
        return us;
    };

    bool okL,okR;
    Vec usL=compute_ustar(rL,vxL,vyL,vzL,EL,ByL,BzL,ptL,SL,okL);
    Vec usR=compute_ustar(rR,vxR,vyR,vzR,ER,ByR,BzR,ptR,SR,okR);
    if(!okL||!okR) return hll();

    // --- Alfven wave speeds ---
    double srL=std::sqrt(usL[0]), srR=std::sqrt(usR[0]);
    double sign_Bx=(Bx>=0)?1.0:-1.0;
    double SLS=SM-std::fabs(Bx)/srL;
    double SRS=SM+std::fabs(Bx)/srR;

    // --- U** states: shared transverse fields (M&K Eqs. 46-47) ---
    double dsr=srL+srR;
    double vy_ss,vz_ss,By_ss,Bz_ss;
    if(dsr<1e-14){
        vy_ss=0;vz_ss=0;By_ss=0;Bz_ss=0;
    } else {
        double vyLs=usL[2]/usL[0],vzLs=usL[3]/usL[0];
        double vyRs=usR[2]/usR[0],vzRs=usR[3]/usR[0];
        vy_ss=(srL*vyLs+srR*vyRs+(usR[6]-usL[6])*sign_Bx)/dsr;
        vz_ss=(srL*vzLs+srR*vzRs+(usR[7]-usL[7])*sign_Bx)/dsr;
        By_ss=(srL*usR[6]+srR*usL[6]+srL*srR*(vyRs-vyLs)*sign_Bx)/dsr;
        Bz_ss=(srL*usR[7]+srR*usL[7]+srL*srR*(vzRs-vzLs)*sign_Bx)/dsr;
    }

    // --- U** energy: R-H across Alfven waves (M&K Eqs. 48-49) ---
    //
    // [FIX 2] Left and right double-star energies have OPPOSITE signs:
    //   E**_L = E*_L - sqrt(rho*_L)*sgn(Bx)*(v*_L.B*_L - v**.B**)
    //   E**_R = E*_R + sqrt(rho*_R)*sgn(Bx)*(v*_R.B*_R - v**.B**)
    //
    // The original code used  us[4] - sqrt(...)  for both states,
    // which is wrong for usR (missing the sign flip).
    // The esign parameter encodes: -1 for left state, +1 for right state.
    //
    double vBss = SM*Bx + vy_ss*By_ss + vz_ss*Bz_ss;

    auto make_uss=[&](const Vec& us, double esign) -> Vec {
        Vec uss(NVAR,0);
        uss[0]=us[0]; uss[1]=us[0]*SM;
        uss[2]=us[0]*vy_ss; uss[3]=us[0]*vz_ss;
        uss[5]=Bx; uss[6]=By_ss; uss[7]=Bz_ss;
        double vBs = SM*Bx + (us[2]/us[0])*us[6] + (us[3]/us[0])*us[7];
        // FIX 2: esign = -1 for left, +1 for right
        uss[4] = us[4] + esign * std::sqrt(us[0]) * sign_Bx * (vBs - vBss);
        return uss;
    };

    Vec ussL=make_uss(usL, -1.0);   // FIX 2: pass esign=-1
    Vec ussR=make_uss(usR, +1.0);   // FIX 2: pass esign=+1  (was make_uss(usR) with -1)

    // Validate double-star states
    for(int k=0;k<NVAR;k++){
        if(!std::isfinite(ussL[k])||!std::isfinite(ussR[k])) return hll();
    }
    if(ussL[0]<0||ussR[0]<0) return hll();

    // --- Select flux region (M&K Eq. 66) ---
    // Region boundaries: SL <= SLS <= SM <= SRS <= SR
    Vec F(NVAR,0);
    if(SLS>=0){
        // SL <= 0 <= SLS : F = F*_L = F_L + SL*(u*_L - u_L)
        for(int k=0;k<NVAR;k++) F[k]=FL[k]+SL*(usL[k]-uL[k]);
    } else if(SM>=0){
        // SLS <= 0 <= SM : F = F**_L = F_L + SL*(u*_L-u_L) + SLS*(u**_L-u*_L)
        for(int k=0;k<NVAR;k++) F[k]=FL[k]+SLS*ussL[k]-(SLS-SL)*usL[k]-SL*uL[k];
    } else if(SRS>=0){
        // SM <= 0 <= SRS : F = F**_R = F_R + SR*(u*_R-u_R) + SRS*(u**_R-u*_R)
        for(int k=0;k<NVAR;k++) F[k]=FR[k]+SRS*ussR[k]-(SRS-SR)*usR[k]-SR*uR[k];
    } else {
        // SRS <= 0 <= SR : F = F*_R = F_R + SR*(u*_R - u_R)
        for(int k=0;k<NVAR;k++) F[k]=FR[k]+SR*(usR[k]-uR[k]);
    }

    // Sanity check: fall back to HLL if HLLD produced non-finite values
    for(int k=0;k<NVAR;k++){
        if(!std::isfinite(F[k])) return hll();
    }

    add_glm(F);
    return F;
}

// ======================== Thread-local scratch buffers ========================
// Pre-allocate all temporary Row arrays used in slic_step to avoid
// repeated heap allocation/deallocation in parallel regions.
struct ScratchBuf {
    int cap = 0;
    Row uc, d0, d1, delta, xL, xR, hL, hR, iflx, s_row;
    void ensure(int N) {
        if (N <= cap) return;
        cap = N;
        auto resize_row = [&](Row& r){ r.assign(N, Vec(NVAR, 0.0)); };
        resize_row(uc); resize_row(d0); resize_row(d1); resize_row(delta);
        resize_row(xL); resize_row(xR); resize_row(hL); resize_row(hR);
        resize_row(iflx); resize_row(s_row);
    }
};

// ======================== SLIC 1D step ========================
void slic_step(ScratchBuf& sc, Row& wp, int n, double dt, double dx, double g) {
    int N=n+4;
    sc.ensure(N);
    Row& uc=sc.uc; Row& d0=sc.d0; Row& d1=sc.d1; Row& delta=sc.delta;
    Row& xL=sc.xL; Row& xR=sc.xR; Row& hL=sc.hL; Row& hR=sc.hR;
    Row& iflx=sc.iflx;

    for(int i=0;i<N;i++) uc[i]=pri2con(wp[i],g);

    // --- Slope computation ---
    for(int i=1;i<N-1;i++)
        for(int k=0;k<NVAR;k++){
            d0[i][k]=uc[i][k]-uc[i-1][k];
            d1[i][k]=uc[i+1][k]-uc[i][k];
            delta[i][k]=0.5*(d0[i][k]+d1[i][k]);
        }

    // --- MUSCL reconstruction ---
    for(int i=1;i<N-1;i++){
        for(int k=0;k<NVAR;k++){
            double den=d1[i][k];
            double r=(std::fabs(den)<1e-30)?0.0:d0[i][k]/den;
            double phi=Minbee(r);
            xL[i][k]=uc[i][k]-0.5*phi*delta[i][k];
            xR[i][k]=uc[i][k]+0.5*phi*delta[i][k];
        }
        // Positivity guard on density
        if(xL[i][0]<0 || xR[i][0]<0){
            xL[i]=uc[i]; xR[i]=uc[i];
        }
    }

    // --- MUSCL-Hancock half-step predictor ---
    auto check_pressure=[&](const Vec& u) -> double {
        double rho=std::max(u[0],1e-14);
        double vx=u[1]/rho,vy=u[2]/rho,vz=u[3]/rho;
        double Bx=u[5],By=u[6],Bz=u[7];
        return (g-1)*(u[4]-0.5*rho*(vx*vx+vy*vy+vz*vz)
                     -0.5*(Bx*Bx+By*By+Bz*Bz));
    };

    for(int i=1;i<N-1;i++){
        Vec fL=phys_flux(xL[i],g),fR=phys_flux(xR[i],g);
        for(int k=0;k<NVAR;k++){
            hL[i][k]=xL[i][k]-0.5*(dt/dx)*(fR[k]-fL[k]);
            hR[i][k]=xR[i][k]-0.5*(dt/dx)*(fR[k]-fL[k]);
        }
        // Positivity fallback: drop to 1st order if half-step state is unphysical
        if(hL[i][0]<0 || hR[i][0]<0 ||
           check_pressure(hL[i])<0 || check_pressure(hR[i])<0){
            hL[i]=uc[i];
            hR[i]=uc[i];
        }
    }

    // --- Interface fluxes ---
    for(int i=1;i<n+2;i++){
        if(g_solver==0)
            iflx[i]=force_flux(hR[i],hL[i+1],dt,dx,g);
        else
            iflx[i]=hlld_flux(hR[i],hL[i+1],g);
    }

    // --- Conservative update ---
    // When GLM is OFF, skip k=5 (normal B component): its x-flux is zero by construction.
    // In the y-sweep the swap makes k=5 correspond to By, which is likewise skipped.
    for(int i=2;i<n+2;i++)
        for(int k=0;k<NVAR;k++){
            if(!g_glm_on && k==5) continue;
            uc[i][k]-=(dt/dx)*(iflx[i][k]-iflx[i-1][k]);
        }

    for(int i=2;i<n+2;i++) wp[i]=con2pri(uc[i],g);
}

// ======================== Boundary conditions ========================
enum BC{TRANSMISSIVE,PERIODIC};

void apply_bc(Grid& w,int nx,int ny,BC bcx,BC bcy){
    for(int j=0;j<ny+4;j++){
        if(bcx==TRANSMISSIVE){
            w[0][j]=w[2][j];w[1][j]=w[2][j];
            w[nx+2][j]=w[nx+1][j];w[nx+3][j]=w[nx+1][j];
        } else {
            w[0][j]=w[nx][j];w[1][j]=w[nx+1][j];
            w[nx+2][j]=w[2][j];w[nx+3][j]=w[3][j];
        }
    }
    for(int i=0;i<nx+4;i++){
        if(bcy==TRANSMISSIVE){
            w[i][0]=w[i][2];w[i][1]=w[i][2];
            w[i][ny+2]=w[i][ny+1];w[i][ny+3]=w[i][ny+1];
        } else {
            w[i][0]=w[i][ny];w[i][1]=w[i][ny+1];
            w[i][ny+2]=w[i][2];w[i][ny+3]=w[i][3];
        }
    }
}

// ======================== Sweeps ========================
void sweep_x(Grid& w,int nx,int ny,double dt,double dx,double g){
    // Each row j is fully independent: no data sharing between rows.
    #pragma omp parallel
    {
        ScratchBuf sc;
        sc.ensure(nx+4);
        #pragma omp for schedule(static)
        for(int j=2;j<ny+2;j++){
            Row& s=sc.s_row;
            for(int i=0;i<nx+4;i++) s[i]=w[i][j];
            slic_step(sc,s,nx,dt,dx,g);
            for(int i=2;i<nx+2;i++) w[i][j]=s[i];
        }
    }
}

// y-sweep: swap (vx<->vy, Bx<->By), run x-solver, swap back
void sweep_y(Grid& w,int nx,int ny,double dt,double dy,double g){
    // Each column i is fully independent: no data sharing between columns.
    #pragma omp parallel
    {
        ScratchBuf sc;
        sc.ensure(ny+4);
        #pragma omp for schedule(static)
        for(int i=2;i<nx+2;i++){
            Row& s=sc.s_row;
            for(int j=0;j<ny+4;j++){
                s[j]=w[i][j];
                std::swap(s[j][1],s[j][2]);   // vx <-> vy
                std::swap(s[j][5],s[j][6]);   // Bx <-> By
            }
            slic_step(sc,s,ny,dt,dy,g);
            for(int j=2;j<ny+2;j++){
                std::swap(s[j][1],s[j][2]);
                std::swap(s[j][5],s[j][6]);
                w[i][j]=s[j];
            }
        }
    }
}

// ======================== CFL + ch ========================
double compute_dt(const Grid& w,int nx,int ny,
                  double dx,double dy,double g,double cfl){
    double smax=1e-10, ch_loc=0;
    #pragma omp parallel for collapse(2) reduction(max:smax,ch_loc) schedule(static)
    for(int i=2;i<nx+2;i++)
        for(int j=2;j<ny+2;j++){
            const Vec& p=w[i][j];
            double cfx=calc_cf(p[0],p[4],p[5],p[6],p[7],g);
            double cfy=calc_cf(p[0],p[4],p[6],p[5],p[7],g); // By is normal in y-dir
            double sx=std::fabs(p[1])+cfx, sy=std::fabs(p[2])+cfy;
            double s=sx/dx+sy/dy;
            if(s>smax) smax=s;
            if(sx>ch_loc) ch_loc=sx;
            if(sy>ch_loc) ch_loc=sy;
        }
    g_ch=ch_loc;
    return cfl/smax;
}

// GLM parabolic decay: psi *= exp(-ch*dt/cp), cp = cr * max(Lx,Ly), cr=0.18
void glm_decay(Grid& w,int nx,int ny,double dt,double Lx,double Ly){
    double cp=0.18*std::max(Lx,Ly);
    double f=std::exp(-g_ch*dt/cp);
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=2;i<nx+2;i++)
        for(int j=2;j<ny+2;j++) w[i][j][8]*=f;
}

std::vector<std::vector<double>> compute_divB(const Grid& w,int nx,int ny,
                                               double dx,double dy){
    std::vector<std::vector<double>> dB(nx,std::vector<double>(ny,0));
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=2;i<nx+2;i++)
        for(int j=2;j<ny+2;j++){
            double d=(w[i+1][j][5]-w[i-1][j][5])/(2*dx)
                    +(w[i][j+1][6]-w[i][j-1][6])/(2*dy);
            dB[i-2][j-2]=std::fabs(d);
        }
    return dB;
}

// ======================== Debug diagnostics ========================
struct Diagnostics {
    double min_rho, min_p, max_divB, max_psi, max_v;
};

Diagnostics compute_diagnostics(const Grid& w, int nx, int ny,
                                 double dx, double dy, double /*g*/) {
    Diagnostics d;
    d.min_rho = 1e30; d.min_p = 1e30;
    d.max_divB = 0; d.max_psi = 0; d.max_v = 0;
    double min_rho=1e30, min_p=1e30, max_divB=0, max_psi=0, max_v=0;
    #pragma omp parallel for collapse(2) schedule(static) \
        reduction(min:min_rho,min_p) reduction(max:max_divB,max_psi,max_v)
    for (int i = 2; i < nx+2; i++)
        for (int j = 2; j < ny+2; j++) {
            const Vec& p = w[i][j];
            if (p[0] < min_rho) min_rho = p[0];
            if (p[4] < min_p)   min_p   = p[4];
            double ap = std::fabs(p[8]);
            if (ap > max_psi) max_psi = ap;
            double v = std::sqrt(p[1]*p[1]+p[2]*p[2]+p[3]*p[3]);
            if (v > max_v) max_v = v;
            // Central-difference div B (ghost cells ensure i±1, j±1 valid)
            double dBx = (w[i+1][j][5] - w[i-1][j][5]) / (2*dx);
            double dBy = (w[i][j+1][6] - w[i][j-1][6]) / (2*dy);
            double ad  = std::fabs(dBx + dBy);
            if (ad > max_divB) max_divB = ad;
        }
    d.min_rho = min_rho; d.min_p = min_p;
    d.max_divB = max_divB; d.max_psi = max_psi; d.max_v = max_v;
    return d;
}

// ======================== Main ========================
using namespace std;
int main(int argc,char* argv[]){
    int test=0,nx=200,ny=200;
    g_glm_on=true; g_solver=1;
    if(argc>1) test  =atoi(argv[1]);
    if(argc>2) nx    =atoi(argv[2]);
    if(argc>3) ny    =atoi(argv[3]);
    if(argc>4) g_glm_on=(atoi(argv[4])!=0);
    if(argc>5) g_solver=atoi(argv[5]);   // 0=FORCE, 1=HLLD

    string solver_name = (g_solver==0) ? "FORCE" : "HLLD";
    cout<<"Test "<<test<<", "<<nx<<"x"<<ny
        <<", "<<solver_name<<", GLM="<<(g_glm_on?"ON":"OFF")
        <<", Threads="<<omp_get_max_threads()<<endl;

    double x0,x1,y0,y1,gama,t_end;
    double cfl = (g_solver==0) ? 0.4 : 0.3;
    BC bcx,bcy;
    switch(test){
    case 0: x0=0;x1=1;y0=0;y1=1;gama=1.4;t_end=0.25;
            bcx=TRANSMISSIVE;bcy=TRANSMISSIVE;break;
    case 1: x0=0;x1=800;y0=0;y1=800;gama=2.0;t_end=80.0;
            bcx=TRANSMISSIVE;bcy=TRANSMISSIVE;break;
    case 2: x0=0;x1=800;y0=0;y1=800;gama=2.0;t_end=80.0;
            bcx=TRANSMISSIVE;bcy=TRANSMISSIVE;break;
    case 3: x0=0;x1=1;y0=0;y1=1;gama=5.0/3.0;t_end=0.5;
            bcx=PERIODIC;bcy=PERIODIC;break;
    case 4: x0=0;x1=1;y0=0;y1=1;gama=5.0/3.0;t_end=0.18;
            bcx=TRANSMISSIVE;bcy=TRANSMISSIVE;break;
    default: cerr<<"Unknown test"<<endl; return 1;
    }

    double dx=(x1-x0)/nx, dy=(y1-y0)/ny;
    Grid w(nx+4,vector<Vec>(ny+4,Vec(NVAR,0.0)));

    // ---- Initial conditions (primitive variables) ----
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=2;i<nx+2;i++)
        for(int j=2;j<ny+2;j++){
            double x=x0+(i-1.5)*dx, y=y0+(j-1.5)*dy;
            switch(test){
            case 0:{
                double r=sqrt((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5));
                w[i][j]=(r<=0.4)?Vec{1,0,0,0,1,0,0,0,0}
                                :Vec{0.125,0,0,0,0.1,0,0,0,0};
                break;}
            case 1:{
                double m=0.5*(x0+x1);
                w[i][j]=(x<=m)?Vec{1,0,0,0,1,0.75, 1,0,0}
                               :Vec{0.125,0,0,0,0.1,0.75,-1,0,0};
                break;}
            case 2:{
                double m=0.5*(y0+y1);
                w[i][j]=(y<=m)?Vec{1,0,0,0,1, 1,0.75,0,0}
                               :Vec{0.125,0,0,0,0.1,-1,0.75,0,0};
                break;}
            case 3:{
                // Orszag-Tang vortex (Dahlburg-Picone 1989 variant)
                double rho=gama*gama;
                w[i][j]={rho,-sin(2*M_PI*y),sin(2*M_PI*x),0,gama,
                         -sin(2*M_PI*y),sin(4*M_PI*x),0,0};
                break;}
            case 4:{
                // MHD Rotor (Toth 2000)
                double r0=0.1,r1=0.115,v0=1.0;
                double r=sqrt((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5));
                double ur=(0.5-y)*v0, vr=(x-0.5)*v0;
                double Bx0=2.5/sqrt(4.0*M_PI);
                double rho,vxv,vyv;
                if(r<r0){
                    rho=10; vxv=ur/r0; vyv=vr/r0;
                } else if(r<r1){
                    double fr=(r1-r)/(r1-r0);
                    rho=1+9*fr; vxv=ur*fr/r; vyv=vr*fr/r;
                } else {
                    rho=1; vxv=0; vyv=0;
                }
                w[i][j]={rho,vxv,vyv,0,0.5,Bx0,0,0,0};
                break;}
            }
        }

    double t=0, Lx=x1-x0, Ly=y1-y0;
    int step=0;

    // ---- Timing accumulators ----
    double t_sweepx=0, t_sweepy=0, t_other=0;
    auto T0 = Clock::now();

    while(t<t_end){
        auto ta = Clock::now();
        apply_bc(w,nx,ny,bcx,bcy);
        double dt=compute_dt(w,nx,ny,dx,dy,gama,cfl);
        if(t+dt>t_end) dt=t_end-t;
        if(dt<=0) break;

        // Strang splitting: X(dt/2) -> Y(dt) -> X(dt/2)
        apply_bc(w,nx,ny,bcx,bcy);
        auto tb = Clock::now(); sweep_x(w,nx,ny,dt*0.5,dx,gama);
        auto tc = Clock::now(); apply_bc(w,nx,ny,bcx,bcy);
                                sweep_y(w,nx,ny,dt,    dy,gama);
        auto td = Clock::now(); apply_bc(w,nx,ny,bcx,bcy);
                                sweep_x(w,nx,ny,dt*0.5,dx,gama);
        auto te = Clock::now();
        if(g_glm_on) glm_decay(w,nx,ny,dt,Lx,Ly);
        auto tf = Clock::now();

        t_sweepx += elapsed(tb,tc) + elapsed(td,te);
        t_sweepy += elapsed(tc,td);
        t_other  += elapsed(ta,tb) + elapsed(te,tf);

        t+=dt; step++;

        if(step%1000==0){
            apply_bc(w,nx,ny,bcx,bcy);
            Diagnostics diag=compute_diagnostics(w,nx,ny,dx,dy,gama);
            cout<<"Step "<<step
                <<"  t="<<t
                <<"  dt="<<dt
                <<"  min_rho="<<diag.min_rho
                <<"  min_p="<<diag.min_p
                <<"  max|divB|="<<diag.max_divB
                <<"  max|psi|="<<diag.max_psi
                <<"  max|v|="<<diag.max_v
                <<endl;
            if(diag.min_rho<0 || diag.min_p<0 || !std::isfinite(diag.max_divB)){
                cerr<<"*** FATAL: unphysical state at step "<<step<<endl;
                break;
            }
        }
    }
    double t_total = elapsed(T0, Clock::now());
    double t_sweep = t_sweepx + t_sweepy;

    cout<<"Finished: "<<step<<" steps, t="<<t<<"\n";
    cout<<"\n=== Timing Summary ===\n";
    cout<<std::fixed<<std::setprecision(3);
    cout<<"  Total wall time : "<<t_total    <<" s\n";
    cout<<"  sweep_x (total) : "<<t_sweepx  <<" s  ("<<100*t_sweepx/t_total<<"%%)\n";
    cout<<"  sweep_y (total) : "<<t_sweepy  <<" s  ("<<100*t_sweepy/t_total<<"%%)\n";
    cout<<"  other   (bc/dt) : "<<t_other   <<" s  ("<<100*t_other /t_total<<"%%)\n";
    cout<<"  per step avg    : "<<1000*t_total/step<<" ms/step\n";
    cout<<"  Mcell-steps/s   : "<<1e-6*(long long)step*nx*ny/t_total<<"\n";
    cout<<"======================\n";

    apply_bc(w,nx,ny,bcx,bcy);
    auto divB=compute_divB(w,nx,ny,dx,dy);
    double maxDB=0;
    for(auto& row:divB) for(auto v:row) if(v>maxDB) maxDB=v;
    cout<<"Max |div B| = "<<maxDB<<endl;

    string gs=g_glm_on?"_glm":"_noglm";
    string ss=(g_solver==0)?"_force":"_hlld";
    string fn="test"+to_string(test)+"_"+to_string(nx)+"x"+to_string(ny)+ss+gs+".dat";
    ofstream out(fn);
    out<<nx<<" "<<ny<<" "<<gama<<" "<<(g_glm_on?1:0)<<"\n";
    for(int j=2;j<ny+2;j++)
        for(int i=2;i<nx+2;i++){
            double x=x0+(i-1.5)*dx, y=y0+(j-1.5)*dy;
            const Vec& p=w[i][j];
            double e=p[4]/((gama-1)*p[0]);
            out<<x<<" "<<y;
            for(int k=0;k<NVAR;k++) out<<" "<<p[k];
            out<<" "<<e<<" "<<divB[i-2][j-2]<<"\n";
        }
    out.close();
    cout<<"Output: "<<fn<<endl;
    return 0;
}