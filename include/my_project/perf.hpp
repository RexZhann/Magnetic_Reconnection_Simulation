#pragma once
// ponytail: global wall-clock perf counters, profiling only — no physics.
// t_recon/t_riemann/t_conupd are thread-time sums (accumulated inside the
// parallel region), rescaled to sweep wall-clock at report time; the rest
// accumulate at serial call sites, i.e. true wall-clock.
namespace my_project { namespace perf {
inline double t_recon       = 0.0;  // MUSCL limiting + Hancock half-step
inline double t_riemann     = 0.0;  // HLLD/FORCE interface fluxes
inline double t_conupd      = 0.0;  // conservative update + con2pri + pri2con
inline double t_ct_assemble = 0.0;  // CT-Contact corner EMF assembly
inline double t_ct_faraday  = 0.0;  // Faraday face update + cell sync + EMF zero
inline double t_hall        = 0.0;  // Hall RK4
inline double t_hyper       = 0.0;  // hyper-resistivity (incl. sub-cycle calls)
inline double t_subcfl      = 0.0;  // sub-cycle CFL scan
inline double t_bc          = 0.0;  // apply_bc (all main-loop calls)
inline double t_dt          = 0.0;  // compute_dt
inline double t_floor       = 0.0;  // apply_floor
inline double t_diag        = 0.0;  // L1/L2/L3 + 200-step diagnostics
inline long long nsub_calls = 0, nsub_sum = 0;
inline int nsub_max = 0, nsub_capped = 0;
}} // namespace my_project::perf
