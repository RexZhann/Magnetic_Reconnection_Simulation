#pragma once
// ponytail: 全局墙钟性能计数器，仅供 profile（阶段 1/3），不参与任何物理。
// t_recon/t_riemann/t_conupd 是各线程时间之和（在并行区内累计），
// 报告时按 sweep 墙钟等比折算；其余计数器在串行调用点累计，即墙钟。
namespace my_project { namespace perf {
inline double t_recon       = 0.0;  // MUSCL 斜率限制 + Hancock 半步
inline double t_riemann     = 0.0;  // HLLD/FORCE 界面通量
inline double t_conupd      = 0.0;  // 守恒更新 + con2pri + pri2con
inline double t_ct_assemble = 0.0;  // CT-Contact 角点 EMF 组装
inline double t_ct_faraday  = 0.0;  // Faraday 面更新 + cell 同步 + EMF 清零
inline double t_hall        = 0.0;  // Hall RK4
inline double t_hyper       = 0.0;  // hyper-resistivity（含子循环内多次调用）
inline double t_subcfl      = 0.0;  // 子循环 CFL 扫描
inline double t_bc          = 0.0;  // apply_bc（主循环内全部调用）
inline double t_dt          = 0.0;  // compute_dt
inline double t_floor       = 0.0;  // apply_floor
inline double t_diag        = 0.0;  // L1/L2/L3 + 200 步诊断
inline long long nsub_calls = 0, nsub_sum = 0;
inline int nsub_max = 0, nsub_capped = 0;
}} // namespace my_project::perf
