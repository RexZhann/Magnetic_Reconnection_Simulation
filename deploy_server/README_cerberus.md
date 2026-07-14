# test29 campaign — cerberus3 部署手册（逐档流水 + 磁盘受限模式）

目标机：cerberus3（ssh yz2019@cerberus3 直登，64 逻辑核，g++ 13.3.0，
无调度系统 → tmux/nohup）。**磁盘硬约束：home quota 剩 ~1.2 GB**（maia:/raid
全组共享，换机不解决）→ 每档跑完 → 回传本机 → 校验 → 手动删服务器副本。
本机 Sym 档已完成（PASS, E=0.0373±0.0021），服务器从 **AB1** 开始。

所有脚本开头带防呆三行（hostname 硬校验 + 负载/磁盘/quota 落 run.log）。

---

## 0. 打包上传（本地执行）

```bash
cd /d/RexZhann
tar czf deploy.tgz --exclude=output --exclude=.git --exclude=code_submission \
    --exclude=code_submission.tar Magnetic_Reconnection_Simulation
ls -lh deploy.tgz     # 应 < 50 MB
scp deploy.tgz yz2019@cerberus3:~/
ssh yz2019@cerberus3
# ---- 以下在服务器 ----
tar xzf deploy.tgz && cd Magnetic_Reconnection_Simulation
chmod +x deploy_server/*.sh
```

## 1. 构建 + 五分钟体检（470 天空转机器，先确认硬件健康）

```bash
bash deploy_server/build.sh          # 裸 g++ -O3 -fopenmp -march=native
free -h && df -h /tmp
python3 -c "import numpy, matplotlib, pandas" \
  || pip install --user -r deploy_server/requirements.txt
```

## 2. 缩尺冒烟 + RESUME 实测（这台机器上必须实测，不沿用本机结论）

```bash
bash deploy_server/smoke_resume.sh
# 期望末行: "PASS: L1 seamless, ... rows to t=20"
# 流程: 64×32 缩域 Sym 跑 t=20，t≈14 处 kill -9，RESUME=1 续跑，
#       校验 L1 时间严格递增无重复且到达 t_end。
```

## 3. 成本校准 + 线程 scaling 扫描（写 OMP_RECOMMENDED）

```bash
bash deploy_server/calibrate_scaling.sh
# 正尺寸 1024×512、t=2 短测 → 外推 = 纯 ×200，无跨尺寸缓存罚猜测。
# 阶梯 8/16/24/32（48 已砍：超线程段证伪）；每档测两遍取较快值；
# 每次计时前打印 uptime（记录邻居负载）；
# 另测 16/24 的 OMP_PROC_BIND=close OMP_PLACES=cores 绑核对照
# （带宽码常有 5-15% 免费收益，双路机防跨 socket 乱跳）。
# 拐点规则: 无绑核阶梯上增核收益 <10% 即停；绑核档更快则一并采纳
# → 写 OMP_RECOMMENDED（+OMP_BIND_RECOMMENDED 标记，run_tier.sh 自动读）。
```

## 4. AB1 正档启动（tmux 内）

```bash
tmux new -s t29
OMP_NUM_THREADS=$(cat OMP_RECOMMENDED) nohup bash deploy_server/run_tier.sh AB1 \
  > ab1.log 2>&1 &
# 脱离: Ctrl-B D;  回来: tmux attach -t t29
# 进度: bash deploy_server/status.sh AB1
# 断点续跑: RESUME=1 OMP_NUM_THREADS=$(cat OMP_RECOMMENDED) \
#           nohup bash deploy_server/run_tier.sh AB1 >> ab1.log 2>&1 &
```

启动时核对 run.log 里的 IC 自检输出：
- `[campaign] tier params ...` 与 CS2008 Table 1 逐项对照
  （AB1: P1=9.5 P2=8 T1=9.5 T2=8 β01=19 β02=4）；
- `[campaign] IC mirror diff` 四项应 ~1e-3 量级（扰动幅度）以下；
- `[campaign] ptot_rms0` 应 ≤ 0.5%（双片标准）。

## 磁盘流水纪律（每档必走）

1. 档完成时 run_tier.sh 自动打印回传命令 + 生成 `<tier>.md5`；
2. 本机执行 scp 回传（tier 目录 + .md5）；
3. **本机校验**：`md5sum -c <tier>.md5`（或 tar tzf 完整性）；
4. 校验通过后**手动**在服务器 `rm -rf output/test29_campaign/<tier>`
   —— 删除刻意不自动化；
5. quota 剩余 < 500 MB 时 run_tier.sh 启动前硬拦截报错。

## 档位阶梯（逐档放行，模式同本机版）

AB1 → AN1 → ABN1 → AB2 → AN2 → ABN2 → AB3 → AN3

每档回传后在**本机**出 summary 与图：

```bash
python output/test29_campaign/analyze_tier.py AB1
python output/test29_campaign/make_stage0_figs.py   # 改 DIR/tier 后复用
```

## 参数表（CS2008 Table 1，2026-07 对照 PDF 逐行核过）

| tier | B01(外) | B02(内) | ρ01 | ρ02 | 备注 |
|---|---|---|---|---|---|
| Sym | 1 | 1 | 1 | 1 | 本机已完成（旧密度约定，见下） |
| AB1 | 1 | 2 | 1 | 1 | 服务器首档 |
| AB2 | 1 | 3 | 1 | 1 | |
| AB3 | 1 | 0.5 | 1 | 1 | |
| AN1 | 1 | 1 | 1 | 2 | |
| AN2 | 1 | 1 | 1 | 3 | |
| AN3 | 1 | 1 | 1 | 0.5 | |
| ABN1 | 2 | 1 | 1 | 2 | |
| ABN2 | 1 | 0.5 | 1 | **4** | ⚠ 旧工作表误记 ρ02=2，已按 PDF（T2=0.594=2.375/4）改 4 |

压强/温度全部由 β_min=4 规则代码生成（P_total=(B_max²/2)(1+β_min)，
Pi = P_total − B0i²/2，Ti = Pi/ρ0i），启动时打印对照。

## AB1 判读要点（summary 模板已在 analyze_tier.py 落实）

- 硬检查：4 项判定（压平衡 0.5% 双片标准 / div-B / floor / A-B <10%），
  **对称性分片报告不设阈值**（已裁决）；
- 平台窗**双窗并报**：主指标"第一个达标窗" + 敏感性"互感前最后达标窗"；
- **X 线漂移 = 本档起的物理信号**：CS2008 [22]——电流片沿入流方向向
  强场侧漂移。几何对应：B02（内侧）更强 ⇒ 两片都向 y=0 靠拢；
  B01（外侧）更强 ⇒ 两片背离 y=0。AB1 是 B02=2 ⇒ 预期两片内漂；
- 入流类诊断在 X 线随动系报告（v − v_X,y；Sym 档该修正为恒等变换）。

## 与 Sym 档的一处已知约定差异（判读时须知）

pilot/本机 Sym 用的是"均匀温度、密度承担压平衡"的实现假设
（当时任务未指明、代码注释已标注；片心 ρ=1.25）。广义 IC 从 AB1 起
按 CS2008 Eq. (4) 原文：**密度独立 tanh 剖面（Sym 参数下均匀 ρ=1），
压平衡全部由 Pi 承担**。两种约定的上游状态完全相同（ρ=1, B=1, p=2），
只有片内部不同。E(Sym) 归一锚点若要严格同约定，可在九档跑完后在
服务器上用新 IC 补跑一个 Sym（约 15-20 h）做一致性检查——是否补跑
由用户在 Stage 2 汇总前裁决。
