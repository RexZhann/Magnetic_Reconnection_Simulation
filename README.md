# 2D MHD Reconnection Code

2D finite-volume MHD (HLLD/FORCE, CT or GLM div-B control, Hall + hyper-resistivity),
parallelised with OpenMP.

```
Makefile           single build entry point
config/            control decks (ctrl.txt = physical/run parameters)
scripts/           run_local.sh (laptop), run_slurm.sh (cluster)
src/ include/      solver source
tests/             unit tests (make test)
visualization_tool/  Python library + CLI for reading and plotting results
output/            simulation results (default output_dir)
deploy_server/     legacy cerberus3 campaign scripts (positional CLI)
docs/              reference material
```

## Workflow

1. **Edit `config/ctrl.txt`**: test id, grid, `t_start`/`t_stop`, `B1`/`B2`/`rho1`/`rho2`, etc.
   Every key is documented inline. Set a key to `default` to keep the test's built-in value.
2. **Run** one of the following:

```bash
./scripts/run_local.sh                          # build + run config/ctrl.txt locally
./scripts/run_local.sh config/ctrl_smoke.txt    # quick 64x32 check of a fresh build
sbatch scripts/run_slurm.sh                     # build + run on a Slurm cluster
sbatch scripts/run_slurm.sh config/my_run.txt   # other deck
```

Results go to `<output_dir>/<label>/` along with `ctrl_used.txt`, a copy of the deck used.
`output_dir` can be any path and is created if it doesn't exist:

```
output_dir = 'output/'                 ! relative to the repo root
output_dir = '/home/me/runs/AB1_hires/' ! absolute
output_dir = '~/runs/'                  ! home directory
output_dir = '$SCRATCH/mhd/'            ! environment variables ($VAR or ${VAR})
label      = ''                         ! empty label = write straight into output_dir
```

`./build/mhd2d --print-outdir config/ctrl.txt` prints the resolved folder without running.

## Visualization

`visualization_tool/` reads every output format (`.dat` snapshots, `l1.csv`, L2 `.f32`
frames) and makes the standard figures. Install it into a venv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"     # editable: edits to the code take effect without reinstalling
pip install -e .             # minimal (numpy + matplotlib): enough for summary/tier/rates/stage0
```

This installs the `mhdviz` command (same as `python -m visualization_tool`), which works
from any directory. With the editable install, the tool finds `build/mhd2d` automatically.
With a normal (non-editable) install, run from inside the repo or set
`MHD2D_ROOT=/path/to/repo` (only needed for ctrl-deck arguments and `sweep run`).
Without installing, you can also run `python -m visualization_tool` from the repo root.
`RUN` can be a run folder or a ctrl deck (the deck is resolved to
`<output_dir>/<label>`):

```bash
python -m visualization_tool harris  output/test11 --times 2 8 20 --gif   # keyframes, rate, GIF
python -m visualization_tool harris  config/ctrl.txt                        # same, via the deck
python -m visualization_tool compare hyper=output/test21 hll=output/test23 --out cmp.png
python -m visualization_tool whistler output/test24                          # test-24 damping fit
python -m visualization_tool checkerboard output/test23 --t 5                # Hall-HLL c_w map
python -m visualization_tool summary output/test29_campaign/AB1              # tier verdict + checks
python -m visualization_tool tier    output/test29_campaign/AB1 --t-cut 170  # tier figures + stats
python -m visualization_tool rates   Sym=output/.../Sym AB1=output/.../AB1 --out rates.png
python -m visualization_tool stage0                                          # annotated Sym/AB1/AB2
python -m visualization_tool sweep   run|analyze|all                         # eta_H stability sweep
```

As a library:

```python
from visualization_tool import io, physics, campaign
from visualization_tool.plotting import harris
snaps = io.load_snapshots("output/test11")
ts = physics.time_series(snaps)            # t, max_vy, max_jz, flux, max_Bz, ...
harris.plot_keyframes(snaps, times=(2, 8, 20), save_path="kf.png")
l1 = io.load_l1("output/test29_campaign/AB1"); first, last, _ = campaign.plateau_windows(l1)
```

| module | contents |
|---|---|
| `io` | `.dat` / `l1.csv` / L2 readers, ctrl-deck -> run folder |
| `physics` | Jz, reconnected flux, time series, flux function, sheet width, ... |
| `campaign` | test-31 pre-registered rules (rate, island crossing, plateau) + tier summary |
| `plotting.*` | `harris`, `whistler`, `compare`, `campaign`, `checkerboard` figures |
| `sweep` | eta_H sweep runner (writes ctrl decks) + analysis |
| `recipes` | figures with hand-placed annotations for specific runs |

## Build

```bash
make            # -> build/mhd2d
make test       # unit tests
make run        # build + run config/ctrl.txt   (make run CTRL=config/other.txt)
make clean
```

- `-march=native` is on by default, so compile on the machine that runs the code.
  `run_slurm.sh` compiles inside the job for this reason. Use `make ARCH=` to turn it off.
- macOS with Apple clang needs OpenMP from Homebrew: `brew install libomp`. The Makefile
  finds it automatically. You can also build with GCC: `make CXX=g++-15`.
- `make DEBUG=1` builds with `-O0 -g`.

## Control deck keys

| group | key | meaning |
|---|---|---|
| problem | `output_dir`, `label`, `test` (required), `resume`, `ic_check` | results go to `output_dir/label` (folder created automatically; `~` and `$VAR` expanded); `resume` restarts test 31 from `ckpt.bin` |
| numerics | `nx`, `ny`, `divb` (`CT`/`GLM`/`none`), `solver` (`HLLD`/`FORCE`), `bc_x`, `bc_y` (`periodic`/`transmissive`), `cfl`, `gamma`, `rho_floor`, `p_floor`, `subcycle_nonideal`, `n_subcycle_max` | boundaries default per test (test 31: periodic in x and y); changing them prints a warning; `driven_ez` and test 28 need `bc_y = transmissive` |
| time_control | `t_start`, `t_stop`, `output_dt`, `l1_dt`, `ckpt_dt`, `l2_stride` | `t_start` shifts the clock; the initial condition is unchanged |
| physics | `eta`, `eta_H`, `hall_di`, `hall_stab`, `psi0`, `driven_ez` | |
| profile | `B1`, `B2`, `rho1`, `rho2` | tests 25–28, 30: 1 = y>0 side, 2 = y<0 side. Test 31: 1 = outer sheet (B01, ρ01), 2 = inner sheet (B02, ρ02) |

The environment variables `RESUME=1`, `IC_CHECK=1` and `L2_STRIDE=n` still work. They are
combined with the deck settings.

## Test ids

- 0 = cylindrical explosion, 1/2 = Brio–Wu x/y, 3 = Orszag–Tang, 4 = rotor, 5–10 = Miyoshi & Kusano (2005)
- 11/12 = Harris sheet (resistive / Hall), 13–15 = Alfvén waves, 16–18 = hyper-resistivity k⁴ scaling
- 19–23 = Harris sheet with hyper-resistivity / Hall variants, 24 = whistler dispersion
- 25 = asymmetric reconnection, 26 = driven (boundary Ez), 27 = CS2008 asymmetry scan,
  28 = driven + Dirichlet y boundaries, 29 = symmetric double sheet,
  30 = resistive asymmetry scan, 31 = CS2008 double-sheet campaign (L1/L2/L3 I/O)

## Legacy positional CLI

The old interface still works, so the `deploy_server/` scripts are unchanged:

```bash
./build/mhd2d <test> [nx ny divb solver etaH t_end label psi0 Ez B2 rho1 B01 B02 rho01 rho02]
#   divb: 0=None 1=GLM 2=CT    solver: 0=FORCE 1=HLLD
```
