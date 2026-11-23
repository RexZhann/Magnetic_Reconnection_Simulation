"""Command-line interface:  python -m visualization_tool <command> ...

RUN arguments accept a run folder (output/AB1) or a ctrl deck
(config/ctrl.txt); a deck is resolved to <output_dir>/<label>.
Figures are written into the run folder unless --out is given.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

from . import campaign, io, recipes, sweep  # noqa: E402
from .plotting import checkerboard, compare, harris, whistler  # noqa: E402
from .plotting import campaign as campaign_plots  # noqa: E402


def _name(run_dir):
    return os.path.basename(os.path.normpath(run_dir))


def cmd_harris(a):
    run = io.resolve_run_dir(a.run)
    out = a.out or run
    snaps = io.load_snapshots(run, a.pattern)
    tag = a.tag or _name(run)
    harris.plot_keyframes(snaps, a.times, os.path.join(out, f"{tag}_keyframes.png"))
    harris.plot_rate(snaps, os.path.join(out, f"{tag}_rate.png"))
    if a.gif:
        harris.make_gif(snaps, os.path.join(out, f"{tag}_reconnection.gif"), fps=a.fps)


def cmd_whistler(a):
    run = io.resolve_run_dir(a.run)
    out = a.out or run
    snaps = io.load_snapshots(run, a.pattern)
    kw = dict(di=a.di, k=a.k, a0=a.a0)
    whistler.plot_decay(snaps, os.path.join(out, "whistler_decay.png"), t_early=a.t_early, **kw)
    whistler.plot_signal(snaps, os.path.join(out, "whistler_signal.png"), **kw)


def cmd_compare(a):
    runs = {}
    for item in a.runs:
        label, _, path = item.rpartition("=")
        run = io.resolve_run_dir(path)
        runs[label or _name(run)] = io.load_snapshots(run, a.pattern)
    compare.compare_runs(runs, a.out, t_max=a.t_max)


def cmd_summary(a):
    run = io.resolve_run_dir(a.run)
    print(campaign.summarize_tier(run, a.tier), end="")
    print(f"\n-> {os.path.join(run, f'tier_{a.tier or _name(run)}_summary.txt')}")


def cmd_tier(a):
    run = io.resolve_run_dir(a.run)
    frames = a.frames.split(",") if a.frames else None
    campaign_plots.plot_tier(run, a.tier, t_cut=a.t_cut, frame_times=frames)


def cmd_rates(a):
    tiers = []
    for item in a.runs:
        label, _, path = item.rpartition("=")
        run = io.resolve_run_dir(path)
        tiers.append((label or _name(run), run))
    campaign_plots.plot_rate_comparison(tiers, a.out)


def cmd_stage0(a):
    recipes.stage0_rate_comparison(a.campaign_dir, a.out)


def cmd_checkerboard(a):
    run = io.resolve_run_dir(a.run)
    snaps = io.load_snapshots(run, a.pattern)
    s = snaps[io.nearest(snaps, a.t)]
    print(f"snapshot t={s['t']:.2f}")
    checkerboard.checkerboard_step_test(di=a.di)
    checkerboard.plot_checkerboard(s, a.out or os.path.join(run, "hll_checkerboard_diag.png"),
                                   di=a.di, rho_floor=a.rho_floor)


def cmd_sweep(a):
    if a.action in ("run", "all"):
        sweep.run_sweep(a.dir)
    if a.action in ("analyze", "all"):
        sweep.analyze_sweep(a.dir)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m visualization_tool",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    PAT = "*_snap*.dat"

    s = sub.add_parser("harris", help="keyframes + rate (+ GIF) for a single-sheet run")
    s.add_argument("run"); s.add_argument("--times", type=float, nargs="+", default=[2, 8, 20])
    s.add_argument("--gif", action="store_true"); s.add_argument("--fps", type=int, default=5)
    s.add_argument("--pattern", default=PAT); s.add_argument("--out")
    s.add_argument("--tag", help="file name prefix (default: run folder name)")
    s.set_defaults(func=cmd_harris)

    s = sub.add_parser("whistler", help="test-24 whistler damping fit")
    s.add_argument("run"); s.add_argument("--di", type=float, default=0.1)
    s.add_argument("--k", type=float, default=6.283185307179586)
    s.add_argument("--a0", type=float, default=0.01)
    s.add_argument("--t-early", type=float, default=0.10)
    s.add_argument("--pattern", default=PAT); s.add_argument("--out")
    s.set_defaults(func=cmd_whistler)

    s = sub.add_parser("compare", help="overlay diagnostics of runs: label=RUN ...")
    s.add_argument("runs", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--t-max", type=float); s.add_argument("--pattern", default=PAT)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("summary", help="test-31 tier verdict + hard checks (text)")
    s.add_argument("run"); s.add_argument("--tier")
    s.set_defaults(func=cmd_summary)

    s = sub.add_parser("tier", help="test-31 tier quick-look figures + t_cut statistics")
    s.add_argument("run"); s.add_argument("--tier")
    s.add_argument("--t-cut", type=float, default=170.0)
    s.add_argument("--frames", help="L2 frame times, e.g. 0,209,300,400")
    s.set_defaults(func=cmd_tier)

    s = sub.add_parser("rates", help="rate + island width of several tiers: label=RUN ...")
    s.add_argument("runs", nargs="+"); s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_rates)

    s = sub.add_parser("stage0", help="annotated Sym/AB1/AB2 rate comparison")
    s.add_argument("--campaign-dir", default="output/test29_campaign"); s.add_argument("--out")
    s.set_defaults(func=cmd_stage0)

    s = sub.add_parser("checkerboard", help="Hall-HLL checkerboard diagnostic")
    s.add_argument("run"); s.add_argument("--t", type=float, default=5.0)
    s.add_argument("--di", type=float, default=1.0)
    s.add_argument("--rho-floor", type=float, default=0.1)
    s.add_argument("--pattern", default=PAT); s.add_argument("--out")
    s.set_defaults(func=cmd_checkerboard)

    s = sub.add_parser("sweep", help="eta_H stability sweep (test 21)")
    s.add_argument("action", choices=["run", "analyze", "all"])
    s.add_argument("--dir", default="output/eta_H_sweep")
    s.set_defaults(func=cmd_sweep)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
