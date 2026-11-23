"""visualization_tool: readers, diagnostics and figures for mhd2d output.

Command line (from the repo root):  python -m visualization_tool --help

Library:
    from visualization_tool import io, physics, campaign, plotting
    snaps = io.load_snapshots("output/test11")
    plotting.harris.plot_keyframes(snaps, times=(2, 8, 20), save_path="kf.png")
    print(campaign.summarize_tier("output/test29_campaign/AB1"))

Modules
    io         .dat snapshots, l1.csv, L2 frames, run-folder / ctrl-deck lookup
    physics    Jz, reconnected flux, time series, flux function, ...
    campaign   test-31 rate / island-crossing / plateau rules + tier summary
    plotting   harris, whistler, compare, campaign, checkerboard figures
    sweep      eta_H stability sweep runner + analysis
    recipes    specific annotated figures (e.g. Sym/AB1/AB2 comparison)
"""

from . import campaign, io, physics, plotting  # noqa: F401

__all__ = ["io", "physics", "campaign", "plotting"]
