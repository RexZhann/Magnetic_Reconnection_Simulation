"""Readers for every file format mhd2d writes.

- ``*.dat``      ASCII snapshot / final output (write_snapshot_file, write_output_file)
- ``l1.csv``     test-31 campaign L1 diagnostics table
- ``l2_*.f32``   test-31 campaign L2 binary frames
- run folders    resolved from a folder path or from a ctrl deck
"""

from __future__ import annotations

import glob
import os
import re
import subprocess

import numpy as np

# Column order of the ASCII .dat files (after the one-line header).
DAT_COLUMNS = ("x", "y", "rho", "vx", "vy", "vz", "p",
               "Bx", "By", "Bz", "psi", "e", "divB")

# Variable order inside L2 frames / the primitive state vector.
PRIM_VARS = ("rho", "vx", "vy", "vz", "p", "Bx", "By", "Bz", "psi")


# --------------------------------------------------------------------------
# Run folders
# --------------------------------------------------------------------------

def _is_repo(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "Makefile")) and \
        os.path.isfile(os.path.join(path, "src", "solver.cpp"))


def repo_root() -> str:
    """Simulation repository root (holds build/mhd2d). Looked up in order:
    $MHD2D_ROOT; the folder above this package (in-repo / editable install);
    the current directory or any parent of it; else the current directory."""
    env = os.environ.get("MHD2D_ROOT")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _is_repo(here):
        return here
    d = os.getcwd()
    while True:
        if _is_repo(d):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.getcwd()
        d = parent


def resolve_run_dir(path: str, binary: str | None = None) -> str:
    """Return the run folder for ``path``.

    ``path`` may be a run folder, or a ctrl deck (``config/ctrl.txt``), in which
    case the folder is resolved by ``build/mhd2d --print-outdir`` so that
    ``~``/``$VAR`` expansion matches the simulation exactly.
    """
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        exe = binary or os.path.join(repo_root(), "build", "mhd2d")
        if not os.path.exists(exe):
            raise FileNotFoundError(
                f"'{path}' looks like a ctrl deck but {exe} is missing; run `make` "
                "or pass the run folder directly")
        out = subprocess.run([exe, "--print-outdir", path], capture_output=True,
                             text=True, check=True, cwd=repo_root()).stdout.strip()
        return out if os.path.isabs(out) else os.path.join(repo_root(), out)
    raise FileNotFoundError(f"no run folder or ctrl deck at '{path}'")


# --------------------------------------------------------------------------
# ASCII snapshots
# --------------------------------------------------------------------------

def load_snapshot(fname: str) -> dict:
    """Load one ASCII ``.dat`` file.

    Header: ``nx ny gamma glm [t]`` (``t`` only in snapshot files).
    Returns a dict with ``nx, ny, gamma, t`` and one (ny, nx) array per column
    in :data:`DAT_COLUMNS`.
    """
    with open(fname, encoding="utf-8", errors="replace") as f:
        header = f.readline().split()
        data = np.loadtxt(f)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    nx, ny = int(header[0]), int(header[1])
    d = {"nx": nx, "ny": ny, "gamma": float(header[2]),
         "t": float(header[4]) if len(header) >= 5 else 0.0, "file": fname}
    for k, key in enumerate(DAT_COLUMNS):
        d[key] = data[:, k].reshape(ny, nx)
    return d


def snapshot_files(run_dir: str, pattern: str = "*_snap*.dat") -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, pattern)))


def load_snapshots(run_dir: str, pattern: str = "*_snap*.dat",
                   verbose: bool = True) -> list[dict]:
    """Load every snapshot in ``run_dir`` matching ``pattern`` (sorted by name)."""
    files = snapshot_files(run_dir, pattern)
    if not files:
        raise FileNotFoundError(
            f"no files matching '{pattern}' in {run_dir}\n"
            "  (set output_dt > 0 in the ctrl deck to get snapshots)")
    snaps = [load_snapshot(f) for f in files]
    if verbose:
        print(f"  loaded {len(snaps)} snapshots from {run_dir}  "
              f"t = {snaps[0]['t']:.3g} -> {snaps[-1]['t']:.3g}")
    return snaps


def load_final(run_dir: str) -> dict | None:
    """Last snapshot of a run, else its final ``.dat`` output, else None."""
    snaps = snapshot_files(run_dir)
    if snaps:
        return load_snapshot(snaps[-1])
    dats = sorted(glob.glob(os.path.join(run_dir, "*.dat")))
    return load_snapshot(dats[-1]) if dats else None


def times(snaps) -> np.ndarray:
    return np.array([s["t"] for s in snaps])


def nearest(snaps, t: float) -> int:
    """Index of the snapshot closest to time ``t``."""
    return int(np.argmin(np.abs(times(snaps) - t)))


# --------------------------------------------------------------------------
# Campaign L1 / L2 (test 31)
# --------------------------------------------------------------------------

class L1Table:
    """Test-31 ``l1.csv``: ``meta`` (header ``# key = value`` pairs),
    ``cols`` (column names) and ``C`` (name -> array)."""

    def __init__(self, meta: dict, cols: list[str], C: dict):
        self.meta, self.cols, self.C = meta, cols, C

    def __getitem__(self, key):
        return self.C[key]

    def __contains__(self, key):
        return key in self.C

    @property
    def T(self) -> np.ndarray:
        return self.C["t"]

    @property
    def Lx(self) -> float:
        return self.meta.get("Lx", 102.4)

    @property
    def Ly(self) -> float:
        return self.meta.get("Ly", 51.2)


def load_l1(path: str) -> L1Table:
    """Read a campaign ``l1.csv`` (or the run folder that contains it)."""
    if os.path.isdir(path):
        path = os.path.join(path, "l1.csv")
    meta, cols, rows = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                for k, v in re.findall(r"(\w+)\s*=\s*([0-9.eE+-]+)", line):
                    try:
                        meta[k] = float(v)
                    except ValueError:
                        pass
            elif line.startswith("t,"):
                cols = line.strip().split(",")
            elif line.strip():
                rows.append([float(v) for v in line.strip().split(",")])
    if cols is None:
        raise ValueError(f"{path}: no 't,...' column header")
    A = np.array(rows)
    return L1Table(meta, cols, {n: A[:, k] for k, n in enumerate(cols)})


def read_l2(path: str):
    """Read one L2 frame -> ``(data, stride, t)``; ``data`` is (9, ny, nx)
    float32 in :data:`PRIM_VARS` order."""
    with open(path, "rb") as f:
        hdr = np.fromfile(f, "<f8", 6)
        nxo, nyo, stride, t = int(hdr[1]), int(hdr[2]), int(hdr[3]), hdr[4]
        d = np.fromfile(f, "<f4").reshape(9, nyo, nxo)
    return d, stride, t


def l2_files(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "l2_*.f32")))


def l2_time(path: str) -> float:
    return float(re.search(r"l2_t(\d+\.\d)", os.path.basename(path)).group(1))
