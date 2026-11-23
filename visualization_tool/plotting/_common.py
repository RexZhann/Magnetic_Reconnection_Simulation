"""Shared figure helpers."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt


def save(fig, path, dpi=150):
    """Save ``fig`` to ``path`` (creating folders) and close it. With
    ``path=None`` the figure is returned open, for interactive use."""
    if path is None:
        return fig
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")
    return path
