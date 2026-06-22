"""
Shared matplotlib style for publication-quality figures (suggestion #6, figures).

Call ``apply_paper_style()`` once before plotting to get consistent, larger fonts
and high-DPI output across every figure in the repo. ``save_figure()`` writes both
a raster PNG (for quick viewing) and a vector PDF (for the manuscript) to the same
basename.

This addresses only the *figure-quality* part of the reviewer's suggestion #6;
mathematical/typographic formatting in the manuscript itself is out of scope.
"""
from __future__ import annotations

import os


def apply_paper_style():
    """Set global matplotlib rcParams for clean, readable, high-DPI figures."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
        "savefig.bbox": "tight",
    })


def save_figure(fig, path_png: str, also_pdf: bool = True):
    """
    Save a figure as PNG and (optionally) a matching vector PDF.

    Args:
        fig: matplotlib Figure.
        path_png: output path ending in .png.
        also_pdf: also write a .pdf alongside (vector, for the manuscript).
    """
    os.makedirs(os.path.dirname(path_png) or ".", exist_ok=True)
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    if also_pdf:
        fig.savefig(os.path.splitext(path_png)[0] + ".pdf", bbox_inches="tight")
