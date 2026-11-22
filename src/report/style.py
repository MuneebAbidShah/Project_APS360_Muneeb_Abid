"""One look for every report figure, sized for the ICLR template.

The ICLR text block is 5.5 in wide (single column), so figures are drawn at
that width (or half of it) with 8-9 pt fonts: they stay readable when included
with \\includegraphics[width=\\linewidth]{...}. Each figure is saved twice:
PNG at 300 dpi (for previews / Word) and PDF (vector, best for LaTeX).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL_WIDTH = 5.5    # inches: ICLR \textwidth
HALF_WIDTH = 2.7    # two figures side by side
GREEN, RED = "#1a7f37", "#c62828"
BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "#555555"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save_figure(fig, out_dir, name):
    """Save `fig` as <name>.png (300 dpi) and <name>.pdf. Returns both paths."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return [str(p) for p in paths]
