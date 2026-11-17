"""Figure style shared by all scripts.

Conventions
-----------
* One y-axis per panel (no dual axes); two measures -> two panels.
* A sector keeps the same colour in every figure.  The eight largest sectors
  get the eight categorical hues (fixed order, largest first); all smaller
  sectors share one neutral grey, and are identified in tables instead.
* Public- vs private-sector ownership always uses orange vs blue.
* Text is dark grey; grids and axes are recessive.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK = "#1a1a19"       # primary text
INK2 = "#52514e"      # secondary text / annotations
MUTED = "#898781"     # axes, ticks
GRID = "#e1e0d9"
NEUTRAL = "#b9b8b0"   # "other" category and null-model marks

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
HUES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# The eight largest sectors of the cleaned universe, largest first.
SECTOR_ORDER = ["Financials", "Capital Goods", "Healthcare", "Auto", "FMCG", "Chemicals", "IT", "Metals & Mining"]
SECTOR_COLORS = dict(zip(SECTOR_ORDER, HUES))
OTHER_LABEL = "Other sectors"


def use_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 8.5,
            "axes.labelcolor": INK2,
            "axes.edgecolor": NEUTRAL,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.5,
            "text.color": INK,
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax, label: str) -> None:
    """Bold "(a)", "(b)" ... above the top-left corner of a panel."""
    ax.text(-0.08, 1.09, label, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=INK,
            ha="left", va="bottom")


def savefig(fig, cfg: dict, name: str) -> Path:
    """Save <name>.pdf (vector, for the report) and <name>.png (for the README)."""
    out = Path(cfg["output"]["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=cfg["output"].get("dpi", 200))
    plt.close(fig)
    return out / f"{name}.pdf"


def sector_color(sector: str) -> str:
    return SECTOR_COLORS.get(sector, NEUTRAL)


def sector_legend_handles(present=None, size: float = 5) -> list:
    """Legend entries for the coloured sectors that are ``present`` plus one
    grey entry for everything else."""
    present = set(present) if present is not None else None
    handles = [
        Line2D([], [], marker="o", ls="", color=c, ms=size, label=s)
        for s, c in SECTOR_COLORS.items()
        if present is None or s in present
    ]
    if present is None or any(s not in SECTOR_COLORS for s in present):
        handles.append(Line2D([], [], marker="o", ls="", color=NEUTRAL, ms=size, label=OTHER_LABEL))
    return handles


def spring_positions(g: nx.Graph, seed: int = 0, k: float | None = None, iterations: int = 300, weight=None,
                     core_radius: float = 0.62, ring_radius: float = 1.0) -> dict:
    """Reproducible layout for a graph with one large and many small components.

    The largest component gets a Fruchterman-Reingold layout scaled to fit inside
    ``core_radius``; every other component is laid out on its own and placed on
    a ring of radius ``ring_radius`` around it.  A plain spring layout would
    instead squeeze the large component into a dot, because the repulsion
    between disconnected pieces dominates.
    """
    comps = sorted(nx.connected_components(g), key=lambda c: (-len(c), sorted(c)[0]))
    if not comps:
        return {}
    def ordered(nodes) -> nx.Graph:
        # an explicit copy with sorted nodes: a subgraph *view* iterates in set
        # (hash) order, which would make the layout differ from run to run
        h = nx.Graph()
        h.add_nodes_from(sorted(nodes))
        h.add_edges_from(sorted(tuple(sorted(e)) for e in g.subgraph(nodes).edges()))
        return h

    pos: dict = {}
    giant = ordered(comps[0])
    p0 = nx.spring_layout(giant, seed=seed, k=k, iterations=iterations, weight=weight)
    xy = np.array([p0[n] for n in giant])
    xy = xy - xy.mean(axis=0)
    rmax = np.sqrt((xy**2).sum(axis=1)).max()
    xy = xy * (core_radius / rmax if rmax > 0 else 1.0)
    pos.update({n: xy[i] for i, n in enumerate(giant)})
    rest = comps[1:]
    for j, comp in enumerate(rest):
        angle = 2 * np.pi * j / len(rest) + 0.35
        centre = ring_radius * np.array([np.cos(angle), np.sin(angle)])
        sub = ordered(comp)
        ps = nx.spring_layout(sub, seed=seed + j + 1, weight=weight)
        q = np.array([ps[n] for n in sub])
        q = q - q.mean(axis=0)
        qmax = np.sqrt((q**2).sum(axis=1)).max()
        q = q * (0.035 * np.sqrt(len(comp)) / qmax if qmax > 0 else 1.0)
        pos.update({n: centre + q[i] for i, n in enumerate(sub)})
    return pos
