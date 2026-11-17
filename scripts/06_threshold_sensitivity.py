#!/usr/bin/env python
"""Analysis D - threshold sensitivity ("what-if" study).

Rebuilds the network for every r on a fine grid and tracks how edges, average
path length, the giant component and several companions respond.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize  # noqa: E402

from nsenet import plotting as P  # noqa: E402
from nsenet.config import load_config  # noqa: E402
from nsenet.metrics import sector_mixing  # noqa: E402
from nsenet.network import adjacency  # noqa: E402
from nsenet.pipeline import prepare, save_numbers, save_table  # noqa: E402
from nsenet.sensitivity import percolation_point, threshold_sweep  # noqa: E402

plt = P.plt


def er_giant_fraction(mean_degree: np.ndarray) -> np.ndarray:
    """Giant-component fraction of an Erdos-Renyi graph: S = 1 - exp(-<k> S)."""
    out = []
    for c in np.atleast_1d(mean_degree):
        out.append(0.0 if c <= 1 else optimize.brentq(lambda s: s - 1 + np.exp(-c * s), 1e-9, 1.0))
    return np.array(out)


def main() -> None:
    cfg = load_config()
    P.use_style()
    d = prepare(cfg, save=False)
    s = cfg["sensitivity"]
    r_star, r_back = cfg["network"]["baseline_threshold"], cfg["network"]["backbone_threshold"]
    grid = np.round(np.arange(s["sweep_min"], s["sweep_max"] + 1e-9, s["sweep_step"]), 4)
    sweep = threshold_sweep(d.corr, grid, d.sectors.to_numpy())
    psu = np.where(d.members["psu"], "PSU", "Private")
    sweep["psu_assortativity"] = [sector_mixing(adjacency(d.corr, r), psu)["assortativity"]
                                  if sweep.at[r, "n_edges"] > 0 else np.nan for r in sweep.index]
    sweep["isolated_fraction"] = sweep["n_isolated"] / sweep["n_nodes"]
    save_table(sweep, cfg, "D_threshold_sweep_fine", float_format="%.5f")

    coarse = sweep.loc[[round(x, 4) for x in s["thresholds_coarse"]],
                       ["n_edges", "density", "mean_degree", "max_degree", "n_isolated", "n_components_nontrivial",
                        "gcc_size", "gcc_fraction", "second_component", "avg_path_length", "diameter",
                        "global_efficiency", "avg_clustering", "transitivity", "intra_sector_fraction",
                        "sector_assortativity", "psu_assortativity"]]
    save_table(coarse, cfg, "D_threshold_sweep_coarse", float_format="%.5f")
    print(coarse.round(3).to_string())

    perc = percolation_point(sweep)
    n = int(sweep["n_nodes"].iloc[0])
    # exponential decay of the edge count over the bulk of the sweep
    seg = sweep.loc[0.30:0.70]
    slope, intercept = np.polyfit(seg.index.values, np.log(seg["n_edges"].values), 1)
    apl_peak_r = float(sweep["avg_path_length"].idxmax())
    connected = sweep.index[sweep["gcc_fraction"] >= 1.0]
    frac90 = sweep.index[sweep["gcc_fraction"] >= 0.9]
    r_c = perc["r_half"]
    numbers = {
        "grid_n": len(grid), "grid_min": float(grid.min()), "grid_max": float(grid.max()), "grid_step": s["sweep_step"],
        "r_half": perc["r_half"], "r_suscept": perc["r_suscept"], "r_second": perc["r_second"],
        "second_peak_size": int(sweep["second_component"].max()),
        "edge_decay_rate": float(-slope), "edge_halving_dr": float(np.log(2) / -slope),
        "edge_factor_per_0p1": float(np.exp(-slope * 0.1)),
        "apl_peak_r": apl_peak_r, "apl_peak": float(sweep["avg_path_length"].max()),
        "diameter_max": int(sweep["diameter"].max()), "diameter_max_r": float(sweep["diameter"].idxmax()),
        "r_fully_connected_max": float(connected.max()) if len(connected) else np.nan,
        "r_gcc90": float(frac90.max()),
        "mean_degree_at_rc": float(sweep.at[r_c, "mean_degree"]),
        "gcc_at_rc": float(sweep.at[r_c, "gcc_fraction"]),
        "er_gcc_at_rc_degree": float(er_giant_fraction(sweep.at[r_c, "mean_degree"])[0]),
        "edges_at_rc": int(sweep.at[r_c, "n_edges"]),
        "last_edge_r": float(sweep.index[sweep["n_edges"] > 0].max()),
        "transitivity_min": float(sweep.loc[0.2:0.7, "transitivity"].min()),
        "transitivity_min_r": float(sweep.loc[0.2:0.7, "transitivity"].idxmin()),
    }
    for r in s["thresholds_coarse"]:
        key = f"r{int(round(r * 100)):02d}"
        row = sweep.loc[round(r, 4)]
        numbers.update({f"{key}_edges": int(row["n_edges"]), f"{key}_gcc": int(row["gcc_size"]),
                        f"{key}_gcc_frac": float(row["gcc_fraction"]), f"{key}_apl": float(row["avg_path_length"]),
                        f"{key}_isolated": int(row["n_isolated"]), f"{key}_density": float(row["density"]),
                        f"{key}_diameter": float(row["diameter"]), f"{key}_clustering": float(row["avg_clustering"]),
                        f"{key}_transitivity": float(row["transitivity"]),
                        f"{key}_intra": float(row.get("intra_sector_fraction", np.nan)),
                        f"{key}_sector_assort": float(row.get("sector_assortativity", np.nan)),
                        f"{key}_psu_assort": float(row["psu_assortativity"]),
                        f"{key}_mean_degree": float(row["mean_degree"])})
    # local sensitivity around the two working thresholds: +-0.02 in r
    for r, tag in ((r_star, "star"), (r_back, "back")):
        lo, hi = sweep.loc[round(r - 0.02, 4)], sweep.loc[round(r + 0.02, 4)]
        numbers[f"{tag}_edges_minus"] = int(lo["n_edges"]); numbers[f"{tag}_edges_plus"] = int(hi["n_edges"])
        numbers[f"{tag}_gcc_minus"] = int(lo["gcc_size"]); numbers[f"{tag}_gcc_plus"] = int(hi["gcc_size"])
        numbers[f"{tag}_edge_change_pct_per_0p02"] = float((hi["n_edges"] / sweep.at[round(r, 4), "n_edges"] - 1) * 100)
    save_numbers("D", numbers)
    print(pd.Series(numbers).to_string())

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.9))
    x = sweep.index.values
    cx = [round(v, 4) for v in s["thresholds_coarse"]]

    def mark(ax, col, color):
        ax.plot(cx, sweep.loc[cx, col], "o", ms=4.5, color=color, zorder=4, mec="white", mew=0.6)

    def vlines(ax):
        ax.axvline(r_c, color=P.RED, lw=0.9, ls=":")
        ax.axvline(r_star, color=P.INK, lw=0.8, ls="--")

    ax = axes[0, 0]
    ax.semilogy(x, sweep["n_edges"].replace(0, np.nan), color=P.BLUE)
    mark(ax, "n_edges", P.BLUE)
    ax.semilogy(seg.index, np.exp(intercept + slope * seg.index.values), color=P.INK, lw=0.9, ls="-.")
    ax.text(0.56, np.exp(intercept + slope * 0.56) * 6, f"$M \\propto e^{{-{-slope:.1f}\\,r}}$", fontsize=8.5, color=P.INK)
    for r in cx:
        ax.annotate(f"{int(sweep.at[r, 'n_edges']):,}", (r, sweep.at[r, "n_edges"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=6.5, color=P.INK2)
    vlines(ax)
    ax.set_ylabel("number of edges $M$ (log scale)"); ax.set_title("Edges")
    P.panel_label(ax, "(a)")

    ax = axes[0, 1]
    ax.plot(x, sweep["gcc_fraction"], color=P.BLUE, label="giant component")
    ax.plot(x, sweep["isolated_fraction"], color=P.ORANGE, label="isolated stocks")
    mark(ax, "gcc_fraction", P.BLUE)
    vlines(ax)
    ax.text(r_c + 0.012, 0.93, f"$r_c\\approx{r_c:.2f}$", color=P.RED, fontsize=8)
    ax.set_ylabel("fraction of the 436 stocks"); ax.set_title("Giant component")
    ax.legend(loc="center left", fontsize=7.5, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    P.panel_label(ax, "(b)")

    ax = axes[0, 2]
    ax.plot(x, sweep["second_component"], color=P.VIOLET, label="2nd-largest component (stocks)")
    ax.plot(x, sweep["susceptibility"], color=P.AQUA, label="mean finite-cluster size")
    vlines(ax)
    ax.set_ylabel("size (number of stocks)"); ax.set_title("Fragmentation signature")
    ax.legend(loc="upper left", fontsize=6.3, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    P.panel_label(ax, "(c)")

    ax = axes[1, 0]
    ax.plot(x, sweep["avg_path_length"], color=P.BLUE, label="average path length")
    ax.plot(x, sweep["diameter"], color=P.NEUTRAL, lw=1.2, label="diameter")
    mark(ax, "avg_path_length", P.BLUE)
    vlines(ax)
    ax.set_ylabel("hops (within the giant component)"); ax.set_title("Path length")
    ax.legend(loc="upper right", fontsize=7, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    P.panel_label(ax, "(d)")

    ax = axes[1, 1]
    ax.plot(x, sweep["avg_clustering"], color=P.BLUE, label="average clustering (all stocks)")
    ax.plot(x, sweep["avg_clustering_deg2plus"], color=P.AQUA, label=r"average clustering ($k\geq2$)")
    ax.plot(x, sweep["transitivity"], color=P.ORANGE, label="transitivity")
    ax.plot(x, sweep["density"], color=P.INK, lw=0.9, ls="--", label="density (random-graph level)")
    vlines(ax)
    ax.set_ylabel("coefficient"); ax.set_title("Clustering")
    ax.legend(loc="upper left", bbox_to_anchor=(0.47, 0.50), fontsize=6.5, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    P.panel_label(ax, "(e)")

    ax = axes[1, 2]
    ax.plot(x, sweep["intra_sector_fraction"], color=P.BLUE, label="same-sector share of edges")
    ax.plot(x, sweep["sector_assortativity"], color=P.VIOLET, label="sector assortativity")
    ax.plot(x, sweep["psu_assortativity"], color=P.ORANGE, label="ownership (PSU) assortativity")
    ax.axhline(float(sweep["intra_sector_fraction"].dropna().iloc[0]) * 0 + sector_mixing(adjacency(d.corr, 0.3), d.sectors.to_numpy())["expected_uniform"],
               color=P.INK, lw=0.9, ls="--")
    ax.text(0.79, 0.105, "same-sector share if edges were random (9.0%)", fontsize=6.5, color=P.INK2, ha="right")
    vlines(ax)
    ax.set_ylabel("coefficient / share"); ax.set_title("Homophily")
    ax.legend(loc="upper left", fontsize=6.5, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    P.panel_label(ax, "(f)")
    for ax in axes.ravel():
        ax.set_xlabel("correlation threshold $r$")
        ax.set_xlim(grid.min(), 0.80)
    fig.tight_layout()
    P.savefig(fig, cfg, "fig07_threshold_sensitivity")

    # --------------------------- observed vs random-graph giant component
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    md = sweep["mean_degree"].values
    ok = md > 0
    ax.semilogx(md[ok], sweep["gcc_fraction"].values[ok], color=P.BLUE, label="NSE correlation network")
    cgrid = np.logspace(-1.3, np.log10(md.max()), 200)
    ax.semilogx(cgrid, er_giant_fraction(cgrid), color=P.INK, ls="--", lw=1.1, label="Erdős–Rényi random graph")
    for r in (0.3, 0.4, 0.5, 0.6):
        ax.annotate(f"$r={r:.1f}$", (sweep.at[r, "mean_degree"], sweep.at[r, "gcc_fraction"]), xytext=(5, -9),
                    textcoords="offset points", fontsize=7, color=P.INK2)
        ax.plot(sweep.at[r, "mean_degree"], sweep.at[r, "gcc_fraction"], "o", ms=4, color=P.BLUE, mec="white", mew=0.6)
    ax.set_xlabel(r"mean degree $\langle k \rangle$ (log scale)"); ax.set_ylabel("giant-component fraction")
    ax.set_title("Connectivity at equal mean degree")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    P.savefig(fig, cfg, "fig08_giant_component_vs_random")


if __name__ == "__main__":
    main()
