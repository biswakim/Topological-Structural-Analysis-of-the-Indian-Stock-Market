#!/usr/bin/env python
"""Step 2 - clean the data, compute returns and correlations, build the networks.

Produces the data-quality audit, the correlation overview figures and the
baseline graph files that the later scripts (and Gephi/Cytoscape) can read.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from nsenet import plotting as P  # noqa: E402
from nsenet.config import REPO_ROOT, load_config  # noqa: E402
from nsenet.network import (  # noqa: E402
    correlation_matrix, critical_correlation, eigen_spectrum, marchenko_pastur_bounds, upper_triangle,
)
from nsenet.nulls import shuffled_correlations  # noqa: E402
from nsenet.pipeline import prepare, save_numbers, save_table, sym  # noqa: E402
from nsenet.preprocess import log_returns, return_summary  # noqa: E402

plt = P.plt


def main() -> None:
    cfg = load_config()
    P.use_style()
    d = prepare(cfg)
    r_star = cfg["network"]["baseline_threshold"]
    r_back = cfg["network"]["backbone_threshold"]
    n, t = d.corr.shape[0], len(d.returns)
    rho = upper_triangle(d.corr)

    # ------------------------------------------------------------------ audit
    save_table(d.audit, cfg, "data_extreme_return_audit", index=False)
    rep = d.clean_report
    cleaning = pd.DataFrame(
        [
            ("Tickers in the Nifty 500 list", rep["tickers_requested"]),
            ("Removed: Yahoo returned no data", len(rep["dropped_no_data"])),
            (f"Removed: price history < {cfg['preprocess']['min_coverage']:.0%} of days (recent listings)",
             len(rep["dropped_low_coverage"])),
            ("Removed: gap at the start of the window", len(rep["dropped_residual_gaps"])),
            ("Stocks retained (network nodes)", rep["tickers_kept"]),
            ("Trading days", rep["trading_days"]),
            ("Daily log returns per stock", rep["n_returns"]),
            ("Price cells forward-filled", rep["cells_forward_filled"]),
            (f"|log return| > {cfg['preprocess']['extreme_return_flag']:.2f} flagged for audit",
             rep["extreme_returns_flagged"]),
            ("Flagged returns classified as mechanical and masked", rep["returns_masked"]),
        ],
        columns=["item", "value"],
    )
    save_table(cleaning, cfg, "data_cleaning_summary", index=False)

    # sector composition before/after cleaning
    from nsenet.data import load_cached
    from nsenet.pipeline import SECTOR_SHORT

    all_members = load_cached(cfg)["members"]
    comp = pd.DataFrame(
        {
            "in_index": all_members["Industry"].map(SECTOR_SHORT).value_counts(),
            "in_network": d.sectors.value_counts(),
            "psu_in_network": d.members.groupby("sector")["psu"].sum(),
        }
    ).fillna(0).astype(int).sort_values("in_network", ascending=False)
    comp.index.name = "sector"
    save_table(comp, cfg, "data_sector_composition")

    summ = return_summary(d.returns)
    save_table(summ, cfg, "data_return_summary", float_format="%.6f")

    # ------------------------------------------------- robustness of the input
    raw_ret = log_returns(d.prices)
    c_raw = correlation_matrix(raw_ret)
    masked = [o["ticker"] for o in cfg["preprocess"]["return_overrides"] if o["ticker"] in d.corr.index]
    mean_before = float(np.mean([c_raw.loc[m].drop(m).mean() for m in masked]))
    mean_after = float(np.mean([d.corr.loc[m].drop(m).mean() for m in masked]))
    spearman = d.returns.rank().corr().to_numpy()
    sp = spearman[np.triu_indices(n, k=1)]
    pearson_spearman = float(np.corrcoef(rho, sp)[0, 1])

    # ------------------------------------------------------ yard-sticks for r
    null = shuffled_correlations(d.returns, cfg["nulls"]["n_shuffles"], cfg["nulls"]["seed"])
    w, v = eigen_spectrum(d.corr)
    mp_lo, mp_hi = marchenko_pastur_bounds(t, n)
    r_bonf = critical_correlation(t, 0.01, n * (n - 1) // 2)
    r_1pct = critical_correlation(t, 0.01)

    quant = np.percentile(rho, [1, 5, 25, 50, 75, 95, 99])
    save_numbers(
        "data",
        {
            "n_requested": rep["tickers_requested"], "n_stocks": n, "n_days": rep["trading_days"], "n_returns": t,
            "first_date": rep["first_date"], "last_date": rep["last_date"],
            "n_no_data": len(rep["dropped_no_data"]), "n_low_cov": len(rep["dropped_low_coverage"]),
            "n_resid_gap": len(rep["dropped_residual_gaps"]), "n_ffill": rep["cells_forward_filled"],
            "n_flagged": rep["extreme_returns_flagged"], "n_masked": rep["returns_masked"],
            "n_pairs": n * (n - 1) // 2, "n_sectors": int(d.sectors.nunique()), "n_psu": int(d.members["psu"].sum()),
            "rho_mean": rho.mean(), "rho_median": np.median(rho), "rho_sd": rho.std(ddof=1),
            "rho_min": rho.min(), "rho_max": rho.max(), "rho_p01": quant[0], "rho_p05": quant[1],
            "rho_p25": quant[2], "rho_p75": quant[4], "rho_p95": quant[5], "rho_p99": quant[6],
            "rho_neg_frac": float((rho < 0).mean()),
            "rho_skew": float(stats.skew(rho)),
            "null_sd": null.std(ddof=1), "null_abs_max": np.abs(null).max(), "null_p999": np.percentile(null, 99.9),
            "r_crit_bonferroni": r_bonf, "r_crit_1pct": r_1pct,
            "frac_sig_bonferroni": float((rho > r_bonf).mean()),
            "lambda1": w[0], "lambda1_share": w[0] / n, "lambda2": w[1], "lambda3": w[2],
            "mp_upper": mp_hi, "mp_lower": mp_lo, "n_eig_above_mp": int((w > mp_hi).sum()), "q_ratio": t / n,
            "v1_min": float(np.abs(v[:, 0]).min()), "v1_all_same_sign": bool((np.sign(v[:, 0]) == np.sign(v[0, 0])).all()),
            "masked_meancorr_before": mean_before, "masked_meancorr_after": mean_after,
            "pearson_spearman_corr": pearson_spearman,
            "mean_ann_vol": float(summ["ann_vol"].mean()), "median_kurtosis": float(summ["excess_kurtosis"].median()),
            "r_star": r_star, "r_back": r_back,
            "yfinance_version": d.meta.get("yfinance_version", "n/a"),
            "downloaded_at": d.meta.get("downloaded_at_utc", "n/a")[:10],
        },
    )
    pairs = pd.DataFrame(
        [(sym(d.corr.index[i]), sym(d.corr.index[j]), d.sectors.iloc[i], d.sectors.iloc[j], d.corr.iat[i, j])
         for i, j in zip(*np.triu_indices(n, k=1)) if d.corr.iat[i, j] > 0.70],
        columns=["stock_a", "stock_b", "sector_a", "sector_b", "rho"],
    ).sort_values("rho", ascending=False)
    save_table(pairs, cfg, "data_top_correlated_pairs", index=False)

    # ------------------------------------------------------------ figure 1
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    ax = axes[0]
    bins = np.linspace(-0.2, 0.95, 116)
    ax.hist(null, bins=bins, density=True, color=P.NEUTRAL, alpha=0.9, label="time-shuffled returns (noise only)")
    ax.hist(rho, bins=bins, density=True, color=P.BLUE, alpha=0.85, label="observed pairs")
    ymax = ax.get_ylim()[1]
    for thr, lab, col, yy in ((r_star, f"$r^*={r_star:.2f}$", P.INK, 0.93), (r_back, f"$r_b={r_back:.2f}$", P.INK2, 0.84)):
        ax.axvline(thr, color=col, lw=1.0, ls="--")
        ax.text(thr + 0.008, ymax * yy, lab, fontsize=8, color=col)
    ax.axvline(r_bonf, color=P.RED, lw=1.0, ls=":")
    ax.text(r_bonf - 0.008, ymax * 0.70, "Bonferroni\n1% bound", fontsize=7, color=P.RED, ha="right")
    ax.set_xlabel(r"Pearson correlation $\rho_{ij}$ of daily log returns")
    ax.set_ylabel("probability density")
    ax.set_title(f"Distribution of the {n*(n-1)//2:,} pairwise correlations")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.62))
    P.panel_label(ax, "(a)")

    ax = axes[1]
    rank = np.arange(1, n + 1)
    ax.loglog(rank, w, "o", ms=3, color=P.BLUE, label="eigenvalues of the correlation matrix")
    ax.axhline(mp_hi, color=P.RED, lw=1.0, ls=":")
    ax.text(n, mp_hi * 1.12, r"Marchenko-Pastur upper edge $\lambda_+$" + f" = {mp_hi:.2f}", color=P.RED, fontsize=7, ha="right")
    ax.annotate(f"market mode\n$\\lambda_1$ = {w[0]:.1f} ({w[0]/n:.0%} of variance)", (1, w[0]), xytext=(2.2, w[0] * 0.55),
                fontsize=8, color=P.INK, arrowprops=dict(arrowstyle="-", color=P.MUTED, lw=0.7))
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel(r"eigenvalue $\lambda$")
    ax.set_title("Eigenvalue spectrum vs. random-matrix bound")
    P.panel_label(ax, "(b)")
    fig.tight_layout()
    P.savefig(fig, cfg, "fig01_correlation_overview")

    # ------------------------------------------------------------ figure 2
    order_sec = d.sectors.value_counts().index.tolist()
    order = sorted(range(n), key=lambda i: (order_sec.index(d.sectors.iloc[i]), -d.corr.iloc[i].mean()))
    c_ord = d.corr.to_numpy()[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    im = ax.imshow(c_ord, cmap="RdBu_r", vmin=-0.8, vmax=0.8, interpolation="nearest", rasterized=True)
    bounds = np.cumsum([int((d.sectors == s).sum()) for s in order_sec])
    starts = np.concatenate([[0], bounds[:-1]])
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="white", lw=0.5)
        ax.axvline(b - 0.5, color="white", lw=0.5)
    ticks = [(s + e) / 2 for s, e in zip(starts, bounds)]
    big = [i for i, s in enumerate(order_sec) if (d.sectors == s).sum() >= 9]
    ax.set_yticks([ticks[i] for i in big])
    ax.set_yticklabels([f"{order_sec[i]} ({(d.sectors == order_sec[i]).sum()})" for i in big], fontsize=7)
    ax.set_xticks([])
    ax.grid(False)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"$\rho_{ij}$", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("Correlation matrix, stocks grouped by NSE sector")
    fig.tight_layout()
    P.savefig(fig, cfg, "fig02_correlation_heatmap")

    # --------------------------------------------------------- graph files
    gdir = REPO_ROOT / "results" / "graphs"
    gdir.mkdir(parents=True, exist_ok=True)
    for thr in (r_star, r_back):
        g = d.graph(thr)
        nx.write_graphml(g, gdir / f"nifty500_r{thr:.2f}.graphml")
        edges = nx.to_pandas_edgelist(g)[["source", "target", "weight"]].sort_values("weight", ascending=False)
        edges.to_csv(gdir / f"nifty500_r{thr:.2f}_edges.csv", index=False, float_format="%.5f")
        print(f"r={thr:.2f}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print(cleaning.to_string(index=False))
    print(f"mean rho={rho.mean():.3f}  lambda1={w[0]:.1f}  r_bonf={r_bonf:.3f}  null sd={null.std():.4f}")


if __name__ == "__main__":
    main()
