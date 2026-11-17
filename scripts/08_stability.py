#!/usr/bin/env python
"""Extension 2 - is the network a property of the market or of this sample?

* moving-block bootstrap over trading days  -> sampling uncertainty of edges,
  of the giant component and of the centrality leaderboard
* first half vs second half of the sample    -> persistence
* removal of the most turbulent days          -> influence of single events
* rolling one-year windows                    -> time variation
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from nsenet import plotting as P  # noqa: E402
from nsenet.config import load_config  # noqa: E402
from nsenet.metrics import sector_mixing, structure_summary  # noqa: E402
from nsenet.network import adjacency, correlation_matrix, upper_triangle  # noqa: E402
from nsenet.nulls import bootstrap_network_stability  # noqa: E402
from nsenet.pipeline import prepare, save_numbers, save_table, sym  # noqa: E402
from nsenet.temporal import rolling_networks, split_half_comparison  # noqa: E402

plt = P.plt


def main() -> None:
    cfg = load_config()
    P.use_style()
    d = prepare(cfg, save=False)
    seed = cfg["nulls"]["seed"]
    r_star, r_back = cfg["network"]["baseline_threshold"], cfg["network"]["backbone_threshold"]
    sectors = d.sectors.to_numpy()
    psu = np.where(d.members["psu"], "PSU", "Private")
    numbers = {}

    # ------------------------------------------------------- bootstrap
    boots = {}
    for r, tag in ((r_star, "star"), (r_back, "back")):
        b = bootstrap_network_stability(d.returns, r, n_boot=500, seed=seed, block=5, top=10)
        boots[r] = b
        lo, hi = np.percentile(b["n_edges"], [2.5, 97.5])
        numbers.update({
            f"{tag}_boot_edges_lo": float(lo), f"{tag}_boot_edges_hi": float(hi),
            f"{tag}_boot_edges_mean": float(b["n_edges"].mean()), f"{tag}_boot_edges_cv": float(b["n_edges"].std() / b["n_edges"].mean()),
            f"{tag}_boot_jaccard_mean": float(b["jaccard"].mean()),
            f"{tag}_boot_jaccard_lo": float(np.percentile(b["jaccard"], 2.5)),
            f"{tag}_boot_jaccard_hi": float(np.percentile(b["jaccard"], 97.5)),
            f"{tag}_boot_gcc_lo": float(np.percentile(b["gcc_fraction"], 2.5)),
            f"{tag}_boot_gcc_hi": float(np.percentile(b["gcc_fraction"], 97.5)),
            f"{tag}_edge_retention_mean": float(b["edge_retention"].mean()),
            f"{tag}_edge_retention_gt90": float((b["edge_retention"] > 0.9).mean()),
            f"{tag}_edge_retention_lt50": float((b["edge_retention"] < 0.5).mean()),
        })
        a0 = adjacency(d.corr, r)
        k0 = pd.Series(a0.sum(axis=1), index=d.corr.index)
        top = pd.DataFrame({"degree_full_sample": k0, "boot_mean_degree": b["degree_mean"], "boot_sd_degree": b["degree_sd"],
                            "freq_in_top10": b["top_frequency"], "sector": d.sectors, "psu": d.members["psu"]})
        top = top.sort_values("freq_in_top10", ascending=False).head(20)
        top.index = [sym(t) for t in top.index]
        save_table(top, cfg, f"E2_bootstrap_top10_frequency_r{r:.2f}")
        full_top10 = [sym(t) for t in k0.sort_values(ascending=False).head(10).index]
        numbers[f"{tag}_top10_min_freq"] = float(top.loc[full_top10, "freq_in_top10"].min())
        numbers[f"{tag}_top10_mean_freq"] = float(top.loc[full_top10, "freq_in_top10"].mean())
        numbers[f"{tag}_n_ever_gt50"] = int((b["top_frequency"] > 0.5).sum())
        numbers[f"{tag}_top1_freq_name"] = top.index[0]
        numbers[f"{tag}_top1_freq"] = float(top["freq_in_top10"].iloc[0])
        print(f"\n=== bootstrap r={r:.2f} ===\n", top.round(3).to_string())

    # ------------------------------------------------------- split half
    for r, tag in ((r_star, "star"), (r_back, "back")):
        sh = split_half_comparison(d.returns, r)
        numbers.update({f"{tag}_split_{k}": v for k, v in sh.items()})
        print(sh)

    # ---------------------------------- influence of the most turbulent days
    mkt = d.market
    order = mkt.abs().sort_values(ascending=False)
    rows = []
    for drop in (0, 1, 5, 10, 20):
        keep = d.returns.drop(index=order.index[:drop])
        c = correlation_matrix(keep)
        row = {"days_removed": drop, "mean_correlation": float(upper_triangle(c).mean())}
        for r, tag in ((r_star, "star"), (r_back, "back")):
            a = adjacency(c, r)
            s = structure_summary(a, with_paths=False)
            a_full = adjacency(d.corr, r)
            iu = np.triu_indices(a.shape[0], k=1)
            row.update({f"edges_{tag}": s["n_edges"], f"gcc_{tag}": s["gcc_size"],
                        f"psu_assort_{tag}": sector_mixing(a, psu)["assortativity"],
                        f"sector_assort_{tag}": sector_mixing(a, sectors)["assortativity"],
                        f"jaccard_{tag}": float((a[iu] & a_full[iu]).sum() / (a[iu] | a_full[iu]).sum()),
                        f"top10_overlap_{tag}": len(set(np.argsort(-a.sum(1), kind="stable")[:10]) &
                                                   set(np.argsort(-a_full.sum(1), kind="stable")[:10]))})
        rows.append(row)
    infl = pd.DataFrame(rows).set_index("days_removed")
    save_table(infl, cfg, "E2_turbulent_day_influence", float_format="%.5f")
    print("\n", infl.round(3).T.to_string())
    worst = order.index[0]
    numbers.update({
        "worst_day": str(worst.date()), "worst_day_return_pct": float(np.expm1(mkt[worst]) * 100),
        "worst_day_psu_return_pct": float(np.expm1(d.returns.loc[worst, d.members["psu"].to_numpy()].mean()) * 100),
        "worst_day_private_return_pct": float(np.expm1(d.returns.loc[worst, ~d.members["psu"].to_numpy()].mean()) * 100),
        "second_worst_day": str(order.index[1].date()), "second_worst_return_pct": float(np.expm1(mkt[order.index[1]]) * 100),
        "top5_days": ", ".join(str(x.date()) for x in order.index[:5]),
    })
    for drop in (1, 5, 10, 20):
        for tag in ("star", "back"):
            numbers[f"drop{drop}_edges_{tag}"] = int(infl.at[drop, f"edges_{tag}"])
            numbers[f"drop{drop}_edge_loss_pct_{tag}"] = float((1 - infl.at[drop, f"edges_{tag}"] / infl.at[0, f"edges_{tag}"]) * 100)
            numbers[f"drop{drop}_psu_assort_{tag}"] = float(infl.at[drop, f"psu_assort_{tag}"])
            numbers[f"drop{drop}_top10_overlap_{tag}"] = int(infl.at[drop, f"top10_overlap_{tag}"])
            numbers[f"drop{drop}_gcc_{tag}"] = int(infl.at[drop, f"gcc_{tag}"])
        numbers[f"drop{drop}_mean_corr"] = float(infl.at[drop, "mean_correlation"])

    # --------------------------------------------------- rolling windows
    t = cfg["temporal"]
    roll = rolling_networks(d.returns, r_star, t["window_days"], t["step_days"], sectors)
    roll["psu_assortativity"] = np.nan
    for end in roll.index:
        chunk = d.returns.loc[roll.at[end, "window_start"]:end]
        a = adjacency(correlation_matrix(chunk), r_star)
        roll.at[end, "psu_assortativity"] = sector_mixing(a, psu)["assortativity"] if a.any() else np.nan
    save_table(roll, cfg, "E2_rolling_windows", float_format="%.5f")
    numbers.update({
        "roll_n": len(roll), "roll_window": t["window_days"], "roll_step": t["step_days"],
        "roll_meancorr_min": float(roll["mean_correlation"].min()), "roll_meancorr_max": float(roll["mean_correlation"].max()),
        "roll_meancorr_min_end": str(roll["mean_correlation"].idxmin().date()),
        "roll_meancorr_max_end": str(roll["mean_correlation"].idxmax().date()),
        "roll_edges_min": int(roll["n_edges"].min()), "roll_edges_max": int(roll["n_edges"].max()),
        "roll_edges_ratio": float(roll["n_edges"].max() / roll["n_edges"].min()),
        "roll_gcc_min": float(roll["gcc_fraction"].min()), "roll_gcc_max": float(roll["gcc_fraction"].max()),
        "roll_mode_min": float(roll["market_mode_share"].min()), "roll_mode_max": float(roll["market_mode_share"].max()),
        "roll_corr_edges_meancorr": float(stats.spearmanr(roll["n_edges"], roll["mean_correlation"])[0]),
        "roll_corr_meancorr_vol": float(stats.spearmanr(roll["mean_correlation"], roll["market_volatility"])[0]),
        "roll_jaccard_mean": float(roll["edge_jaccard_prev"].mean()),
        "roll_first_end": str(roll.index[0].date()), "roll_last_end": str(roll.index[-1].date()),
    })
    has_worst = (roll["window_start"] <= worst) & (roll.index >= worst)
    if has_worst.any() and (~has_worst & (roll.index > worst)).any():
        last_in = roll.index[has_worst][-1]
        first_out = roll.index[~has_worst & (roll.index > worst)][0]
        numbers.update({"roll_last_with_worst": str(last_in.date()), "roll_first_without_worst": str(first_out.date()),
                        "roll_edges_last_with_worst": int(roll.at[last_in, "n_edges"]),
                        "roll_edges_first_without_worst": int(roll.at[first_out, "n_edges"]),
                        "roll_edges_mean_with_worst": float(roll.loc[has_worst, "n_edges"].mean()),
                        "roll_edges_mean_without_worst": float(roll.loc[~has_worst, "n_edges"].mean()),
                        "roll_n_with_worst": int(has_worst.sum())})
    print(roll[["mean_correlation", "market_mode_share", "market_volatility", "n_edges", "gcc_fraction",
                "avg_clustering", "intra_sector_fraction", "psu_assortativity", "edge_jaccard_prev"]].round(3).to_string())

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.6))
    ax = axes[0, 0]
    for r, colr in ((r_star, P.BLUE), (r_back, P.ORANGE)):
        b = boots[r]
        ax.hist(b["n_edges"] / b["n_edges_full"], bins=30, color=colr, alpha=0.8, label=f"$r={r:.2f}$")
    ax.axvline(1, color=P.INK, lw=0.9, ls="--")
    ax.set_xlabel("edges in bootstrap sample / edges in full sample"); ax.set_ylabel("bootstrap replicates")
    ax.set_title("Sampling variability of the edge count"); ax.legend()
    P.panel_label(ax, "(a)")

    ax = axes[0, 1]
    for r, colr in ((r_star, P.BLUE), (r_back, P.ORANGE)):
        ret = np.sort(boots[r]["edge_retention"])[::-1]
        ax.plot(np.linspace(0, 1, len(ret)), ret, color=colr, label=f"$r={r:.2f}$")
    ax.set_xlabel("fraction of full-sample edges (sorted)"); ax.set_ylabel("share of replicates containing the edge")
    ax.set_title("How reliably is each edge recovered?"); ax.legend(loc="lower left")
    P.panel_label(ax, "(b)")

    ax = axes[0, 2]
    x = infl.index.astype(str)
    w = 0.38
    xs = np.arange(len(x))
    ax.bar(xs - w / 2, infl["edges_star"] / infl.at[0, "edges_star"], w, color=P.BLUE, label=f"$r^*={r_star:.2f}$")
    ax.bar(xs + w / 2, infl["edges_back"] / infl.at[0, "edges_back"], w, color=P.ORANGE, label=f"$r_b={r_back:.2f}$")
    ax.set_xticks(xs); ax.set_xticklabels(x)
    ax.set_xlabel("number of most turbulent days removed (of 739)"); ax.set_ylabel("edges relative to full sample")
    ax.set_title("A handful of days carry many edges"); ax.legend(loc="upper right")
    P.panel_label(ax, "(c)")

    dates = roll.index
    ax = axes[1, 0]
    ax.plot(dates, roll["mean_correlation"], color=P.BLUE, label="mean pairwise correlation")
    ax.plot(dates, roll["market_mode_share"], color=P.ORANGE, label=r"market-mode share $\lambda_1/N$")
    ax.set_ylabel("value"); ax.set_title("Collective behaviour (250-day windows)"); ax.legend(loc="upper right", fontsize=7)
    P.panel_label(ax, "(d)")
    ax = axes[1, 1]
    ax.semilogy(dates, roll["n_edges"], color=P.BLUE)
    ax.axhline(int(adjacency(d.corr, r_star).sum() // 2), color=P.INK, lw=0.9, ls="--")
    ax.text(dates[0], int(adjacency(d.corr, r_star).sum() // 2) * 1.08, "full sample", fontsize=7, color=P.INK2)
    ax.set_ylabel(f"edges at $r^*={r_star:.2f}$ (log scale)"); ax.set_title("Network size through time")
    P.panel_label(ax, "(e)")
    ax = axes[1, 2]
    ax.plot(dates, roll["gcc_fraction"], color=P.BLUE, label="giant-component fraction")
    ax.plot(dates, roll["intra_sector_fraction"], color=P.VIOLET, label="same-sector share of edges")
    ax.plot(dates, roll["psu_assortativity"], color=P.ORANGE, label="ownership assortativity")
    ax.set_ylabel("value"); ax.set_title("Structure through time"); ax.legend(loc="center left", fontsize=7)
    P.panel_label(ax, "(f)")
    for ax in axes[1]:
        ax.set_xlabel("last day of the window")
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
    fig.tight_layout()
    P.savefig(fig, cfg, "fig10_stability")
    save_numbers("E2", numbers)
    print(pd.Series(numbers).to_string())


if __name__ == "__main__":
    main()
