#!/usr/bin/env python
"""Extension 1 - how much of the network is "just the market"?

(i)  The single-index model predicts rho_ij = rho_im * rho_jm.  Thresholding
     that matrix gives the network the market factor alone would generate.
(ii) Regressing the market out of every stock and re-building the network from
     the residuals shows the structure that is left: sectors.
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
from nsenet.config import load_config  # noqa: E402
from nsenet.metrics import (  # noqa: E402
    normalized_mutual_information, sector_mixing, sector_table, structure_summary, threshold_for_edge_count,
    triangle_census,
)
from nsenet.network import (  # noqa: E402
    adjacency, correlation_matrix, market_model, one_factor_correlation, residual_returns, threshold_graph,
    upper_triangle,
)
from nsenet.pipeline import prepare, save_numbers, save_table, sym  # noqa: E402

plt = P.plt


def louvain_labels(a: np.ndarray, seed: int) -> np.ndarray:
    g = nx.from_numpy_array(a.astype(int))
    comms = nx.community.louvain_communities(g, seed=seed)
    lab = np.zeros(a.shape[0], dtype=int)
    for i, c in enumerate(comms):
        lab[list(c)] = i
    return lab


def main() -> None:
    cfg = load_config()
    P.use_style()
    d = prepare(cfg, save=False)
    seed = cfg["nulls"]["seed"]
    r_star, r_back = cfg["network"]["baseline_threshold"], cfg["network"]["backbone_threshold"]
    sectors = d.sectors.to_numpy()
    psu = np.where(d.members["psu"], "PSU", "Private")
    n = len(sectors)
    iu = np.triu_indices(n, k=1)
    numbers = {}

    mm = market_model(d.returns, d.market)
    mm.join(d.members[["Company Name", "sector", "psu"]]).sort_values("rho_m", ascending=False).to_csv(
        Path(cfg["output"]["tables_dir"]) / "E1_market_model.csv", float_format="%.5f")
    numbers["rho_m_mean"] = mm["rho_m"].mean(); numbers["rho_m_min"] = mm["rho_m"].min(); numbers["rho_m_max"] = mm["rho_m"].max()
    numbers["rho_m_psu_mean"] = mm.loc[d.members["psu"], "rho_m"].mean()
    numbers["rho_m_private_mean"] = mm.loc[~d.members["psu"], "rho_m"].mean()
    numbers["r2_mean"] = mm["r2"].mean()
    numbers["rho_m_top"] = sym(mm["rho_m"].idxmax())
    if "nifty500" in d.benchmarks:
        b = d.benchmarks["nifty500"].reindex(d.returns.index)
        ok = b.notna()
        numbers["ew_vs_nifty500_corr"] = float(np.corrcoef(d.market[ok], b[ok])[0, 1])
    if "nifty50" in d.benchmarks:
        b = d.benchmarks["nifty50"].reindex(d.returns.index)
        ok = b.notna()
        numbers["ew_vs_nifty50_corr"] = float(np.corrcoef(d.market[ok], b[ok])[0, 1])
    numbers["market_ann_vol"] = float(d.market.std(ddof=1) * np.sqrt(252))

    # ---------------------------------------- (i) one-factor prediction
    c1 = one_factor_correlation(mm["rho_m"])
    rows = []
    for r, tag in ((r_star, "star"), (r_back, "back")):
        a_obs, a_one = adjacency(d.corr, r), adjacency(c1, r)
        e_obs, e_one = a_obs[iu], a_one[iu]
        both = int((e_obs & e_one).sum())
        k_obs, k_one = a_obs.sum(axis=1), a_one.sum(axis=1)
        only_obs = e_obs & ~e_one
        same_sector = (sectors[iu[0]] == sectors[iu[1]])
        same_psu = (psu[iu[0]] == "PSU") & (psu[iu[1]] == "PSU")
        vals = {
            "edges_obs": int(e_obs.sum()), "edges_one": int(e_one.sum()), "edges_both": both,
            "recall": both / e_obs.sum(), "precision": both / max(e_one.sum(), 1),
            "jaccard": both / (e_obs | e_one).sum(),
            "degree_spearman": float(stats.spearmanr(k_obs, k_one)[0]),
            "excess_edges": int(only_obs.sum()),
            "excess_same_sector": float(same_sector[only_obs].mean()),
            "explained_same_sector": float(same_sector[e_obs & e_one].mean()) if both else np.nan,
            "excess_psu_pair": float(same_psu[only_obs].mean()),
            "one_clustering": structure_summary(a_one, with_paths=False)["avg_clustering_deg2plus"],
            "one_transitivity": structure_summary(a_one, with_paths=False)["transitivity"],
            "one_gcc": structure_summary(a_one, with_paths=False)["gcc_size"],
            "one_sector_assort": sector_mixing(a_one, sectors)["assortativity"],
            "one_psu_assort": sector_mixing(a_one, psu)["assortativity"],
        }
        rows.append({"threshold": r, **vals})
        numbers.update({f"{tag}_of_{k}": v for k, v in vals.items()})
        if r == r_star:
            keep = (k_obs, k_one)
    save_table(pd.DataFrame(rows).set_index("threshold"), cfg, "E1_one_factor_vs_observed", float_format="%.5f")
    resid_err = upper_triangle(d.corr) - upper_triangle(c1)
    numbers["of_resid_mean"] = float(resid_err.mean()); numbers["of_resid_sd"] = float(resid_err.std())
    numbers["of_corr_of_corr"] = float(np.corrcoef(upper_triangle(d.corr), upper_triangle(c1))[0, 1])

    # ------------------------------------------ (ii) residual network
    resid = residual_returns(d.returns, d.market)
    c_res = correlation_matrix(resid)
    rho_res = upper_triangle(c_res)
    numbers.update({"res_mean": rho_res.mean(), "res_sd": rho_res.std(), "res_max": rho_res.max(),
                    "res_p99": float(np.percentile(rho_res, 99)), "res_neg_frac": float((rho_res < 0).mean())})
    comp_rows = []
    for m_edges, tag in ((int(adjacency(d.corr, r_star).sum() // 2), "star"), (int(adjacency(d.corr, r_back).sum() // 2), "back")):
        thr = threshold_for_edge_count(c_res.to_numpy(), m_edges)
        a_res = adjacency(c_res, thr)
        a_raw = adjacency(d.corr, r_star if tag == "star" else r_back)
        for name, a in (("raw", a_raw), ("residual", a_res)):
            s = structure_summary(a)
            mix, mixp = sector_mixing(a, sectors), sector_mixing(a, psu)
            tri = triangle_census(a, sectors)
            lab = louvain_labels(a, seed)
            connected = a.sum(axis=1) > 0
            row = {"network": name, "edges": s["n_edges"], "threshold": (r_star if tag == "star" else r_back) if name == "raw" else thr,
                   "isolated": s["n_isolated"], "gcc": s["gcc_size"], "max_degree": s["max_degree"],
                   "avg_clustering": s["avg_clustering"], "transitivity": s["transitivity"],
                   "avg_path_length": s["avg_path_length"],
                   "intra_sector_fraction": mix["intra_fraction"], "sector_assortativity": mix["assortativity"],
                   "psu_assortativity": mixp["assortativity"], "mono_sector_triangles": tri["mono_fraction"],
                   "louvain_communities": int(len(np.unique(lab[connected]))),
                   "nmi_louvain_vs_sector": normalized_mutual_information(lab[connected], sectors[connected]),
                   "degree_gini": float(_gini(a.sum(axis=1)))}
            comp_rows.append({"match": tag, **row})
            pre = f"{tag}_{'raw' if name == 'raw' else 'res'}"
            numbers.update({f"{pre}_{k}": v for k, v in row.items() if k not in ("network",)})
        if tag == "back":
            a_res_back, thr_back = a_res, thr
        else:
            a_res_star = a_res
    comp = pd.DataFrame(comp_rows)
    save_table(comp, cfg, "E1_raw_vs_residual_network", index=False, float_format="%.5f")
    print(comp.T.to_string())

    # leaderboard and sector cohesion of the residual network (matched to r*)
    k_res = pd.Series(a_res_star.sum(axis=1), index=d.corr.index)
    top = k_res.sort_values(ascending=False).head(10)
    lb = pd.DataFrame({"stock": [sym(t) for t in top.index], "sector": d.sectors[top.index].to_list(),
                       "psu": d.members.loc[top.index, "psu"].to_list(), "degree": top.to_list()})
    save_table(lb, cfg, "E1_residual_top10_degree", index=False)
    numbers["res_top10_sectors"] = int(lb["sector"].nunique())
    numbers["res_top10_psu"] = int(lb["psu"].sum())
    numbers["res_top1"] = lb["stock"].iloc[0]; numbers["res_top1_sector"] = lb["sector"].iloc[0]
    numbers["res_top10_top_sector"] = lb["sector"].value_counts().index[0]
    numbers["res_top10_top_sector_n"] = int(lb["sector"].value_counts().iloc[0])
    k_raw = adjacency(d.corr, r_star).sum(axis=1)
    numbers["res_raw_degree_spearman"] = float(stats.spearmanr(k_raw, k_res)[0])
    st = sector_table(a_res_star, sectors)
    save_table(st, cfg, "E1_residual_sector_cohesion")
    big = st[st["n_stocks"] >= 9].sort_values("internal_density", ascending=False)
    numbers["res_densest_sector"] = big.index[0]; numbers["res_densest_density"] = float(big["internal_density"].iloc[0])
    numbers["res_second_sector"] = big.index[1]; numbers["res_second_density"] = float(big["internal_density"].iloc[1])
    print(lb.to_string()); print(st.round(3).to_string())

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(11.6, 7.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.95, 0.95])
    ax = fig.add_subplot(gs[0, 0])
    k_obs, k_one = keep
    col = np.where(d.members["psu"], P.ORANGE, P.BLUE)
    ax.scatter(k_one, k_obs, s=11, c=col, alpha=0.8, lw=0)
    lim = max(k_one.max(), k_obs.max()) * 1.04
    ax.plot([0, lim], [0, lim], color=P.INK, lw=0.9, ls="--")
    ax.set_xlabel(r"degree predicted by the market factor alone")
    ax.set_ylabel(f"observed degree at $r^*={r_star:.2f}$")
    ax.set_title("(a) One factor predicts the hubs")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=P.ORANGE, ms=4, label="public-sector (PSU)"),
                       Line2D([], [], marker="o", ls="", color=P.BLUE, ms=4, label="private-sector"),
                       Line2D([], [], ls="--", color=P.INK, lw=0.9, label="observed = predicted")],
              loc="lower right", fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-0.35, 0.95, 131)
    ax.hist(upper_triangle(d.corr), bins=bins, density=True, color=P.BLUE, alpha=0.8, label="raw returns")
    ax.hist(rho_res, bins=bins, density=True, color=P.ORANGE, alpha=0.75, label="market-neutral residuals")
    ax.set_xlabel("pairwise correlation"); ax.set_ylabel("probability density")
    ax.set_title("(b) Correlations before / after")
    ax.legend(loc="upper right", fontsize=7)

    ax = fig.add_subplot(gs[:, 1:])
    g_res = threshold_graph(c_res, thr_back, d.sectors, d.names)
    sub = g_res.subgraph([x for x, k_ in g_res.degree() if k_ > 0])
    comps = [c for c in nx.connected_components(sub) if len(c) >= 3]
    sub = sub.subgraph(set().union(*comps))
    pos = P.spring_positions(sub, seed=3, k=0.42, iterations=500, weight=None)
    deg = dict(sub.degree())
    nx.draw_networkx_edges(sub, pos, ax=ax, edgelist=sorted(tuple(sorted(e)) for e in sub.edges()),
                           width=0.35, alpha=0.28, edge_color=P.MUTED)
    nodes = sorted(sub.nodes())
    ax.scatter([pos[x][0] for x in nodes], [pos[x][1] for x in nodes], s=[10 + 4.5 * deg[x] for x in nodes],
               c=[P.sector_color(d.sectors[x]) for x in nodes], edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_axis_off(); ax.set_aspect("equal")
    ax.set_title(f"(c) Market-neutral network: the {int(a_res_back.sum()//2)} strongest residual links")
    ax.legend(handles=P.sector_legend_handles(set(d.sectors[nodes]), size=5), loc="lower left", ncol=3, fontsize=7,
              bbox_to_anchor=(0.0, -0.06), columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout()
    P.savefig(fig, cfg, "fig09_market_mode")
    save_numbers("E1", numbers)
    print(pd.Series(numbers).to_string())


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else 0.0


if __name__ == "__main__":
    main()
