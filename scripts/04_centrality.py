#!/usr/bin/env python
"""Analysis B - top-10 centrality leaderboards and what they mean economically."""
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
from nsenet.metrics import centrality_table, top_k  # noqa: E402
from nsenet.network import market_model  # noqa: E402
from nsenet.pipeline import prepare, save_numbers, save_table, sym  # noqa: E402

plt = P.plt
MEASURES = ["degree_centrality", "betweenness", "eigenvector"]


def leaderboard(table: pd.DataFrame, k: int) -> pd.DataFrame:
    """Side-by-side top-k for the three centralities."""
    cols = {}
    for m, short in zip(MEASURES, ("degree", "betweenness", "eigenvector")):
        top = table.sort_values(m, ascending=False).head(k)
        cols[f"{short}_stock"] = [sym(t) for t in top.index]
        cols[f"{short}_sector"] = top["sector"].to_list()
        cols[f"{short}_psu"] = top["psu"].to_list()
        cols[f"{short}_value"] = top[m].to_list()
    out = pd.DataFrame(cols, index=pd.RangeIndex(1, k + 1, name="rank"))
    return out


def main() -> None:
    cfg = load_config()
    P.use_style()
    d = prepare(cfg, save=False)
    k_top = cfg["centrality"]["top_k"]
    r_star, r_back = cfg["network"]["baseline_threshold"], cfg["network"]["backbone_threshold"]
    mm = market_model(d.returns, d.market)
    numbers = {}
    tables = {}

    for r, tag in ((r_star, "star"), (r_back, "back")):
        g = d.graph(r)
        tab = centrality_table(g)  # nx.degree_centrality / betweenness_centrality / eigenvector_centrality
        tab["psu"] = d.members["psu"]
        tab["rho_market"] = mm["rho_m"]
        tab["beta"] = mm["beta"]
        tab["ann_vol"] = d.returns.std(ddof=1) * np.sqrt(252)
        tab["median_traded_value_cr"] = d.members["median_traded_value"] / 1e7
        tab["market_cap_cr"] = d.members["market_cap"] / 1e7
        for m in MEASURES:
            tab[f"rank_{m}"] = tab[m].rank(ascending=False, method="min").astype(int)
        tables[r] = tab
        save_table(tab.sort_values("degree", ascending=False), cfg, f"B_centrality_all_r{r:.2f}", float_format="%.6f")
        lb = leaderboard(tab, k_top)
        save_table(lb, cfg, f"B_top{k_top}_leaderboard_r{r:.2f}", float_format="%.4f")
        print(f"\n=== r = {r:.2f} ===\n", lb.to_string())

        n, n_psu = len(tab), int(tab["psu"].sum())
        for m, short in zip(MEASURES, ("deg", "btw", "eig")):
            top = tab.sort_values(m, ascending=False).head(k_top)
            x = int(top["psu"].sum())
            numbers[f"{tag}_psu_in_top_{short}"] = x
            numbers[f"{tag}_psu_top_{short}_p"] = float(stats.hypergeom.sf(x - 1, n, n_psu, k_top))
            numbers[f"{tag}_top1_{short}"] = sym(top.index[0])
            numbers[f"{tag}_top1_{short}_value"] = float(top[m].iloc[0])
            numbers[f"{tag}_top_sectors_{short}"] = int(top["sector"].nunique())
        top50 = tab.sort_values("degree", ascending=False).head(50)
        numbers[f"{tag}_psu_in_top50_deg"] = int(top50["psu"].sum())
        numbers[f"{tag}_psu_mean_degree"] = float(tab.loc[tab["psu"], "degree"].mean())
        numbers[f"{tag}_private_mean_degree"] = float(tab.loc[~tab["psu"], "degree"].mean())
        numbers[f"{tag}_psu_degree_mwu_p"] = float(stats.mannwhitneyu(tab.loc[tab["psu"], "degree"],
                                                                      tab.loc[~tab["psu"], "degree"],
                                                                      alternative="greater").pvalue)
        # overlap between the three top-10 lists
        sets = {m: set(tab.sort_values(m, ascending=False).head(k_top).index) for m in MEASURES}
        numbers[f"{tag}_overlap_deg_eig"] = len(sets["degree_centrality"] & sets["eigenvector"])
        numbers[f"{tag}_overlap_deg_btw"] = len(sets["degree_centrality"] & sets["betweenness"])
        numbers[f"{tag}_overlap_all"] = len(set.intersection(*sets.values()))
        # rank correlations among centralities and with stock characteristics
        cols = MEASURES + ["rho_market", "beta", "ann_vol", "median_traded_value_cr"]
        if tab["market_cap_cr"].notna().sum() > 50:
            cols.append("market_cap_cr")
        sp = tab[cols].corr(method="spearman")
        save_table(sp, cfg, f"B_spearman_r{r:.2f}")
        numbers[f"{tag}_sp_deg_eig"] = sp.loc["degree_centrality", "eigenvector"]
        numbers[f"{tag}_sp_deg_btw"] = sp.loc["degree_centrality", "betweenness"]
        numbers[f"{tag}_sp_deg_rhom"] = sp.loc["degree_centrality", "rho_market"]
        numbers[f"{tag}_sp_eig_rhom"] = sp.loc["eigenvector", "rho_market"]
        numbers[f"{tag}_sp_btw_rhom"] = sp.loc["betweenness", "rho_market"]
        numbers[f"{tag}_sp_deg_size"] = sp.loc["degree_centrality", "median_traded_value_cr"]
        numbers[f"{tag}_sp_deg_vol"] = sp.loc["degree_centrality", "ann_vol"]
        numbers[f"{tag}_sp_deg_beta"] = sp.loc["degree_centrality", "beta"]
        if "market_cap_cr" in sp:
            numbers[f"{tag}_sp_deg_mcap"] = sp.loc["degree_centrality", "market_cap_cr"]
        conn_tab = tab[tab["degree"] > 0]
        numbers[f"{tag}_sp_deg_size_connected"] = float(stats.spearmanr(conn_tab["degree"], conn_tab["median_traded_value_cr"])[0])
        numbers[f"{tag}_min_degree_rhom_gt068"] = int(tab.loc[tab["rho_market"] > 0.68, "degree"].min())
        numbers[f"{tag}_median_degree_rhom_lt050"] = float(tab.loc[tab["rho_market"] < 0.50, "degree"].median())
        numbers[f"{tag}_n_rhom_lt050"] = int((tab["rho_market"] < 0.50).sum())
        numbers[f"{tag}_n_btw_zero"] = int((tab["betweenness"] == 0).sum())
        numbers[f"{tag}_n_articulation"] = len(list(nx.articulation_points(g)))
        numbers[f"{tag}_degree_assortativity"] = float(nx.degree_assortativity_coefficient(g))
        numbers[f"{tag}_max_core"] = int(tab["core_number"].max())
        numbers[f"{tag}_n_max_core"] = int((tab["core_number"] == tab["core_number"].max()).sum())

    # ------------------------------------------- what do the bridges bridge?
    g = d.graph(r_star)
    gcc_nodes = max(nx.connected_components(g), key=len)
    rows = []
    for t in tables[r_star].sort_values("betweenness", ascending=False).head(10).index:
        h = g.subgraph(gcc_nodes - {t})
        pieces = sorted(nx.connected_components(h), key=len, reverse=True)
        cut = set().union(*pieces[1:]) if len(pieces) > 1 else set()
        sec = d.sectors[list(cut)].value_counts() if cut else pd.Series(dtype=int)
        rows.append({"stock": sym(t), "degree": int(g.degree(t)), "betweenness": tables[r_star].at[t, "betweenness"],
                     "stocks_cut_off_if_removed": len(cut),
                     "main_sector_cut_off": sec.index[0] if len(sec) else "",
                     "n_in_main_sector": int(sec.iloc[0]) if len(sec) else 0,
                     "neighbours": " ".join(sym(x) for x in sorted(g[t], key=lambda x: -g[t][x]["weight"])[:8])})
    bridges = pd.DataFrame(rows)
    save_table(bridges, cfg, "B_bridges", index=False, float_format="%.5f")
    print(bridges.to_string())
    for name in ("INDHOTEL", "ETERNAL", "NAUKRI"):
        if name in bridges["stock"].values:
            row = bridges.set_index("stock").loc[name]
            numbers[f"bridge_{name}_degree"] = int(row["degree"])
            numbers[f"bridge_{name}_cut"] = int(row["stocks_cut_off_if_removed"])
            numbers[f"bridge_{name}_cut_sector_n"] = int(row["n_in_main_sector"])
    it_nodes = [n for n in gcc_nodes if d.sectors[n] == "IT"]
    numbers["it_in_gcc"] = len(it_nodes)
    numbers["it_total"] = int((d.sectors == "IT").sum())
    numbers["it_external_edges"] = int(sum(1 for u in it_nodes for v in g[u] if d.sectors[v] != "IT"))

    # ------------------------------------------------ index heavyweights
    tab = tables[r_star]
    heavy = [h + ".NS" for h in cfg["centrality"]["heavyweights"] if h + ".NS" in tab.index]
    hw = tab.loc[heavy, ["name", "sector", "degree", "rank_degree_centrality", "betweenness", "rank_betweenness",
                         "eigenvector", "rank_eigenvector", "rho_market", "median_traded_value_cr"]].copy()
    hw["degree_backbone"] = tables[r_back].loc[heavy, "degree"]
    hw["size_rank"] = tab["median_traded_value_cr"].rank(ascending=False).astype(int).loc[heavy]
    hw.index = [sym(t) for t in hw.index]
    save_table(hw, cfg, "B_heavyweights", float_format="%.5f")
    print("\nHeavyweights\n", hw.round(4).to_string())
    numbers["n_heavy"] = len(heavy)
    numbers["heavy_median_rank_deg"] = float(hw["rank_degree_centrality"].median())
    numbers["heavy_best_rank_deg"] = int(hw["rank_degree_centrality"].min())
    numbers["heavy_best_rank_name"] = hw["rank_degree_centrality"].idxmin()
    numbers["heavy_isolated"] = int((hw["degree"] == 0).sum())
    numbers["heavy_in_top50"] = int((hw["rank_degree_centrality"] <= 50).sum())
    numbers["heavy_median_degree"] = float(hw["degree"].median())
    numbers["heavy_mean_rho_market"] = float(hw["rho_market"].mean())
    numbers["all_mean_rho_market"] = float(tab["rho_market"].mean())
    top10 = tab.sort_values("degree", ascending=False).head(10)
    numbers["top10_mean_rho_market"] = float(top10["rho_market"].mean())
    numbers["top10_median_size_rank"] = float(tab["median_traded_value_cr"].rank(ascending=False).loc[top10.index].median())
    for h in ("RELIANCE", "TCS", "HDFCBANK"):
        if h in hw.index:
            numbers[f"{h}_degree"] = int(hw.loc[h, "degree"])
            numbers[f"{h}_rank_deg"] = int(hw.loc[h, "rank_degree_centrality"])
            numbers[f"{h}_rho_market"] = float(hw.loc[h, "rho_market"])
    # stability of the leaderboard between the two thresholds
    t10a = set(tables[r_star].sort_values("degree", ascending=False).head(10).index)
    t10b = set(tables[r_back].sort_values("degree", ascending=False).head(10).index)
    numbers["top10_deg_overlap_thresholds"] = len(t10a & t10b)
    numbers["sp_degree_between_thresholds"] = float(stats.spearmanr(tables[r_star]["degree"], tables[r_back]["degree"])[0])

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    psu = tab["psu"].to_numpy()
    col = np.where(psu, P.ORANGE, P.BLUE)
    lab_set = list(tab.sort_values("betweenness", ascending=False).head(6).index) + heavy[:6]

    show = [h + ".NS" for h in ("RELIANCE", "HDFCBANK", "TCS", "SBIN", "LT", "BHARTIARTL") if h + ".NS" in tab.index]
    off_b = {"RELIANCE": (5, 6), "HDFCBANK": (-44, 10), "TCS": (-4, 8), "SBIN": (-30, 4), "LT": (-16, 5), "BHARTIARTL": (-10, -11)}
    off_c = {"RELIANCE": (-8, 10), "HDFCBANK": (-22, -12), "TCS": (-18, 8), "SBIN": (5, 3), "LT": (5, 3), "BHARTIARTL": (-30, -11)}

    def label_heavy(ax, xcol, offsets):
        for t in show:
            ax.plot(tab.at[t, xcol], tab.at[t, "degree"], "o", ms=5, mfc="none", mec=P.INK, mew=0.9, zorder=5)
            ax.annotate(sym(t), (tab.at[t, xcol], tab.at[t, "degree"]), xytext=offsets[sym(t)], textcoords="offset points",
                        fontsize=6.5, color=P.INK)

    ax = axes[0]
    conn = tab["degree"] > 0
    ax.scatter(tab.loc[conn, "degree"], tab.loc[conn, "betweenness"].clip(lower=2e-6), s=14, c=col[conn.to_numpy()],
               alpha=0.8, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    for rank, t in enumerate(tab.sort_values("betweenness", ascending=False).head(6).index):
        left = tab.at[t, "degree"] < 15
        ax.annotate(sym(t), (tab.at[t, "degree"], tab.at[t, "betweenness"]),
                    xytext=(6, 8 - 9 * (rank % 2)) if left else (-6, 9 - 8 * (rank % 3)),
                    textcoords="offset points", fontsize=6.5, ha="left" if left else "right", color=P.INK)
    ax.set_ylim(top=0.3)
    ax.set_xlabel("degree $k$"); ax.set_ylabel("betweenness centrality")
    ax.set_title("Bridges are not the hubs")
    P.panel_label(ax, "(a)")

    ax = axes[1]
    ax.scatter(tab["rho_market"], tab["degree"], s=14, c=col, alpha=0.8, lw=0)
    label_heavy(ax, "rho_market", off_b)
    ax.set_xlabel(r"correlation with the market return, $\rho_{i,m}$"); ax.set_ylabel(f"degree $k$ at $r^*={r_star:.2f}$")
    ax.set_title("Degree tracks market co-movement")
    P.panel_label(ax, "(b)")

    ax = axes[2]
    ax.scatter(tab["median_traded_value_cr"], tab["degree"], s=14, c=col, alpha=0.8, lw=0)
    ax.set_xscale("log")
    label_heavy(ax, "median_traded_value_cr", off_c)
    ax.set_xlabel("median daily traded value (Rs crore, log scale)"); ax.set_ylabel(f"degree $k$ at $r^*={r_star:.2f}$")
    ax.set_title("The most traded giants are not hubs")
    P.panel_label(ax, "(c)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=P.ORANGE, ms=5, label="public-sector (PSU)"),
               Line2D([], [], marker="o", ls="", color=P.BLUE, ms=5, label="private-sector")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=7)
    fig.tight_layout()
    P.savefig(fig, cfg, "fig04_centrality")
    save_numbers("B", numbers)
    print("\n", pd.Series(numbers).to_string())


if __name__ == "__main__":
    main()
