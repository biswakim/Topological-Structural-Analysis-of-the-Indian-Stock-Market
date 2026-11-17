#!/usr/bin/env python
"""Analysis C - local clustering and sector homophily (plus ownership homophily)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nsenet import plotting as P  # noqa: E402
from nsenet.config import load_config  # noqa: E402
from nsenet.metrics import (  # noqa: E402
    homophily_permutation_test, sector_density_matrix, sector_mixing, sector_table, triangle_census,
)
from nsenet.network import adjacency  # noqa: E402
from nsenet.nulls import random_graph_comparison  # noqa: E402
from nsenet.pipeline import prepare, save_numbers, save_table, sym  # noqa: E402

plt = P.plt


def draw_network(ax, g: nx.Graph, colors: dict, title: str, label_nodes=(), seed: int = 11) -> None:
    sub = g.subgraph([n for n, k in g.degree() if k > 0])
    pos = P.spring_positions(sub, seed=seed, k=0.55, iterations=400, weight=None)
    deg = dict(sub.degree())
    nx.draw_networkx_edges(sub, pos, ax=ax, edgelist=sorted(tuple(sorted(e)) for e in sub.edges()),
                           width=0.3, alpha=0.18, edge_color=P.MUTED)
    nodes = sorted(sub.nodes(), key=lambda n: (deg[n], n))
    ax.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes], s=[9 + 2.6 * deg[n] for n in nodes],
               c=[colors[n] for n in nodes], edgecolors="white", linewidths=0.4, zorder=3)
    # call-outs: labels sit on a ring around the core, joined to their node by a thin leader
    labs = [n for n in label_nodes if n in pos]
    if labs:
        centre = np.mean([pos[n] for n in labs], axis=0)
        ang = {n: np.arctan2(pos[n][1] - centre[1], pos[n][0] - centre[0]) for n in labs}
        ordered = sorted(labs, key=lambda n: ang[n])
        slots = np.linspace(-np.pi, np.pi, len(ordered), endpoint=False) + np.pi / len(ordered)
        for n, a in zip(ordered, slots):
            target = centre + 0.74 * np.array([np.cos(a), np.sin(a)])
            ax.annotate(sym(n), pos[n], xytext=target, fontsize=6, color=P.INK, ha="center", va="center", zorder=5,
                        arrowprops=dict(arrowstyle="-", color=P.INK2, lw=0.4, shrinkA=1, shrinkB=1),
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")


def main() -> None:
    cfg = load_config()
    P.use_style()
    d = prepare(cfg, save=False)
    seed, n_perm, n_rand = cfg["nulls"]["seed"], cfg["nulls"]["n_permutations"], cfg["nulls"]["n_random_graphs"]
    r_star, r_back = cfg["network"]["baseline_threshold"], cfg["network"]["backbone_threshold"]
    sectors = d.sectors.to_numpy()
    psu = np.where(d.members["psu"], "PSU", "Private")
    numbers = {}
    store = {}

    for r, tag in ((r_star, "star"), (r_back, "back")):
        g = d.graph(r)
        a = adjacency(d.corr, r)
        k = a.sum(axis=1)
        # --- clustering (networkx reference functions) -------------------
        c_local = pd.Series(nx.clustering(g))
        numbers[f"{tag}_avg_clustering"] = nx.average_clustering(g)
        numbers[f"{tag}_avg_clustering_deg2"] = float(c_local[k >= 2].mean())
        numbers[f"{tag}_transitivity"] = nx.transitivity(g)
        numbers[f"{tag}_n_deg2plus"] = int((k >= 2).sum())
        numbers[f"{tag}_n_c_one"] = int(((c_local == 1) & (k >= 2)).sum())
        numbers[f"{tag}_density"] = nx.density(g)
        numbers[f"{tag}_triangles"] = int(sum(nx.triangles(g).values()) // 3)
        # --- against random graphs --------------------------------------
        rg = random_graph_comparison(a, n_graphs=n_rand, seed=seed)
        save_table(rg, cfg, f"C_random_graph_comparison_r{r:.2f}")
        numbers[f"{tag}_er_clustering"] = rg.loc["avg_clustering", "ER G(n,m) mean"]
        numbers[f"{tag}_er_transitivity"] = rg.loc["transitivity", "ER G(n,m) mean"]
        numbers[f"{tag}_cm_clustering"] = rg.loc["avg_clustering", "degree-preserving mean"]
        numbers[f"{tag}_cm_transitivity"] = rg.loc["transitivity", "degree-preserving mean"]
        numbers[f"{tag}_cm_transitivity_z"] = rg.loc["transitivity", "degree-preserving z"]
        numbers[f"{tag}_cm_clustering_z"] = rg.loc["avg_clustering", "degree-preserving z"]
        numbers[f"{tag}_apl"] = rg.loc["avg_path_length", "observed"]
        numbers[f"{tag}_er_apl"] = rg.loc["avg_path_length", "ER G(n,m) mean"]
        numbers[f"{tag}_cm_apl"] = rg.loc["avg_path_length", "degree-preserving mean"]
        numbers[f"{tag}_gcc_frac"] = rg.loc["gcc_fraction", "observed"]
        numbers[f"{tag}_er_gcc_frac"] = rg.loc["gcc_fraction", "ER G(n,m) mean"]
        numbers[f"{tag}_sigma"] = rg.attrs["small_world_sigma"]
        numbers[f"{tag}_c_ratio"] = rg.attrs["C_ratio"]
        numbers[f"{tag}_l_ratio"] = rg.attrs["L_ratio"]
        # --- homophily ----------------------------------------------------
        mix = sector_mixing(a, sectors)
        mix_psu = sector_mixing(a, psu)
        census = triangle_census(a, sectors)
        perm = homophily_permutation_test(a, sectors, n_perm=n_perm, seed=seed)
        census_psu = triangle_census(a, psu)
        st = sector_table(a, sectors)
        st["mean_clustering_deg2plus"] = [float(c_local.to_numpy()[(sectors == s) & (k >= 2)].mean())
                                          if ((sectors == s) & (k >= 2)).any() else np.nan for s in st.index]
        save_table(st, cfg, f"C_sector_cohesion_r{r:.2f}")
        save_table(sector_density_matrix(a, sectors), cfg, f"C_sector_density_matrix_r{r:.2f}")
        for key in ("intra_fraction", "expected_uniform", "expected_degree", "assortativity"):
            numbers[f"{tag}_sector_{key}"] = mix[key]
            numbers[f"{tag}_psu_{key}"] = mix_psu[key]
        numbers[f"{tag}_sector_ratio_uniform"] = mix["intra_fraction"] / mix["expected_uniform"]
        numbers[f"{tag}_tri_total"] = census["triangles"]
        numbers[f"{tag}_tri_mono"] = census["mono"]
        numbers[f"{tag}_tri_two"] = census["two_same"]
        numbers[f"{tag}_tri_mixed"] = census["mixed"]
        numbers[f"{tag}_tri_mono_frac"] = census["mono_fraction"]
        numbers[f"{tag}_tri_mono_null"] = perm["triangles"]["null_mean"]
        numbers[f"{tag}_tri_mono_z"] = perm["triangles"]["z"]
        numbers[f"{tag}_tri_mono_p"] = perm["triangles"]["p_value"]
        numbers[f"{tag}_edge_null"] = perm["edges"]["null_mean"]
        numbers[f"{tag}_edge_z"] = perm["edges"]["z"]
        numbers[f"{tag}_edge_p"] = perm["edges"]["p_value"]
        numbers[f"{tag}_tri_allpsu_frac"] = census_psu["mono_fraction"]
        # share of triangles made only of PSUs
        ap = a[np.ix_(psu == "PSU", psu == "PSU")].astype(float)
        tri_psu = np.trace(ap @ ap @ ap) / 6
        numbers[f"{tag}_tri_psu_only"] = int(round(tri_psu))
        numbers[f"{tag}_tri_psu_only_frac"] = float(tri_psu / census["triangles"])
        numbers[f"{tag}_psu_internal_density"] = float(ap.sum() / (ap.shape[0] * (ap.shape[0] - 1)))
        numbers[f"{tag}_psu_edges_share"] = float(ap.sum() / 2 / mix["n_edges"])
        best = st[st["n_stocks"] >= 9].sort_values("density_ratio", ascending=False)
        numbers[f"{tag}_most_cohesive_sector"] = best.index[0]
        numbers[f"{tag}_most_cohesive_ratio"] = float(best["density_ratio"].iloc[0])
        numbers[f"{tag}_most_cohesive_density"] = float(best["internal_density"].iloc[0])
        dens = st[st["n_stocks"] >= 9].sort_values("internal_density", ascending=False)
        numbers[f"{tag}_densest_sector"] = dens.index[0]
        numbers[f"{tag}_densest_sector_density"] = float(dens["internal_density"].iloc[0])
        numbers[f"{tag}_sparsest_sector"] = dens.index[-1]
        numbers[f"{tag}_sparsest_sector_density"] = float(dens["internal_density"].iloc[-1])
        comps = sorted(nx.connected_components(g), key=len, reverse=True)
        numbers[f"{tag}_psu_in_gcc"] = int(d.members.loc[list(comps[0]), "psu"].sum())
        numbers[f"{tag}_psu_isolated"] = int(d.members.loc[[n for n in g if g.degree(n) == 0], "psu"].sum())
        numbers[f"{tag}_psu_share_of_gcc"] = float(d.members.loc[list(comps[0]), "psu"].mean())
        numbers[f"{tag}_second_comp_size"] = len(comps[1])
        numbers[f"{tag}_second_comp_sector"] = d.sectors[list(comps[1])].value_counts().index[0]
        numbers[f"{tag}_second_comp_members"] = ", ".join(sorted(sym(n) for n in comps[1]))
        numbers[f"{tag}_n_components"] = int(sum(len(c) > 1 for c in comps))
        for sec_name, short in (("Healthcare", "health"), ("FMCG", "fmcg"), ("IT", "it"), ("Financials", "fin")):
            numbers[f"{tag}_{short}_isolated"] = int(st.loc[sec_name, "n_isolated"])
            numbers[f"{tag}_{short}_n"] = int(st.loc[sec_name, "n_stocks"])
            numbers[f"{tag}_{short}_internal_density"] = float(st.loc[sec_name, "internal_density"])
            numbers[f"{tag}_{short}_external_density"] = float(st.loc[sec_name, "external_density"])
        store[r] = dict(g=g, a=a, k=k, c_local=c_local, st=st, rg=rg, perm=perm, census=census)
        print(f"\n=== r={r:.2f} ===")
        print(rg.round(4).to_string())
        print("sigma", rg.attrs)
        print("sector mixing", mix, "\npsu mixing", mix_psu, "\ncensus", census, "\nperm", perm)
        print(st.round(3).to_string())

    # cliques in the backbone
    gb = store[r_back]["g"]
    # several maximum cliques exist and overlap heavily; order them deterministically
    # (size, then number of PSUs, then names) so that the reported one does not depend on hash order
    cliques = sorted((sorted(c) for c in nx.find_cliques(gb)),
                     key=lambda c: (-len(c), -int(sum(d.members.loc[n, "psu"] for n in c)), c))
    biggest = cliques[0]
    numbers["back_max_clique"] = len(biggest)
    numbers["back_n_max_cliques"] = int(sum(len(c) == len(biggest) for c in cliques))
    numbers["back_max_clique_psu"] = int(sum(d.members.loc[n, "psu"] for n in biggest))
    numbers["back_max_clique_members"] = ", ".join(sorted(sym(n) for n in biggest))
    numbers["back_max_clique_sectors"] = int(d.sectors.loc[biggest].nunique())
    clique_rows = [{"size": len(c), "n_psu": int(sum(d.members.loc[n, "psu"] for n in c)),
                    "n_sectors": int(d.sectors.loc[c].nunique()), "members": " ".join(sorted(sym(n) for n in c))}
                   for c in cliques[:15]]
    save_table(pd.DataFrame(clique_rows), cfg, "C_largest_cliques_backbone", index=False)
    print("largest cliques\n", pd.DataFrame(clique_rows).to_string())

    # top local-clustering stocks among well-connected nodes
    s0 = store[r_star]
    cl = pd.DataFrame({"degree": s0["k"], "clustering": s0["c_local"].to_numpy(), "sector": sectors,
                       "psu": d.members["psu"].to_numpy()}, index=d.corr.index)
    save_table(cl.sort_values("clustering", ascending=False), cfg, "C_local_clustering_all")
    numbers["star_ck_spearman"] = float(cl[cl.degree >= 2][["degree", "clustering"]].corr(method="spearman").iloc[0, 1])
    lo = cl[cl.degree >= 50].sort_values("clustering")
    numbers["star_lowest_c_hub"] = sym(lo.index[0])
    numbers["star_lowest_c_hub_value"] = float(lo["clustering"].iloc[0])
    numbers["star_lowest_c_hub_degree"] = int(lo["degree"].iloc[0])
    numbers["star_hub_mean_c"] = float(cl[cl.degree >= 50]["clustering"].mean())
    numbers["star_n_hubs50"] = int((cl.degree >= 50).sum())
    numbers["star_low_mean_c"] = float(cl[(cl.degree >= 2) & (cl.degree < 10)]["clustering"].mean())

    # --------------------------------------------- figure 5: the network
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.6))
    sec_col = {n: P.sector_color(d.sectors[n]) for n in gb.nodes()}
    own_col = {n: (P.ORANGE if d.members.at[n, "psu"] else P.BLUE) for n in gb.nodes()}
    top_lab = [n for n, _ in sorted(gb.degree(), key=lambda x: -x[1])[:12]]
    draw_network(axes[0], gb, sec_col, f"(a) Backbone network, $r_b={r_back:.2f}$, coloured by NSE sector", top_lab)
    present = set(d.sectors[[n for n, k_ in gb.degree() if k_ > 0]])
    axes[0].legend(handles=P.sector_legend_handles(present, size=5), loc="lower left", ncol=3, fontsize=6.5,
                   bbox_to_anchor=(0.0, -0.08), columnspacing=0.8, handletextpad=0.2)
    draw_network(axes[1], gb, own_col, "(b) Same layout, coloured by ownership", top_lab)
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color=P.ORANGE, ms=5, label="public-sector (PSU)"),
                            Line2D([], [], marker="o", ls="", color=P.BLUE, ms=5, label="private-sector")],
                   loc="lower left", fontsize=7, bbox_to_anchor=(0.0, -0.05), ncol=2)
    fig.tight_layout()
    P.savefig(fig, cfg, "fig05_network_backbone")

    # ------------------------------------ figure 6: clustering & homophily
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2))
    ax = axes[0, 0]
    m2 = cl.degree >= 2
    ax.scatter(cl.degree[m2], cl.clustering[m2], s=13, c=[P.sector_color(s) for s in cl.sector[m2]], alpha=0.85, lw=0)
    ax.set_xscale("log")
    ax.axhline(numbers["star_er_clustering"], color=P.INK, lw=1, ls="--")
    ax.text(150, numbers["star_er_clustering"] + 0.02, "random graph (same density)", fontsize=7, color=P.INK2, ha="right")
    ax.axhline(numbers["star_avg_clustering_deg2"], color=P.RED, lw=1, ls=":")
    ax.text(150, numbers["star_avg_clustering_deg2"] + 0.02, r"observed mean ($k\geq 2$)", fontsize=7, color=P.RED, ha="right",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    ax.set_ylim(-0.36, 1.06); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("degree $k$ (log scale)"); ax.set_ylabel("local clustering coefficient $C_i$")
    ax.set_title(f"Local clustering vs. degree, $r^*={r_star:.2f}$")
    ax.legend(handles=P.sector_legend_handles(set(cl.sector[m2]), size=4), fontsize=6.5, ncol=3, loc="lower left")
    P.panel_label(ax, "(a)")

    ax = axes[0, 1]
    dm = sector_density_matrix(s0["a"], sectors)
    keep = s0["st"].index[s0["st"]["n_stocks"] >= 9].tolist()
    dm = dm.loc[keep, keep]
    im = ax.imshow(dm.to_numpy(), cmap="Blues", vmin=0, vmax=np.nanmax(dm.to_numpy()))
    ax.set_xticks(range(len(keep))); ax.set_xticklabels(keep, rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep, fontsize=6.5)
    ax.grid(False)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("edge density between sectors", fontsize=8)
    ax.set_title(f"Sector-to-sector edge density, $r^*={r_star:.2f}$")
    P.panel_label(ax, "(b)")

    ax = axes[1, 0]
    st = s0["st"][s0["st"]["n_stocks"] >= 9].sort_values("internal_density")
    y = np.arange(len(st))
    ax.hlines(y, st["external_density"], st["internal_density"], color=P.GRID, lw=2.5)
    ax.plot(st["external_density"], y, "o", color=P.NEUTRAL, ms=6, label="density of links to other sectors")
    ax.plot(st["internal_density"], y, "o", color=P.BLUE, ms=6, label="density of links within the sector")
    ax.set_yticks(y); ax.set_yticklabels([f"{s} ({n})" for s, n in zip(st.index, st["n_stocks"])], fontsize=7)
    ax.set_xlabel("edge density"); ax.set_title(f"Within- vs. between-sector density, $r^*={r_star:.2f}$")
    ax.legend(loc="lower right", fontsize=7)
    P.panel_label(ax, "(c)")

    ax = axes[1, 1]
    labels = [f"$r^*={r_star:.2f}$\nedges", f"$r^*={r_star:.2f}$\ntriangles", f"$r_b={r_back:.2f}$\nedges",
              f"$r_b={r_back:.2f}$\ntriangles"]
    obs = [store[r_star]["perm"]["edges"]["observed"], store[r_star]["perm"]["triangles"]["observed"],
           store[r_back]["perm"]["edges"]["observed"], store[r_back]["perm"]["triangles"]["observed"]]
    nul = [store[r_star]["perm"]["edges"]["null_mean"], store[r_star]["perm"]["triangles"]["null_mean"],
           store[r_back]["perm"]["edges"]["null_mean"], store[r_back]["perm"]["triangles"]["null_mean"]]
    x = np.arange(4)
    ax.bar(x - 0.19, obs, 0.36, color=P.BLUE, label="observed")
    ax.bar(x + 0.19, nul, 0.36, color=P.NEUTRAL, label="sector labels shuffled (null mean)")
    for xi, o, nl in zip(x, obs, nul):
        ax.text(xi - 0.19, o + 0.008, f"{o:.1%}", ha="center", fontsize=7, color=P.INK)
        ax.text(xi + 0.19, nl + 0.008, f"{nl:.1%}", ha="center", fontsize=7, color=P.INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("share inside a single sector")
    ax.set_title("Same-sector edges and triangles vs. label-shuffled null")
    ax.legend(loc="upper left", fontsize=7)
    P.panel_label(ax, "(d)")
    fig.tight_layout()
    P.savefig(fig, cfg, "fig06_clustering_homophily")

    save_numbers("C", numbers)
    print(pd.Series(numbers).to_string())


if __name__ == "__main__":
    main()
