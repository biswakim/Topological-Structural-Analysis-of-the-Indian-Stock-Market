"""Network measurements.

Two families of functions live here:

* ``networkx``-based functions used for the baseline network (they are the
  reference implementations named in the assignment), and
* fast ``numpy``/``scipy`` functions that work directly on a boolean adjacency
  matrix.  They are used wherever the same quantity has to be recomputed
  hundreds of times (threshold sweeps, bootstraps, permutation tests).

``tests/test_metrics.py`` checks that the two families agree.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


# --------------------------------------------------------------------------- #
# A. Degree
# --------------------------------------------------------------------------- #
def degree_series(g: nx.Graph) -> pd.Series:
    return pd.Series(dict(nx.degree(g)), name="degree").astype(int)


def degree_distribution(degrees: np.ndarray | pd.Series) -> pd.DataFrame:
    """Empirical P(k) and complementary CDF P(K >= k) for every observed k."""
    k = np.asarray(degrees, dtype=int)
    values, counts = np.unique(k, return_counts=True)
    pk = counts / counts.sum()
    ccdf = pk[::-1].cumsum()[::-1]
    return pd.DataFrame({"k": values, "count": counts, "pk": pk, "ccdf": ccdf})


def log_binned_pdf(degrees: np.ndarray, bins_per_decade: int = 8) -> pd.DataFrame:
    """Logarithmically binned degree density for k >= 1 (reduces tail noise)."""
    k = np.asarray(degrees, dtype=float)
    k = k[k >= 1]
    if len(k) == 0:
        return pd.DataFrame(columns=["k_mid", "density"])
    n_bins = max(int(np.ceil(np.log10(k.max() + 1) * bins_per_decade)), 1)
    edges = np.unique(np.floor(np.logspace(0, np.log10(k.max() + 1), n_bins + 1)).astype(int))
    edges = np.append(edges[edges <= k.max()], k.max() + 1)
    edges = np.unique(edges)
    counts, _ = np.histogram(k, bins=edges)
    width = np.diff(edges)
    mid = np.sqrt(edges[:-1] * (edges[1:] - 1).clip(min=1))
    dens = counts / (width * len(degrees))
    keep = counts > 0
    return pd.DataFrame({"k_mid": mid[keep], "density": dens[keep], "count": counts[keep]})


# --------------------------------------------------------------------------- #
# B. Centrality
# --------------------------------------------------------------------------- #
def centrality_table(g: nx.Graph) -> pd.DataFrame:
    """Degree, betweenness and eigenvector centrality (plus a few companions).

    Notes on conventions
    --------------------
    * Degree centrality  = k_i / (N - 1).
    * Betweenness is the normalised, unweighted shortest-path betweenness.
    * Eigenvector centrality is the leading eigenvector of the adjacency
      matrix.  On a disconnected graph it is supported on the component with
      the largest spectral radius (in practice the giant component); nodes in
      other components receive 0, which is the standard behaviour.
    """
    n = g.number_of_nodes()
    deg = dict(nx.degree(g))
    table = pd.DataFrame(index=list(g.nodes()))
    table["degree"] = pd.Series(deg)
    table["degree_centrality"] = pd.Series(nx.degree_centrality(g))
    table["betweenness"] = pd.Series(nx.betweenness_centrality(g, normalized=True))
    try:
        eig = nx.eigenvector_centrality(g, max_iter=5000, tol=1e-9)
    except nx.PowerIterationFailedConvergence:
        eig = nx.eigenvector_centrality_numpy(g)
    table["eigenvector"] = pd.Series(eig)
    table["closeness"] = pd.Series(nx.closeness_centrality(g))
    table["clustering"] = pd.Series(nx.clustering(g))
    table["core_number"] = pd.Series(nx.core_number(g))
    table["strength"] = pd.Series(dict(g.degree(weight="weight")))
    if nx.get_node_attributes(g, "sector"):
        table["sector"] = pd.Series(nx.get_node_attributes(g, "sector"))
    if nx.get_node_attributes(g, "name"):
        table["name"] = pd.Series(nx.get_node_attributes(g, "name"))
    table.index.name = "Ticker"
    assert len(table) == n
    return table


def top_k(table: pd.DataFrame, column: str, k: int = 10) -> pd.DataFrame:
    cols = [c for c in ("name", "sector", "degree", column) if c in table.columns]
    cols = list(dict.fromkeys(cols))
    out = table.sort_values(column, ascending=False).head(k)[cols].copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


# --------------------------------------------------------------------------- #
# Fast structural summary from an adjacency matrix
# --------------------------------------------------------------------------- #
def local_clustering(a: np.ndarray) -> np.ndarray:
    """Local clustering coefficient of every node (0 when degree < 2)."""
    af = a.astype(np.float64)
    k = af.sum(axis=1)
    tri2 = ((af @ af) * af).sum(axis=1)  # = 2 x (triangles through node i)
    denom = k * (k - 1)
    return np.divide(tri2, denom, out=np.zeros_like(tri2), where=denom > 0)


def transitivity(a: np.ndarray) -> float:
    """Global clustering = 3 x triangles / connected triples."""
    af = a.astype(np.float64)
    k = af.sum(axis=1)
    closed = ((af @ af) * af).sum()
    triples = (k * (k - 1)).sum()
    return float(closed / triples) if triples > 0 else 0.0


def component_sizes(a: np.ndarray) -> np.ndarray:
    """Sizes of connected components, largest first (isolated nodes count 1)."""
    _, lab = connected_components(csr_matrix(a), directed=False)
    return np.sort(np.bincount(lab))[::-1]


def giant_component_mask(a: np.ndarray) -> np.ndarray:
    _, lab = connected_components(csr_matrix(a), directed=False)
    return lab == np.bincount(lab).argmax()


def path_statistics(a: np.ndarray) -> dict:
    """Average shortest-path length and diameter of the giant component, and
    the global efficiency of the whole graph (mean of 1/d over all pairs,
    with 1/inf = 0, so it stays well defined when the graph fragments)."""
    n = a.shape[0]
    out = {"avg_path_length": np.nan, "diameter": np.nan, "global_efficiency": 0.0}
    if not a.any() or n < 2:
        return out
    d_all = shortest_path(csr_matrix(a), method="D", unweighted=True, directed=False)
    iu = np.triu_indices(n, k=1)
    with np.errstate(divide="ignore"):
        out["global_efficiency"] = float((1.0 / d_all[iu]).mean())
    mask = giant_component_mask(a)
    ng = int(mask.sum())
    if ng >= 2:
        d = d_all[np.ix_(mask, mask)]
        ig = np.triu_indices(ng, k=1)
        out["avg_path_length"] = float(d[ig].mean())
        out["diameter"] = int(d[ig].max())
    return out


def structure_summary(a: np.ndarray, with_paths: bool = True) -> dict:
    """All scalar descriptors of one threshold network."""
    n = a.shape[0]
    k = a.sum(axis=1)
    m = int(k.sum() // 2)
    sizes = component_sizes(a)
    c_loc = local_clustering(a)
    out = {
        "n_nodes": n,
        "n_edges": m,
        "density": 2 * m / (n * (n - 1)),
        "mean_degree": float(k.mean()),
        "max_degree": int(k.max()),
        "n_isolated": int((k == 0).sum()),
        "n_components_nontrivial": int((sizes >= 2).sum()),
        "gcc_size": int(sizes[0]),
        "gcc_fraction": float(sizes[0] / n),
        "second_component": int(sizes[1]) if len(sizes) > 1 else 0,
        "avg_clustering": float(c_loc.mean()),
        "avg_clustering_deg2plus": float(c_loc[k >= 2].mean()) if (k >= 2).any() else np.nan,
        "transitivity": transitivity(a),
    }
    # mean size of the *finite* (non-giant) clusters: the percolation "susceptibility"
    rest = sizes[1:].astype(float)
    out["susceptibility"] = float((rest**2).sum() / rest.sum()) if rest.sum() > 0 else 0.0
    if with_paths:
        out.update(path_statistics(a))
    return out


# --------------------------------------------------------------------------- #
# C. Sector homophily
# --------------------------------------------------------------------------- #
def _codes(labels) -> tuple[np.ndarray, list]:
    cat = pd.Categorical(list(labels))
    return cat.codes.astype(int), list(cat.categories)


def sector_mixing(a: np.ndarray, labels) -> dict:
    """Edge-level homophily statistics.

    intra_fraction      share of edges joining two stocks of the same sector
    expected_uniform    share expected if edges were placed uniformly at
                        random over all pairs (depends only on sector sizes)
    expected_degree     share expected under degree-preserving random mixing
                        (sum_s a_s^2, Newman 2003)
    assortativity       Newman's attribute assortativity coefficient r
    """
    codes, cats = _codes(labels)
    n, s = len(codes), len(cats)
    onehot = np.zeros((n, s))
    onehot[np.arange(n), codes] = 1.0
    e = onehot.T @ a.astype(float) @ onehot  # e[s,t] = edge-ends between s and t
    total = e.sum()
    if total == 0:
        return {"n_edges": 0, "intra_fraction": np.nan, "expected_uniform": np.nan,
                "expected_degree": np.nan, "assortativity": np.nan}
    e_norm = e / total
    a_s = e_norm.sum(axis=1)
    sizes = onehot.sum(axis=0)
    same_pairs = (sizes * (sizes - 1) / 2).sum()
    all_pairs = n * (n - 1) / 2
    tr, sq = np.trace(e_norm), (a_s**2).sum()
    return {
        "n_edges": int(total // 2),
        "intra_fraction": float(tr),
        "expected_uniform": float(same_pairs / all_pairs),
        "expected_degree": float(sq),
        "assortativity": float((tr - sq) / (1 - sq)) if sq < 1 else np.nan,
    }


def sector_density_matrix(a: np.ndarray, labels) -> pd.DataFrame:
    """S x S matrix of realised edge density between (and within) sectors."""
    codes, cats = _codes(labels)
    n, s = len(codes), len(cats)
    onehot = np.zeros((n, s))
    onehot[np.arange(n), codes] = 1.0
    e = onehot.T @ a.astype(float) @ onehot
    sizes = onehot.sum(axis=0)
    possible = np.outer(sizes, sizes)
    np.fill_diagonal(possible, sizes * (sizes - 1))  # diagonal of e counts each edge twice
    dens = np.divide(e, possible, out=np.full_like(e, np.nan), where=possible > 0)
    return pd.DataFrame(dens, index=cats, columns=cats)


def sector_table(a: np.ndarray, labels) -> pd.DataFrame:
    """Per-sector cohesion: internal vs external density, E-I index, clustering."""
    codes, cats = _codes(labels)
    k = a.sum(axis=1)
    c_loc = local_clustering(a)
    rows = []
    n = len(codes)
    for s, name in enumerate(cats):
        inside = codes == s
        ns = int(inside.sum())
        internal = int(a[np.ix_(inside, inside)].sum() // 2)
        external = int(a[np.ix_(inside, ~inside)].sum())
        poss_int = ns * (ns - 1) / 2
        poss_ext = ns * (n - ns)
        rows.append(
            {
                "sector": name,
                "n_stocks": ns,
                "internal_edges": internal,
                "external_edges": external,
                "internal_density": internal / poss_int if poss_int else np.nan,
                "external_density": external / poss_ext if poss_ext else np.nan,
                "ei_index": (external - internal) / (external + internal) if (external + internal) else np.nan,
                "mean_degree": float(k[inside].mean()),
                "mean_clustering": float(c_loc[inside].mean()),
                "n_isolated": int((k[inside] == 0).sum()),
            }
        )
    out = pd.DataFrame(rows).set_index("sector")
    out["density_ratio"] = out["internal_density"] / out["external_density"]
    return out.sort_values("n_stocks", ascending=False)


def triangle_census(a: np.ndarray, labels) -> dict:
    """Classify every triangle by the sector composition of its corners.

    mono      all three stocks in the same sector
    two_same  exactly two stocks share a sector
    mixed     three different sectors
    """
    codes, cats = _codes(labels)
    af = a.astype(np.float64)
    total = np.trace(af @ af @ af) / 6.0
    mono = 0.0
    two_same = 0.0
    for s in range(len(cats)):
        inside = codes == s
        if inside.sum() < 2:
            continue
        a_ss = af[np.ix_(inside, inside)]
        a_so = af[np.ix_(inside, ~inside)]
        mono += np.trace(a_ss @ a_ss @ a_ss) / 6.0
        two_same += (a_ss * (a_so @ a_so.T)).sum() / 2.0
    mixed = total - mono - two_same
    return {
        "triangles": int(round(total)),
        "mono": int(round(mono)),
        "two_same": int(round(two_same)),
        "mixed": int(round(mixed)),
        "mono_fraction": float(mono / total) if total > 0 else np.nan,
    }


def homophily_permutation_test(a: np.ndarray, labels, n_perm: int = 2000, seed: int = 0) -> dict:
    """Shuffle sector labels over the fixed network.

    Keeps the topology and the sector sizes, destroys only the link between
    the two.  Returns observed values, null mean / sd, z-scores and one-sided
    empirical p-values for (i) the intra-sector edge share and (ii) the share
    of mono-sector triangles.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(list(labels))
    obs_edge = sector_mixing(a, labels)["intra_fraction"]
    obs_tri = triangle_census(a, labels)["mono_fraction"]
    null_edge = np.empty(n_perm)
    null_tri = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(labels)
        null_edge[b] = sector_mixing(a, perm)["intra_fraction"]
        null_tri[b] = triangle_census(a, perm)["mono_fraction"]

    def _summ(obs, null):
        sd = null.std(ddof=1)
        return {
            "observed": float(obs),
            "null_mean": float(null.mean()),
            "null_sd": float(sd),
            "z": float((obs - null.mean()) / sd) if sd > 0 else np.nan,
            "p_value": float((1 + (null >= obs).sum()) / (n_perm + 1)),
        }

    return {"edges": _summ(obs_edge, null_edge), "triangles": _summ(obs_tri, null_tri), "n_perm": n_perm}


# --------------------------------------------------------------------------- #
# Partition comparison (used by the market-mode extension)
# --------------------------------------------------------------------------- #
def normalized_mutual_information(labels_a, labels_b) -> float:
    """NMI (arithmetic normalisation) between two labelings of the same nodes."""
    a, _ = _codes(labels_a)
    b, _ = _codes(labels_b)
    n = len(a)
    cont = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(cont, (a, b), 1)
    pij = cont / n
    pi, pj = pij.sum(axis=1), pij.sum(axis=0)
    nz = pij > 0
    mi = (pij[nz] * np.log(pij[nz] / np.outer(pi, pj)[nz])).sum()
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()  # noqa: E731
    denom = (h(pi) + h(pj)) / 2
    return float(mi / denom) if denom > 0 else 0.0


def threshold_for_edge_count(corr: np.ndarray, m: int) -> float:
    """The threshold that leaves exactly ``m`` edges (density matching)."""
    c = np.asarray(corr)
    vals = np.sort(c[np.triu_indices_from(c, k=1)])[::-1]
    m = int(min(max(m, 1), len(vals) - 1))
    return float((vals[m - 1] + vals[m]) / 2)
