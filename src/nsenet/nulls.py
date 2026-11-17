"""Null models and resampling.

A number such as "average clustering = 0.6" means little on its own.  Each
function below produces the reference against which an observed statistic is
judged:

``erdos_renyi_adjacency``     same N and M, edges placed uniformly at random
``degree_preserving_rewire``  same degree sequence, everything else randomised
``shuffled_correlations``     same marginal return distributions, all
                              cross-correlation destroyed (finite-T noise floor)
``bootstrap_days``            resample trading days -> sampling uncertainty
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import structure_summary
from .network import adjacency


# --------------------------------------------------------------------------- #
# Random-graph nulls
# --------------------------------------------------------------------------- #
def erdos_renyi_adjacency(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """G(n, m): exactly ``m`` edges chosen uniformly among the n(n-1)/2 pairs."""
    iu = np.triu_indices(n, k=1)
    pick = rng.choice(len(iu[0]), size=m, replace=False)
    a = np.zeros((n, n), dtype=bool)
    a[iu[0][pick], iu[1][pick]] = True
    return a | a.T


def degree_preserving_rewire(a: np.ndarray, rng: np.random.Generator, swaps_per_edge: float = 10.0) -> np.ndarray:
    """Maslov-Sneppen double-edge swaps.

    Pick two edges (a-b, c-d) and replace them by (a-d, c-b) unless that would
    create a self-loop or a multi-edge.  Every node keeps its degree exactly.
    """
    a = a.copy()
    iu = np.where(np.triu(a, k=1))
    edges = np.column_stack(iu)
    m = len(edges)
    if m < 2:
        return a
    n_attempts = int(swaps_per_edge * m)
    picks = rng.integers(0, m, size=(n_attempts, 2))
    flips = rng.random(n_attempts) < 0.5
    for (e1, e2), flip in zip(picks, flips):
        if e1 == e2:
            continue
        u, v = edges[e1]
        x, y = edges[e2]
        if flip:
            x, y = y, x
        # proposed: u-y and x-v
        if u == y or x == v or a[u, y] or a[x, v]:
            continue
        a[u, v] = a[v, u] = False
        a[x, y] = a[y, x] = False
        a[u, y] = a[y, u] = True
        a[x, v] = a[v, x] = True
        edges[e1] = (u, y)
        edges[e2] = (x, v)
    return a


def random_graph_comparison(a: np.ndarray, n_graphs: int = 200, seed: int = 0) -> pd.DataFrame:
    """Observed statistics next to their ER and degree-preserving null
    distributions (mean, sd, z-score).  Also returns the small-world indices

        sigma = (C / C_rand) / (L / L_rand)

    computed against the ER ensemble, on the giant component for L.
    """
    rng = np.random.default_rng(seed)
    obs = structure_summary(a)
    n, m = a.shape[0], obs["n_edges"]
    keys = ["avg_clustering", "transitivity", "avg_path_length", "gcc_fraction", "max_degree"]
    draws = {"ER G(n,m)": [], "degree-preserving": []}
    for _ in range(n_graphs):
        draws["ER G(n,m)"].append(structure_summary(erdos_renyi_adjacency(n, m, rng)))
        draws["degree-preserving"].append(structure_summary(degree_preserving_rewire(a, rng)))
    rows = []
    for key in keys:
        row = {"statistic": key, "observed": obs[key]}
        for name, lst in draws.items():
            vals = np.array([d[key] for d in lst], dtype=float)
            mu, sd = np.nanmean(vals), np.nanstd(vals, ddof=1)
            row[f"{name} mean"] = mu
            row[f"{name} sd"] = sd
            row[f"{name} z"] = (obs[key] - mu) / sd if sd > 0 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).set_index("statistic")
    c_ratio = out.loc["avg_clustering", "observed"] / out.loc["avg_clustering", "ER G(n,m) mean"]
    l_ratio = out.loc["avg_path_length", "observed"] / out.loc["avg_path_length", "ER G(n,m) mean"]
    out.attrs["small_world_sigma"] = float(c_ratio / l_ratio)
    out.attrs["C_ratio"] = float(c_ratio)
    out.attrs["L_ratio"] = float(l_ratio)
    return out


# --------------------------------------------------------------------------- #
# Correlation-level nulls
# --------------------------------------------------------------------------- #
def _corr(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    x = x / x.std(axis=0, ddof=1)
    return (x.T @ x) / (x.shape[0] - 1)


def shuffled_correlations(returns: pd.DataFrame, n_shuffles: int = 20, seed: int = 0) -> np.ndarray:
    """Off-diagonal correlations after independently permuting every stock's
    return series in time.  The result is the distribution of correlations
    that arises purely from having a finite number of observations."""
    rng = np.random.default_rng(seed)
    x = returns.to_numpy(dtype=float)
    t, n = x.shape
    iu = np.triu_indices(n, k=1)
    out = []
    for _ in range(n_shuffles):
        perm = np.argsort(rng.random((t, n)), axis=0)
        out.append(_corr(np.take_along_axis(x, perm, axis=0))[iu])
    return np.concatenate(out)


def bootstrap_days(returns: pd.DataFrame, n_boot: int, seed: int = 0, block: int = 1):
    """Yield correlation matrices from resampled trading days.

    ``block`` > 1 gives a moving-block bootstrap that preserves short-range
    serial dependence (volatility clustering); ``block`` = 1 is the ordinary
    i.i.d. bootstrap over days.  Cross-sectional dependence is always kept
    intact because whole days (rows) are resampled.
    """
    rng = np.random.default_rng(seed)
    x = returns.to_numpy(dtype=float)
    t = x.shape[0]
    for _ in range(n_boot):
        if block <= 1:
            idx = rng.integers(0, t, size=t)
        else:
            starts = rng.integers(0, t - block + 1, size=int(np.ceil(t / block)))
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:t]
        c = _corr(x[idx])
        np.fill_diagonal(c, 1.0)
        yield c


def bootstrap_network_stability(
    returns: pd.DataFrame, r: float, n_boot: int = 200, seed: int = 0, block: int = 5, top: int = 10
) -> dict:
    """How much of the baseline network survives resampling of the data?

    Returns the distribution of the edge count, the edge-set Jaccard overlap
    with the full-sample network, per-edge retention frequencies, and for each
    stock the fraction of replicates in which it is in the degree top-``top``.
    """
    x_corr = _corr(returns.to_numpy(dtype=float))
    a0 = adjacency(x_corr, r)
    n = a0.shape[0]
    iu = np.triu_indices(n, k=1)
    e0 = a0[iu]
    edge_keep = np.zeros(e0.sum())
    in_top = np.zeros(n)
    deg_sum = np.zeros(n)
    deg_sq = np.zeros(n)
    n_edges, jacc, gcc = [], [], []
    for c in bootstrap_days(returns, n_boot, seed, block):
        a = adjacency(c, r)
        eb = a[iu]
        n_edges.append(int(eb.sum()))
        jacc.append((e0 & eb).sum() / max((e0 | eb).sum(), 1))
        edge_keep += eb[e0]
        k = a.sum(axis=1)
        deg_sum += k
        deg_sq += k.astype(float) ** 2
        in_top[np.argsort(-k, kind="stable")[:top]] += 1
        gcc.append(structure_summary(a, with_paths=False)["gcc_fraction"])
    mean_k = deg_sum / n_boot
    sd_k = np.sqrt(np.maximum(deg_sq / n_boot - mean_k**2, 0))
    return {
        "n_edges": np.array(n_edges),
        "jaccard": np.array(jacc),
        "gcc_fraction": np.array(gcc),
        "edge_retention": edge_keep / n_boot,
        "top_frequency": pd.Series(in_top / n_boot, index=returns.columns),
        "degree_mean": pd.Series(mean_k, index=returns.columns),
        "degree_sd": pd.Series(sd_k, index=returns.columns),
        "n_edges_full": int(e0.sum()),
    }
