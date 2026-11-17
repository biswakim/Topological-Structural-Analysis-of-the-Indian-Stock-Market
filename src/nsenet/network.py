"""From return series to correlation matrices to threshold networks."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of the columns of ``returns``.

    rho_ij = ( <R_i R_j> - <R_i><R_j> ) / ( sigma_i sigma_j )
    """
    x = returns.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    x = x / x.std(axis=0, ddof=1)
    c = (x.T @ x) / (x.shape[0] - 1)
    c = np.clip((c + c.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return pd.DataFrame(c, index=returns.columns, columns=returns.columns)


def upper_triangle(corr: pd.DataFrame | np.ndarray) -> np.ndarray:
    """The N(N-1)/2 distinct off-diagonal correlations as a flat vector."""
    c = np.asarray(corr)
    return c[np.triu_indices_from(c, k=1)]


# --------------------------------------------------------------------------- #
# Threshold network
# --------------------------------------------------------------------------- #
def adjacency(corr: pd.DataFrame | np.ndarray, r: float, absolute: bool = False) -> np.ndarray:
    """Boolean adjacency matrix: A_ij = 1 iff rho_ij > r (i != j)."""
    c = np.asarray(corr, dtype=float)
    a = (np.abs(c) > r) if absolute else (c > r)
    a = a.copy()
    np.fill_diagonal(a, False)
    return a


def threshold_graph(
    corr: pd.DataFrame,
    r: float,
    sectors: pd.Series | dict | None = None,
    names: pd.Series | dict | None = None,
    absolute: bool = False,
) -> nx.Graph:
    """Unweighted threshold network.

    Every stock is a node (isolated stocks are *kept*: how many stocks fall out
    of the network is itself a result).  An edge i-j exists iff rho_ij > r; the
    correlation is stored as the edge attribute ``weight`` and the metric
    distance ``d = sqrt(2 (1 - rho))`` as ``distance`` for weighted variants.
    """
    a = adjacency(corr, r, absolute)
    c = corr.to_numpy()
    labels = list(corr.index)
    g = nx.Graph(threshold=float(r))
    g.add_nodes_from(labels)
    ii, jj = np.where(np.triu(a, k=1))
    g.add_edges_from(
        (labels[i], labels[j], {"weight": float(c[i, j]), "distance": float(np.sqrt(2 * (1 - c[i, j])))})
        for i, j in zip(ii, jj)
    )
    if sectors is not None:
        nx.set_node_attributes(g, {n: dict(sectors).get(n, "Unknown") for n in labels}, "sector")
    if names is not None:
        nx.set_node_attributes(g, {n: dict(names).get(n, n) for n in labels}, "name")
    return g


# --------------------------------------------------------------------------- #
# Statistical yard-sticks for choosing r
# --------------------------------------------------------------------------- #
def critical_correlation(n_obs: int, alpha: float = 0.01, n_tests: int = 1) -> float:
    """Smallest |rho| that is significantly non-zero (two-sided t-test).

    With ``n_tests`` > 1 a Bonferroni correction is applied, which is the
    relevant case here because N(N-1)/2 pairs are tested at once.
    """
    a = alpha / n_tests
    t = stats.t.ppf(1 - a / 2, df=n_obs - 2)
    return float(t / np.sqrt(n_obs - 2 + t**2))


def marchenko_pastur_bounds(n_obs: int, n_assets: int) -> tuple[float, float]:
    """Eigenvalue support of a pure-noise correlation matrix (Q = T/N)."""
    q = n_assets / n_obs
    return float((1 - np.sqrt(q)) ** 2), float((1 + np.sqrt(q)) ** 2)


def eigen_spectrum(corr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors (columns) of ``corr``."""
    w, v = np.linalg.eigh(corr.to_numpy())
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


# --------------------------------------------------------------------------- #
# Market-mode handling (extension)
# --------------------------------------------------------------------------- #
def market_return(returns: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.Series:
    """The market factor: a benchmark index if supplied, else the
    equal-weighted cross-sectional mean return of the universe."""
    if benchmark is not None:
        m = benchmark.reindex(returns.index)
        if m.notna().mean() > 0.98:
            return m.fillna(0.0).rename("market")
    return returns.mean(axis=1).rename("market")


def market_model(returns: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """OLS single-index model  R_i = a_i + b_i R_m + e_i  for every stock.

    Returns a table with alpha, beta, the stock-market correlation ``rho_m``
    and R^2 (= rho_m^2).
    """
    m = market.to_numpy(dtype=float)
    x = returns.to_numpy(dtype=float)
    mc = m - m.mean()
    beta = (mc @ (x - x.mean(axis=0))) / (mc @ mc)
    alpha = x.mean(axis=0) - beta * m.mean()
    rho_m = beta * m.std(ddof=1) / x.std(axis=0, ddof=1)
    return pd.DataFrame({"alpha": alpha, "beta": beta, "rho_m": rho_m, "r2": rho_m**2}, index=returns.columns)


def residual_returns(returns: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Returns with the market component regressed out (e_i of the model)."""
    mm = market_model(returns, market)
    m = market.to_numpy(dtype=float)[:, None]
    resid = returns.to_numpy(dtype=float) - mm["alpha"].to_numpy() - m * mm["beta"].to_numpy()
    return pd.DataFrame(resid, index=returns.index, columns=returns.columns)


def one_factor_correlation(rho_m: pd.Series) -> pd.DataFrame:
    """Correlation matrix implied by a pure single-index model:
    rho_ij = rho_im * rho_jm for i != j.  Thresholding it gives the network
    that the market mode *alone* would produce."""
    b = rho_m.to_numpy(dtype=float)
    c = np.outer(b, b)
    np.fill_diagonal(c, 1.0)
    return pd.DataFrame(c, index=rho_m.index, columns=rho_m.index)
