import networkx as nx
import numpy as np

from nsenet.network import (
    adjacency, critical_correlation, marchenko_pastur_bounds, market_model, market_return, one_factor_correlation,
    residual_returns, threshold_graph,
)


def test_correlation_matches_numpy(synthetic):
    returns, _, corr = synthetic
    assert np.allclose(corr.to_numpy(), np.corrcoef(returns.to_numpy(), rowvar=False), atol=1e-10)
    assert np.allclose(np.diag(corr), 1.0)


def test_threshold_graph_edges(synthetic):
    _, sectors, corr = synthetic
    g = threshold_graph(corr, 0.4, sectors)
    a = adjacency(corr, 0.4)
    assert g.number_of_nodes() == corr.shape[0]          # isolated nodes are kept
    assert g.number_of_edges() == a.sum() // 2
    assert all(d["weight"] > 0.4 for _, _, d in g.edges(data=True))
    assert nx.get_node_attributes(g, "sector")[corr.index[0]] == sectors.iloc[0]
    assert not a.diagonal().any() and (a == a.T).all()


def test_edges_monotone_in_threshold(synthetic):
    _, _, corr = synthetic
    counts = [adjacency(corr, r).sum() for r in np.arange(0.0, 1.0, 0.05)]
    assert all(x >= y for x, y in zip(counts, counts[1:]))


def test_residuals_are_orthogonal_to_market(synthetic):
    returns, _, _ = synthetic
    m = market_return(returns)
    resid = residual_returns(returns, m)
    cov = (resid.to_numpy() * (m.to_numpy()[:, None] - m.mean())).mean(axis=0)
    assert np.abs(cov).max() < 1e-12
    mm = market_model(returns, m)
    assert np.allclose(mm["rho_m"], [np.corrcoef(returns[c], m)[0, 1] for c in returns], atol=1e-10)
    c1 = one_factor_correlation(mm["rho_m"])
    assert np.isclose(c1.iloc[0, 1], mm["rho_m"].iloc[0] * mm["rho_m"].iloc[1])


def test_yardsticks():
    assert 0.09 < critical_correlation(739, 0.01) < 0.10
    assert critical_correlation(739, 0.01, 10_000) > critical_correlation(739, 0.01)
    lo, hi = marchenko_pastur_bounds(1000, 250)
    assert np.isclose(hi, 2.25) and np.isclose(lo, 0.25)
