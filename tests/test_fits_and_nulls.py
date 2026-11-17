import numpy as np

from nsenet import fits as F
from nsenet.metrics import structure_summary
from nsenet.network import adjacency
from nsenet.nulls import bootstrap_days, degree_preserving_rewire, erdos_renyi_adjacency, shuffled_correlations
from nsenet.sensitivity import percolation_point, threshold_sweep


def test_powerlaw_mle_recovers_exponent():
    rng = np.random.default_rng(3)
    x = F._sample_powerlaw(2.5, 2, 4000, rng)
    fit = F.fit_powerlaw(x, 2)
    assert abs(fit.params["alpha"] - 2.5) < 0.08


def test_exponential_beats_powerlaw_on_geometric_data():
    rng = np.random.default_rng(4)
    x = rng.geometric(0.08, 3000) + 4
    fits = F.fit_tail(x, 5)
    r, p = F.vuong(fits["power law"], fits["exponential"])
    assert r < 0 and p < 0.01
    assert abs(fits["exponential"].params["lambda"] - (-np.log(1 - 0.08))) < 0.01


def test_powerlaw_beats_exponential_on_powerlaw_data():
    rng = np.random.default_rng(5)
    x = F._sample_powerlaw(2.2, 1, 3000, rng)
    fits = F.fit_tail(x, 1)
    r, p = F.vuong(fits["power law"], fits["exponential"])
    assert r > 0 and p < 0.01
    gof = F.powerlaw_gof(x[:600], n_boot=40, seed=1)
    assert gof["p_value"] > 0.1                       # a true power law is not rejected


def test_rewiring_preserves_degrees(synthetic):
    _, _, corr = synthetic
    a = adjacency(corr, 0.38)
    rng = np.random.default_rng(0)
    b = degree_preserving_rewire(a, rng)
    assert (a.sum(axis=1) == b.sum(axis=1)).all()
    assert (b == b.T).all() and not b.diagonal().any()
    assert (a != b).any()
    er = erdos_renyi_adjacency(a.shape[0], int(a.sum() // 2), rng)
    assert er.sum() == a.sum()
    # real correlation networks are far more clustered than their ER counterpart
    assert structure_summary(a)["transitivity"] > 2 * structure_summary(er)["transitivity"]


def test_shuffle_destroys_correlation(synthetic):
    returns, _, _ = synthetic
    null = shuffled_correlations(returns, n_shuffles=3, seed=0)
    assert abs(null.mean()) < 0.01
    assert abs(null.std() - 1 / np.sqrt(len(returns))) < 0.01
    c = next(bootstrap_days(returns, 1, seed=0, block=5))
    assert c.shape == (returns.shape[1],) * 2 and np.allclose(np.diag(c), 1.0)


def test_threshold_sweep_shape(synthetic):
    _, sectors, corr = synthetic
    sweep = threshold_sweep(corr, np.arange(0.1, 0.9, 0.05), sectors.to_numpy())
    assert sweep["n_edges"].is_monotonic_decreasing
    assert sweep["gcc_size"].is_monotonic_decreasing
    perc = percolation_point(sweep)
    assert 0.1 <= perc["r_half"] <= 0.9
    # the synthetic sector factors must be visible as homophily
    assert sweep["intra_sector_fraction"].dropna().iloc[-1] > 0.5
