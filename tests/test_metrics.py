import networkx as nx
import numpy as np
import pytest

from nsenet.metrics import (
    centrality_table, degree_distribution, local_clustering, normalized_mutual_information, sector_mixing,
    structure_summary, threshold_for_edge_count, transitivity, triangle_census,
)
from nsenet.network import adjacency, threshold_graph


@pytest.fixture(scope="module")
def net(synthetic):
    _, sectors, corr = synthetic
    r = 0.38
    return threshold_graph(corr, r, sectors), adjacency(corr, r), sectors


def test_fast_metrics_agree_with_networkx(net):
    g, a, _ = net
    s = structure_summary(a)
    assert s["n_edges"] == g.number_of_edges()
    assert np.isclose(s["avg_clustering"], nx.average_clustering(g))
    assert np.isclose(s["transitivity"], nx.transitivity(g))
    assert np.isclose(transitivity(a), nx.transitivity(g))
    gcc = g.subgraph(max(nx.connected_components(g), key=len))
    assert s["gcc_size"] == gcc.number_of_nodes()
    assert np.isclose(s["avg_path_length"], nx.average_shortest_path_length(gcc))
    assert s["diameter"] == nx.diameter(gcc)
    assert np.isclose(s["global_efficiency"], nx.global_efficiency(g))
    assert np.allclose(local_clustering(a), [nx.clustering(g)[n] for n in g.nodes()])


def test_centrality_table(net):
    g, _, _ = net
    tab = centrality_table(g)
    assert np.allclose(tab["degree_centrality"], tab["degree"] / (len(tab) - 1))
    assert tab["betweenness"].between(0, 1).all()
    assert np.isclose((tab["eigenvector"] ** 2).sum(), 1.0, atol=1e-6)


def test_degree_distribution_sums_to_one(net):
    g, _, _ = net
    dist = degree_distribution([d for _, d in g.degree()])
    assert np.isclose(dist["pk"].sum(), 1.0) and np.isclose(dist["ccdf"].iloc[0], 1.0)


def test_sector_assortativity_matches_networkx(net):
    g, a, sectors = net
    mix = sector_mixing(a, sectors.to_numpy())
    assert np.isclose(mix["assortativity"], nx.attribute_assortativity_coefficient(g, "sector"))
    intra = sum(g.nodes[u]["sector"] == g.nodes[v]["sector"] for u, v in g.edges()) / g.number_of_edges()
    assert np.isclose(mix["intra_fraction"], intra)


def test_triangle_census_brute_force(net):
    g, a, sectors = net
    sec = sectors.to_dict()
    counts = {"mono": 0, "two_same": 0, "mixed": 0}
    for clique in nx.enumerate_all_cliques(g):
        if len(clique) > 3:
            break
        if len(clique) == 3:
            k = len({sec[n] for n in clique})
            counts["mono" if k == 1 else "two_same" if k == 2 else "mixed"] += 1
    census = triangle_census(a, sectors.to_numpy())
    assert census["triangles"] == sum(counts.values())
    for key in counts:
        assert census[key] == counts[key]


def test_nmi_and_density_matching(synthetic):
    _, sectors, corr = synthetic
    lab = sectors.to_numpy()
    assert np.isclose(normalized_mutual_information(lab, lab), 1.0)
    rng = np.random.default_rng(0)
    assert normalized_mutual_information(lab, rng.permutation(lab)) < 0.2
    thr = threshold_for_edge_count(corr.to_numpy(), 500)
    assert adjacency(corr, thr).sum() // 2 == 500
