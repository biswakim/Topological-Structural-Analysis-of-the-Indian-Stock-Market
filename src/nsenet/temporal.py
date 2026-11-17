"""Rolling-window analysis: is the network a stable object or a moving target?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import sector_mixing, structure_summary
from .network import adjacency, correlation_matrix, upper_triangle


def rolling_networks(
    returns: pd.DataFrame, r: float, window: int = 250, step: int = 21, sectors=None
) -> pd.DataFrame:
    """Slide a ``window``-day estimation window through the sample.

    For every window: mean pairwise correlation, share of variance carried by
    the first eigenvector (the "market mode"), and the structure of the
    threshold-``r`` network.  ``edge_jaccard_prev`` measures how many edges
    are shared with the previous window.
    """
    rows = []
    prev = None
    t = len(returns)
    for end in range(window, t + 1, step):
        chunk = returns.iloc[end - window : end]
        corr = correlation_matrix(chunk)
        a = adjacency(corr, r)
        iu = np.triu_indices(a.shape[0], k=1)
        e = a[iu]
        row = {
            "window_end": chunk.index[-1],
            "window_start": chunk.index[0],
            "mean_correlation": float(upper_triangle(corr).mean()),
            "market_mode_share": float(np.linalg.eigvalsh(corr.to_numpy())[-1] / corr.shape[0]),
            "market_volatility": float(chunk.mean(axis=1).std(ddof=1) * np.sqrt(252)),
        }
        row.update(structure_summary(a))
        if sectors is not None and row["n_edges"] > 0:
            row["intra_sector_fraction"] = sector_mixing(a, sectors)["intra_fraction"]
        row["edge_jaccard_prev"] = (
            float((e & prev).sum() / max((e | prev).sum(), 1)) if prev is not None else np.nan
        )
        prev = e
        rows.append(row)
    return pd.DataFrame(rows).set_index("window_end")


def split_half_comparison(returns: pd.DataFrame, r: float) -> dict:
    """Networks from the first and second half of the sample, compared."""
    half = len(returns) // 2
    c1 = correlation_matrix(returns.iloc[:half])
    c2 = correlation_matrix(returns.iloc[half:])
    a1, a2 = adjacency(c1, r), adjacency(c2, r)
    iu = np.triu_indices(a1.shape[0], k=1)
    e1, e2 = a1[iu], a2[iu]
    k1, k2 = a1.sum(axis=1), a2.sum(axis=1)
    from scipy.stats import spearmanr

    return {
        "split_date": str(returns.index[half].date()),
        "edges_first": int(e1.sum()),
        "edges_second": int(e2.sum()),
        "edge_jaccard": float((e1 & e2).sum() / max((e1 | e2).sum(), 1)),
        "corr_of_correlations": float(np.corrcoef(upper_triangle(c1), upper_triangle(c2))[0, 1]),
        "degree_spearman": float(spearmanr(k1, k2)[0]),
        "mean_corr_first": float(upper_triangle(c1).mean()),
        "mean_corr_second": float(upper_triangle(c2).mean()),
    }
