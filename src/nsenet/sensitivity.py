"""D. Threshold sensitivity: rebuild the network for many values of r."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import sector_mixing, structure_summary
from .network import adjacency


def threshold_sweep(corr: pd.DataFrame, thresholds, sectors=None, absolute: bool = False) -> pd.DataFrame:
    """One row of structural descriptors per threshold.

    The correlation matrix is computed once; only the (cheap) thresholding and
    graph measurements are repeated, so a fine sweep costs seconds.
    """
    rows = []
    for r in thresholds:
        a = adjacency(corr, float(r), absolute)
        row = {"threshold": round(float(r), 4)}
        row.update(structure_summary(a))
        if sectors is not None and row["n_edges"] > 0:
            mix = sector_mixing(a, sectors)
            row["intra_sector_fraction"] = mix["intra_fraction"]
            row["sector_assortativity"] = mix["assortativity"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


def percolation_point(sweep: pd.DataFrame) -> dict:
    """Locate the connectivity transition of the sweep.

    Three complementary estimates are returned:

    r_half     largest r at which the giant component still holds >= 50% of nodes
    r_suscept  r at which the mean finite-cluster size (susceptibility) peaks
    r_second   r at which the second-largest component is largest
    """
    s = sweep.sort_index()
    half = s.index[s["gcc_fraction"] >= 0.5]
    return {
        "r_half": float(half.max()) if len(half) else np.nan,
        "r_suscept": float(s["susceptibility"].idxmax()),
        "r_second": float(s["second_component"].idxmax()),
    }


def edge_survival(corr: pd.DataFrame, thresholds) -> pd.Series:
    """Fraction of all N(N-1)/2 pairs whose correlation exceeds each r
    (the empirical survival function of the correlation distribution)."""
    c = corr.to_numpy()
    vals = np.sort(c[np.triu_indices_from(c, k=1)])
    thr = np.asarray(list(thresholds), dtype=float)
    frac = 1.0 - np.searchsorted(vals, thr, side="right") / len(vals)
    return pd.Series(frac, index=thr, name="edge_fraction")
