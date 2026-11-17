"""Synthetic return generator - FOR TESTS ONLY.

Produces returns from a market + sector factor model with fat-tailed
idiosyncratic noise.  It exists so the unit tests can check the code against a
data set whose correlation structure is known in advance.  No number in the
report comes from this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def factor_model_returns(
    n_sectors: int = 6,
    stocks_per_sector: int = 20,
    n_days: int = 500,
    market_vol: float = 0.009,
    sector_vol: float = 0.008,
    idio_vol: float = 0.014,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = n_sectors * stocks_per_sector
    market = rng.standard_normal(n_days) * market_vol
    sector_f = rng.standard_normal((n_days, n_sectors)) * sector_vol
    sector_of = np.repeat(np.arange(n_sectors), stocks_per_sector)
    beta_m = rng.uniform(0.5, 1.5, n)
    beta_s = rng.uniform(0.3, 1.4, n)
    idio = rng.standard_t(df=5, size=(n_days, n)) * idio_vol / np.sqrt(5 / 3)
    r = market[:, None] * beta_m + sector_f[:, sector_of] * beta_s + idio
    cols = [f"S{sector_of[i]}_{i:03d}.NS" for i in range(n)]
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    sectors = pd.Series([f"Sector {s}" for s in sector_of], index=cols, name="sector")
    return pd.DataFrame(r, index=idx, columns=cols), sectors
