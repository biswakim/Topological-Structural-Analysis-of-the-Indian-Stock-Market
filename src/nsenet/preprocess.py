"""Price cleaning and log-return computation.

The cleaning policy is intentionally conservative and fully reported: every
stock or day that is removed is counted in the returned ``report`` dict so the
write-up can state exactly what happened to the raw download.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_prices(
    adj_close: pd.DataFrame,
    min_coverage: float = 0.97,
    max_ffill_days: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Turn the raw download into a complete (no-NaN) price panel.

    Steps
    -----
    1. Drop tickers for which Yahoo returned nothing at all.
    2. Drop calendar rows on which fewer than half of the remaining tickers
       traded (exchange holidays that leak in through a few stale quotes,
       special one-hour sessions, ...).
    3. Drop tickers observed on fewer than ``min_coverage`` of the trading
       days.  In practice this removes companies that listed (IPO / demerger)
       after the start of the window: they cannot be correlated over the full
       sample.
    4. Forward-fill gaps of at most ``max_ffill_days`` (a suspended or
       non-traded stock keeps its last price, i.e. a zero return).
    5. Drop any ticker that still has a missing value.
    """
    report: dict = {"tickers_requested": int(adj_close.shape[1]), "rows_raw": int(adj_close.shape[0])}
    px = adj_close.sort_index()
    px = px[~px.index.duplicated(keep="last")]
    px = px.where(px > 0)  # non-positive prices are data errors

    empty = px.columns[px.notna().sum() == 0].tolist()
    px = px.drop(columns=empty)
    report["dropped_no_data"] = empty

    traded_frac = px.notna().mean(axis=1)
    thin_days = px.index[traded_frac < 0.5]
    px = px.loc[traded_frac >= 0.5]
    report["dropped_thin_days"] = [str(d.date()) for d in thin_days]

    coverage = px.notna().mean()
    short = coverage[coverage < min_coverage].sort_values()
    px = px.drop(columns=short.index)
    report["dropped_low_coverage"] = {k: round(float(v), 4) for k, v in short.items()}

    n_gaps = int(px.isna().sum().sum())
    px = px.ffill(limit=max_ffill_days)
    report["cells_forward_filled"] = n_gaps - int(px.isna().sum().sum())

    still = px.columns[px.isna().any()].tolist()
    px = px.drop(columns=still)
    report["dropped_residual_gaps"] = still

    report["tickers_kept"] = int(px.shape[1])
    report["trading_days"] = int(px.shape[0])
    report["first_date"] = str(px.index.min().date())
    report["last_date"] = str(px.index.max().date())
    return px, report


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r"""Daily logarithmic returns  :math:`R_i(t) = \ln\,[P_i(t) / P_i(t-1)]`."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def return_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-stock descriptive statistics of the return series."""
    out = pd.DataFrame(
        {
            "mean_daily": returns.mean(),
            "vol_daily": returns.std(ddof=1),
            "ann_vol": returns.std(ddof=1) * np.sqrt(252),
            "skew": returns.skew(),
            "excess_kurtosis": returns.kurt(),
            "min": returns.min(),
            "max": returns.max(),
            "zero_return_frac": (returns == 0).mean(),
        }
    )
    out.index.name = "Ticker"
    return out
