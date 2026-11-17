"""The shared path from the cached CSVs to a ready-to-analyse data set.

Every analysis script starts with :func:`prepare`, so they all see exactly the
same stocks, the same trading days and the same correlation matrix.  Keeping
this in one place is what makes the separate scripts comparable: a change to
the cleaning policy propagates to every result at once.

The module also holds the two small output helpers (:func:`save_table`,
:func:`save_numbers`).  Scalars destined for the report are written to
``results/numbers/<tag>.json`` and turned into LaTeX macros by
``scripts/09_export_report_assets.py``, so no figure quoted in the write-up is
typed in by hand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .config import REPO_ROOT
from .data import load_cached, price_paths
from .network import correlation_matrix, threshold_graph
from .preprocess import clean_prices, log_returns

# NSE's industry names are too long for axis labels and table headers.
SECTOR_SHORT = {
    "Financial Services": "Financials",
    "Capital Goods": "Capital Goods",
    "Healthcare": "Healthcare",
    "Automobile and Auto Components": "Auto",
    "Fast Moving Consumer Goods": "FMCG",
    "Chemicals": "Chemicals",
    "Information Technology": "IT",
    "Metals & Mining": "Metals & Mining",
    "Oil Gas & Consumable Fuels": "Oil & Gas",
    "Consumer Durables": "Consumer Dur.",
    "Power": "Power",
    "Consumer Services": "Consumer Serv.",
    "Realty": "Realty",
    "Services": "Services",
    "Construction": "Construction",
    "Construction Materials": "Cement",
    "Telecommunication": "Telecom",
    "Textiles": "Textiles",
    "Media Entertainment & Publication": "Media",
    "Diversified": "Diversified",
}

PROCESSED = REPO_ROOT / "data" / "processed"
NUMBERS = REPO_ROOT / "results" / "numbers"
REFERENCE = REPO_ROOT / "data" / "reference"


def sym(ticker: str) -> str:
    """'RELIANCE.NS' -> 'RELIANCE' (Yahoo suffix removed for display)."""
    return ticker[:-3] if str(ticker).endswith(".NS") else str(ticker)


@dataclass
class Data:
    """Everything an analysis script needs, built once and shared."""

    prices: pd.DataFrame            # cleaned adjusted closes, days x stocks
    returns: pd.DataFrame           # daily log returns, mechanical resets masked
    corr: pd.DataFrame              # Pearson correlation matrix of the returns
    members: pd.DataFrame           # per-stock reference data, indexed by ticker
    audit: pd.DataFrame             # every extreme return, with its classification
    clean_report: dict              # counts for the data-quality table
    meta: dict = field(default_factory=dict)        # download provenance
    benchmarks: pd.DataFrame | None = None          # Nifty 50 / Nifty 500 levels

    @property
    def sectors(self) -> pd.Series:
        return self.members["sector"]

    @property
    def names(self) -> pd.Series:
        return self.members["Company Name"]

    @property
    def market(self) -> pd.Series:
        """Equal-weighted market factor: the cross-sectional mean return.

        Preferred to an index level because it is built from exactly the
        stocks in the network, with the same cleaning applied to it.
        """
        return self.returns.mean(axis=1).rename("market")

    def graph(self, r: float) -> nx.Graph:
        """Unweighted threshold network at correlation threshold ``r``."""
        return threshold_graph(self.corr, r, self.sectors, self.names)


# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #
def load_psu_flags(tickers: pd.Index) -> pd.Series:
    """Government-majority ownership flag from ``data/reference/psu_list.csv``.

    Hand-compiled; used only for grouping and colouring, never in the
    construction of the network.
    """
    path = REFERENCE / "psu_list.csv"
    if not path.exists():
        return pd.Series(False, index=tickers, name="psu")
    listed = set(pd.read_csv(path)["Symbol"].astype(str).str.strip())
    return pd.Series([sym(t) in listed for t in tickers], index=tickers, name="psu")


def _volume_multiple(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Traded volume as a multiple of its trailing median.

    A mechanical price reset (a split the data vendor failed to back-adjust)
    normally happens on ordinary volume; genuine news does not.
    """
    med = volume.rolling(window).median().shift(1)
    return volume / med.replace(0, np.nan)


# --------------------------------------------------------------------------- #
# The extreme-return audit
# --------------------------------------------------------------------------- #
def audit_extreme_returns(returns: pd.DataFrame, volume: pd.DataFrame | None,
                          flag: float, overrides: list[dict],
                          notes: dict | None = None) -> pd.DataFrame:
    """Every daily return larger than ``flag`` in absolute value, classified.

    NSE's widest daily price band for ordinary stocks is 20%, so a larger move
    needs an explanation.  A row is marked *mechanical* when the (ticker, date)
    pair appears in ``overrides`` - the hand-checked list in ``config.yaml`` -
    and *genuine* otherwise.  The traded volume, as a multiple of its trailing
    median, is reported alongside because it discriminates between the two: a
    share-count change happens on ordinary volume, news does not.
    """
    keyed = {(o["ticker"], pd.Timestamp(o["date"])): o["reason"] for o in overrides}
    notes = {tuple(k.split("|")): v for k, v in (notes or {}).items()}
    volmult = _volume_multiple(volume.reindex(index=returns.index, columns=returns.columns)) \
        if volume is not None else None

    rows = []
    big = returns.abs() > flag
    for tic in returns.columns[big.any()]:
        for dt in returns.index[big[tic]]:
            r = float(returns.at[dt, tic])
            vm = np.nan
            if volmult is not None and pd.notna(volmult.at[dt, tic]):
                vm = round(float(volmult.at[dt, tic]), 4)
            rows.append(
                {
                    "date": dt.date(),
                    "ticker": tic,
                    "log_return": round(r, 4),
                    "simple_return_pct": round(100 * (np.exp(r) - 1), 4),
                    "volume_multiple": vm,
                    "classification": ("mechanical (masked)" if (tic, dt) in keyed
                                       else "genuine (kept)"),
                    "note": keyed.get((tic, dt), notes.get((tic, str(dt.date())), "")),
                }
            )
    audit = pd.DataFrame(rows, columns=["date", "ticker", "log_return", "simple_return_pct",
                                        "volume_multiple", "classification", "note"])
    if len(audit):
        audit = (audit.assign(_abs=audit["log_return"].abs())
                 .sort_values(["classification", "_abs"], ascending=[True, False])
                 .drop(columns="_abs").reset_index(drop=True))
    return audit


def apply_overrides(returns: pd.DataFrame, overrides: list[dict]) -> tuple[pd.DataFrame, int]:
    """Set the listed (ticker, date) returns to zero; return the new frame and a count.

    Zero, rather than deletion, keeps the panel rectangular: the stock simply
    has no economic return on the day its share count changed.
    """
    out = returns.copy()
    n = 0
    for o in overrides:
        tic, dt = o["ticker"], pd.Timestamp(o["date"])
        if tic in out.columns and dt in out.index:
            out.at[dt, tic] = 0.0
            n += 1
    return out, n


# --------------------------------------------------------------------------- #
# The main entry point
# --------------------------------------------------------------------------- #
def prepare(cfg: dict, save: bool = True) -> Data:
    """Clean prices, build returns, audit extreme moves, correlate.

    The audit is the only place where a return is altered.  A return is changed
    if and only if the (ticker, date) pair appears in
    ``preprocess.return_overrides`` in ``config.yaml``; each of those entries
    was checked by hand against the unadjusted close and the traded volume and
    is documented in ``data/reference/return_audit.md``.  Such a day is a
    share-count change or a demerger, not a return to a continuing shareholder,
    so that single return is set to zero rather than left to contaminate every
    correlation the stock takes part in.
    """
    raw = load_cached(cfg)
    pre = cfg["preprocess"]

    prices, rep = clean_prices(raw["adj_close"], pre["min_coverage"], pre["max_ffill_days"])
    returns = log_returns(prices)
    rep["n_returns"] = int(len(returns))

    # ------------------------------------------------------------------ audit
    overrides = pre.get("return_overrides", []) or []
    audit = audit_extreme_returns(
        returns,
        raw.get("volume"),
        float(pre.get("extreme_return_flag", 0.20)),
        overrides,
        pre.get("genuine_event_notes"),
    )
    rep["extreme_returns_flagged"] = int(len(audit))
    returns, rep["returns_masked"] = apply_overrides(returns, overrides)

    # ------------------------------------------------------------ correlations
    corr = correlation_matrix(returns)

    # --------------------------------------------------------- reference data
    members = raw["members"].set_index("Ticker").reindex(returns.columns)
    members.index.name = "Ticker"
    members["sector"] = members["Industry"].map(SECTOR_SHORT).fillna(members["Industry"])
    members["psu"] = load_psu_flags(members.index)

    if "close" in raw and "volume" in raw:
        c = raw["close"].reindex(index=prices.index, columns=members.index)
        v = raw["volume"].reindex(index=prices.index, columns=members.index)
        members["median_traded_value"] = (c * v).median(axis=0)
    else:
        members["median_traded_value"] = np.nan

    mcap_path = price_paths(Path(cfg["data"]["raw_dir"]), cfg["data"]["universe"], cfg["data"]["start"],
                            cfg["data"]["end"], cfg["data"]["interval"])["market_cap"]
    members["market_cap"] = (
        pd.read_csv(mcap_path).set_index("Ticker")["MarketCap"].reindex(members.index)
        if mcap_path.exists() else np.nan
    )

    # Index levels become log returns on the same trading days as the stocks,
    # so the equal-weighted market factor can be compared with them directly.
    benchmarks = None
    if "benchmarks" in raw:
        # No forward-filling here: a day on which the index did not report is
        # left missing and excluded by the consumer, rather than invented.
        benchmarks = log_returns(raw["benchmarks"].reindex(prices.index)).reindex(returns.index)

    d = Data(prices=prices[returns.columns], returns=returns, corr=corr, members=members,
             audit=audit, clean_report=rep, meta=raw.get("meta", {}),
             benchmarks=benchmarks)

    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        d.prices.to_csv(PROCESSED / "prices_clean.csv", float_format="%.4f")
        d.returns.to_csv(PROCESSED / "log_returns.csv", float_format="%.8f")
        d.corr.to_csv(PROCESSED / "correlation_matrix.csv", float_format="%.6f")
        d.members.to_csv(PROCESSED / "stocks.csv")
    return d


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def save_table(df: pd.DataFrame, cfg: dict, name: str, index: bool = True,
               float_format: str = "%.6g") -> Path:
    """Write a results table to ``results/tables/<name>.csv``."""
    out = Path(cfg["output"]["tables_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    df.to_csv(path, index=index, float_format=float_format)
    print(f"  table  -> {path.name}")
    return path


def save_numbers(tag: str, numbers: dict) -> Path:
    """Write the scalars a report section needs to ``results/numbers/<tag>.json``."""
    NUMBERS.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in numbers.items():
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        clean[k] = v
    path = NUMBERS / f"{tag}.json"
    path.write_text(json.dumps(clean, indent=2, sort_keys=True, default=str))
    print(f"  numbers-> {path.name} ({len(clean)} values)")
    return path


def load_numbers(tag: str) -> dict:
    path = NUMBERS / f"{tag}.json"
    return json.loads(path.read_text()) if path.exists() else {}
