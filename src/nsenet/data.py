"""Data acquisition: index constituents and adjusted-close prices.

Design rule (from the assignment): *save data locally to prevent redundant
server requests*.  Every function here is therefore "cache first": it returns
the file on disk when it exists and only talks to a server when it does not
(or when ``force=True``).
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Official constituent lists (the links given in the assignment hand-out).
CONSTITUENT_URLS: dict[str, list[str]] = {
    "nifty100": [
        "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    ],
    "nifty200": [
        "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    ],
    "nifty500": [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    ],
}

# NSE servers refuse requests that do not look like a browser.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.niftyindices.com/",
}


# --------------------------------------------------------------------------- #
# Constituents
# --------------------------------------------------------------------------- #
def constituents_path(raw_dir: Path, universe: str) -> Path:
    return Path(raw_dir) / f"ind_{universe}list.csv"


def _clean_constituents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip())
    needed = {"Company Name", "Industry", "Symbol"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Constituent file is missing columns {sorted(missing)}; got {list(df.columns)}")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    df = df[df["Symbol"].str.len() > 0].drop_duplicates("Symbol").reset_index(drop=True)
    df["Ticker"] = df["Symbol"] + ".NS"  # Yahoo Finance suffix for NSE listings
    return df


def load_constituents(raw_dir: Path, universe: str, force: bool = False) -> pd.DataFrame:
    """Return the constituent table (Company Name, Industry, Symbol, Ticker, ...).

    Tries the cached CSV first, then each official URL in turn.  If every URL
    fails (NSE occasionally blocks scripted access) the error message explains
    how to drop a manually downloaded CSV into ``data/raw``.
    """
    path = constituents_path(raw_dir, universe)
    if path.exists() and not force:
        return _clean_constituents(pd.read_csv(path))

    import requests  # imported lazily: analysis code never needs the network

    errors = []
    for url in CONSTITUENT_URLS[universe]:
        try:
            resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")))
            df = _clean_constituents(df)
            path.write_bytes(resp.content)
            print(f"[constituents] {len(df)} symbols from {url}")
            return df
        except Exception as exc:  # noqa: BLE001 - we want to try the next mirror
            errors.append(f"  {url}\n    -> {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not download the constituent list from any mirror:\n"
        + "\n".join(errors)
        + f"\n\nWorkaround: open one of the links above in a browser and save the file as\n  {path}\n"
        "then run this script again."
    )


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def price_paths(raw_dir: Path, universe: str, start: str, end: str, interval: str) -> dict[str, Path]:
    stem = f"{universe}_{start}_{end}_{interval}"
    raw_dir = Path(raw_dir)
    return {
        "adj_close": raw_dir / f"adj_close_{stem}.csv",
        "close": raw_dir / f"close_{stem}.csv",
        "volume": raw_dir / f"volume_{stem}.csv",
        "benchmarks": raw_dir / f"benchmarks_{stem}.csv",
        "market_cap": raw_dir / f"market_cap_{universe}.csv",
        "meta": raw_dir / f"download_log_{stem}.json",
    }


def _extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Pull one price field out of a ``yf.download`` frame, whatever its layout.

    yfinance has changed its column layout several times (flat for one ticker,
    (field, ticker) or (ticker, field) MultiIndex for many).  This helper is
    deliberately tolerant so that the pipeline does not break on an upgrade.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=tickers, dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            if field in raw.columns.get_level_values(level):
                out = raw.xs(field, axis=1, level=level)
                if isinstance(out, pd.Series):
                    out = out.to_frame(tickers[0])
                return out
        return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)
    if field in raw.columns:
        return raw[[field]].rename(columns={field: tickers[0]})
    return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    batch_size: int = 50,
    pause_seconds: float = 1.0,
    max_retries: int = 3,
) -> dict[str, pd.DataFrame]:
    """Download Adj Close / Close / Volume for ``tickers`` in polite batches."""
    import yfinance as yf

    fields = {"adj_close": "Adj Close", "close": "Close", "volume": "Volume"}
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in fields}

    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    for b, batch in enumerate(batches, 1):
        raw = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False,  # keep the explicit "Adj Close" column
                    actions=False,
                    group_by="column",
                    threads=True,
                    progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"[prices] batch {b}/{len(batches)} attempt {attempt} failed: {exc}")
            time.sleep(pause_seconds * 3 * attempt)
        got = 0
        for key, field in fields.items():
            frame = _extract_field(raw, field, batch).reindex(columns=batch)
            parts[key].append(frame)
            if key == "adj_close":
                got = int(frame.notna().any().sum())
        print(f"[prices] batch {b}/{len(batches)}: {got}/{len(batch)} tickers returned data")
        time.sleep(pause_seconds)

    out = {}
    for key in fields:
        df = pd.concat(parts[key], axis=1).sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        out[key] = df.reindex(columns=tickers)
    return out


def download_market_caps(tickers: list[str], pause_seconds: float = 0.15) -> pd.DataFrame:
    """Best-effort current market capitalisation (INR) for each ticker.

    Used only to cross-reference network centrality with economic size
    (Analysis B).  Failures are recorded as NaN rather than raised.
    """
    import yfinance as yf

    rows = []
    for i, t in enumerate(tickers, 1):
        cap = np.nan
        for attempt in range(2):
            try:
                fi = yf.Ticker(t).fast_info
                cap = float(fi["marketCap"]) if fi["marketCap"] is not None else np.nan
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.0 + attempt)
        rows.append({"Ticker": t, "MarketCap": cap})
        if i % 50 == 0 or i == len(tickers):
            ok = sum(not np.isnan(r["MarketCap"]) for r in rows)
            print(f"[mcap] {i}/{len(tickers)} done ({ok} with a value)")
        time.sleep(pause_seconds)
    return pd.DataFrame(rows)


def fetch_all(cfg: dict, force: bool = False, skip_mcap: bool = False) -> dict[str, Path]:
    """Populate ``data/raw`` (constituents, prices, benchmarks, market caps)."""
    d = cfg["data"]
    raw_dir = Path(d["raw_dir"])
    paths = price_paths(raw_dir, d["universe"], d["start"], d["end"], d["interval"])

    members = load_constituents(raw_dir, d["universe"], force=force)
    tickers = members["Ticker"].tolist()

    if paths["adj_close"].exists() and not force:
        print(f"[prices] cache hit -> {paths['adj_close'].name} (use --force to refresh)")
    else:
        frames = download_prices(
            tickers, d["start"], d["end"], d["interval"], d["batch_size"], d["pause_seconds"]
        )
        for key, df in frames.items():
            df.to_csv(paths[key], float_format="%.6f")
        import yfinance as yf

        meta = {
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "yfinance_version": yf.__version__,
            "universe": d["universe"],
            "start": d["start"],
            "end_exclusive": d["end"],
            "interval": d["interval"],
            "tickers_requested": len(tickers),
            "tickers_with_data": int(frames["adj_close"].notna().any().sum()),
            "rows": int(len(frames["adj_close"])),
            "first_date": str(frames["adj_close"].index.min().date()) if len(frames["adj_close"]) else None,
            "last_date": str(frames["adj_close"].index.max().date()) if len(frames["adj_close"]) else None,
        }
        paths["meta"].write_text(json.dumps(meta, indent=2))
        print(f"[prices] saved {meta['tickers_with_data']} tickers x {meta['rows']} rows")

    if paths["benchmarks"].exists() and not force:
        print(f"[bench]  cache hit -> {paths['benchmarks'].name}")
    else:
        bench = d.get("benchmark_tickers", {})
        if bench:
            frames = download_prices(list(bench.values()), d["start"], d["end"], d["interval"], 10, 0.5)
            px = frames["adj_close"]
            if px.isna().all().all():
                px = frames["close"]
            px = px.rename(columns={v: k for k, v in bench.items()})
            px.to_csv(paths["benchmarks"], float_format="%.6f")
            print(f"[bench]  saved {list(px.columns)}")

    if skip_mcap:
        print("[mcap]   skipped")
    elif paths["market_cap"].exists() and not force:
        print(f"[mcap]   cache hit -> {paths['market_cap'].name}")
    else:
        download_market_caps(tickers).to_csv(paths["market_cap"], index=False)

    return paths


# --------------------------------------------------------------------------- #
# Loading cached data (no network access below this line)
# --------------------------------------------------------------------------- #
def load_cached(cfg: dict) -> dict[str, pd.DataFrame]:
    """Load everything ``fetch_all`` stored.  Raises if prices are missing."""
    d = cfg["data"]
    raw_dir = Path(d["raw_dir"])
    paths = price_paths(raw_dir, d["universe"], d["start"], d["end"], d["interval"])
    if not paths["adj_close"].exists():
        raise FileNotFoundError(
            f"{paths['adj_close']} not found. Run `python scripts/01_download_data.py` first."
        )

    def _read(p: Path) -> pd.DataFrame:
        return pd.read_csv(p, index_col=0, parse_dates=True)

    out = {
        "members": _clean_constituents(pd.read_csv(constituents_path(raw_dir, d["universe"]))),
        "adj_close": _read(paths["adj_close"]),
    }
    for key in ("close", "volume", "benchmarks"):
        if paths[key].exists():
            out[key] = _read(paths[key])
    if paths["market_cap"].exists():
        out["market_cap"] = pd.read_csv(paths["market_cap"])
    if paths["meta"].exists():
        out["meta"] = json.loads(paths["meta"].read_text())
    return out
