#!/usr/bin/env python
"""Step 1 - download and cache everything the analysis needs.

    python scripts/01_download_data.py            # uses the cache when present
    python scripts/01_download_data.py --force    # re-download
    python scripts/01_download_data.py --skip-mcap

Outputs (all under data/raw/):
    ind_<universe>list.csv        official constituent list with industry labels
    adj_close_*.csv               adjusted close prices (dates x tickers)
    close_*.csv, volume_*.csv     unadjusted close and traded volume
    benchmarks_*.csv              Nifty 50 / Nifty 500 index levels
    market_cap_*.csv              current market capitalisation (best effort)
    download_log_*.json           provenance (timestamp, yfinance version, counts)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsenet.config import load_config  # noqa: E402
from nsenet.data import fetch_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to a config.yaml (default: repo root)")
    ap.add_argument("--force", action="store_true", help="ignore the cache and download again")
    ap.add_argument("--skip-mcap", action="store_true", help="do not fetch market capitalisations")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = fetch_all(cfg, force=args.force, skip_mcap=args.skip_mcap)
    print("\nDone. Files in", paths["adj_close"].parent)
    for k, p in paths.items():
        print(f"  {'ok ' if p.exists() else 'MISSING'}  {p.name}")


if __name__ == "__main__":
    main()
