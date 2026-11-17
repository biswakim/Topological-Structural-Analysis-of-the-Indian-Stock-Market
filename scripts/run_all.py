#!/usr/bin/env python
"""Run the whole analysis (steps 2-9) from the cached data.

    python scripts/run_all.py            # everything except the download
    python scripts/run_all.py --download # also run step 1 first
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "02_build_network.py",
    "03_degree_distribution.py",
    "04_centrality.py",
    "05_clustering_homophily.py",
    "06_threshold_sensitivity.py",
    "07_market_mode.py",
    "08_stability.py",
    "09_export_report_assets.py",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="run 01_download_data.py first")
    args = ap.parse_args()
    steps = (["01_download_data.py"] if args.download else []) + STEPS
    for step in steps:
        t0 = time.time()
        print(f"\n{'=' * 70}\n>>> {step}\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, str(HERE / step)], check=True)
        print(f"<<< {step} finished in {time.time() - t0:.0f}s", flush=True)
    print("\nAll steps finished. Figures: results/figures  Tables: results/tables  Report assets: report/generated")


if __name__ == "__main__":
    main()
