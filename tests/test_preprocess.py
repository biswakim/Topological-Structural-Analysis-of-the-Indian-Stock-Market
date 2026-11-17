import numpy as np
import pandas as pd

from nsenet.pipeline import apply_overrides, audit_extreme_returns
from nsenet.preprocess import clean_prices, log_returns


def _prices():
    idx = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(0)
    data = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (100, 5)), axis=0))
    px = pd.DataFrame(data, index=idx, columns=list("ABCDE"))
    px.loc[idx[:40], "B"] = np.nan          # listed late
    px["C"] = np.nan                        # never traded
    px.loc[idx[50:52], "D"] = np.nan        # two-day gap (fillable)
    return px


def test_log_returns_definition():
    px = pd.DataFrame({"X": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    r = log_returns(px)
    assert np.allclose(r["X"].to_numpy(), [np.log(1.1), np.log(0.9)])


def test_clean_prices_drops_and_reports():
    clean, rep = clean_prices(_prices(), min_coverage=0.97, max_ffill_days=3)
    assert list(clean.columns) == ["A", "D", "E"]
    assert rep["dropped_no_data"] == ["C"]
    assert "B" in rep["dropped_low_coverage"]
    assert rep["cells_forward_filled"] == 2
    assert not clean.isna().any().any()


def test_override_masks_single_return():
    idx = pd.bdate_range("2024-01-01", periods=30)
    r = pd.DataFrame(0.001, index=idx, columns=["A.NS", "B.NS"])
    r.loc[idx[10], "A.NS"] = -1.5
    vol = pd.DataFrame(1000.0, index=idx, columns=r.columns)
    ov = [{"ticker": "A.NS", "date": str(idx[10].date()), "reason": "split"}]
    audit = audit_extreme_returns(r, vol, 0.2, ov)
    assert len(audit) == 1 and audit.loc[0, "classification"].startswith("mechanical")
    masked, n = apply_overrides(r, ov)
    assert n == 1 and masked.loc[idx[10], "A.NS"] == 0.0 and masked.loc[idx[10], "B.NS"] == 0.001
