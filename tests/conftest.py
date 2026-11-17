import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nsenet.network import correlation_matrix  # noqa: E402
from nsenet.synthetic import factor_model_returns  # noqa: E402


@pytest.fixture(scope="session")
def synthetic():
    returns, sectors = factor_model_returns(n_sectors=6, stocks_per_sector=15, n_days=600, seed=1)
    return returns, sectors, correlation_matrix(returns)
