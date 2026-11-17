"""Maximum-likelihood fits of discrete heavy-tail candidates to a degree sequence.

Implements the workflow of Clauset, Shalizi & Newman (SIAM Review 2009) for
integer data, written from scratch so every step is inspectable:

1. fit a discrete power law to the tail k >= k_min by MLE,
2. choose k_min by minimising the Kolmogorov-Smirnov distance,
3. assess the fit with a semi-parametric bootstrap p-value,
4. compare against alternatives (exponential, log-normal, power law with
   exponential cut-off) using Vuong's normalised log-likelihood-ratio test.

A straight line through a log-log histogram is *not* evidence of a power law;
this module exists so that the report can make that statement quantitatively.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

_KCAP = 200_000  # upper summation limit for numerically normalised models


# --------------------------------------------------------------------------- #
# log-pmf of each candidate on the support k >= kmin
# --------------------------------------------------------------------------- #
def logpmf_powerlaw(k, alpha, kmin):
    return -alpha * np.log(k) - np.log(special.zeta(alpha, kmin))


def logpmf_exponential(k, lam, kmin):
    return np.log1p(-np.exp(-lam)) - lam * (k - kmin)


def logpmf_lognormal(k, mu, sigma, kmin):
    k = np.asarray(k, dtype=float)
    upper = stats.norm.cdf((np.log(k + 0.5) - mu) / sigma)
    lower = stats.norm.cdf((np.log(k - 0.5) - mu) / sigma)
    norm = stats.norm.sf((np.log(kmin - 0.5) - mu) / sigma)
    return np.log(np.clip(upper - lower, 1e-300, None)) - np.log(max(norm, 1e-300))


def logpmf_truncated_powerlaw(k, alpha, lam, kmin):
    grid = np.arange(kmin, _KCAP, dtype=float)
    logw = -alpha * np.log(grid) - lam * grid
    lognorm = special.logsumexp(logw)
    k = np.asarray(k, dtype=float)
    return -alpha * np.log(k) - lam * k - lognorm


# --------------------------------------------------------------------------- #
# MLE for each candidate
# --------------------------------------------------------------------------- #
@dataclass
class Fit:
    name: str
    params: dict
    loglik: float
    n: int
    kmin: int
    pointwise: np.ndarray  # log-likelihood of every tail observation

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik


def fit_powerlaw(tail: np.ndarray, kmin: int) -> Fit:
    def nll(a):
        return -logpmf_powerlaw(tail, a, kmin).sum()

    res = optimize.minimize_scalar(nll, bounds=(1.01, 12.0), method="bounded")
    a = float(res.x)
    pw = logpmf_powerlaw(tail, a, kmin)
    return Fit("power law", {"alpha": a}, float(pw.sum()), len(tail), kmin, pw)


def fit_exponential(tail: np.ndarray, kmin: int) -> Fit:
    excess = tail.mean() - kmin
    lam = float(np.log1p(1.0 / excess)) if excess > 0 else 50.0  # closed-form MLE
    pw = logpmf_exponential(tail, lam, kmin)
    return Fit("exponential", {"lambda": lam}, float(pw.sum()), len(tail), kmin, pw)


def fit_lognormal(tail: np.ndarray, kmin: int) -> Fit:
    logs = np.log(tail)
    x0 = np.array([logs.mean(), max(logs.std(), 0.3)])

    def nll(p):
        mu, sigma = p
        if sigma <= 1e-3:
            return np.inf
        return -logpmf_lognormal(tail, mu, sigma, kmin).sum()

    res = optimize.minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 4000})
    mu, sigma = map(float, res.x)
    pw = logpmf_lognormal(tail, mu, sigma, kmin)
    return Fit("log-normal", {"mu": mu, "sigma": sigma}, float(pw.sum()), len(tail), kmin, pw)


def fit_truncated_powerlaw(tail: np.ndarray, kmin: int) -> Fit:
    def nll(p):
        alpha, lam = p
        if lam <= 1e-8 or alpha < -5 or alpha > 12:
            return np.inf
        return -logpmf_truncated_powerlaw(tail, alpha, lam, kmin).sum()

    best = None
    for a0 in (0.5, 1.5, 2.5):
        res = optimize.minimize(nll, np.array([a0, 1.0 / max(tail.mean(), 1.0)]), method="Nelder-Mead",
                                options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    alpha, lam = map(float, best.x)
    pw = logpmf_truncated_powerlaw(tail, alpha, lam, kmin)
    return Fit("power law + cut-off", {"alpha": alpha, "lambda": lam}, float(pw.sum()), len(tail), kmin, pw)


# --------------------------------------------------------------------------- #
# k_min selection and goodness of fit for the power law
# --------------------------------------------------------------------------- #
def _ks_powerlaw(tail: np.ndarray, alpha: float, kmin: int) -> float:
    ks = np.arange(kmin, tail.max() + 1)
    emp = np.searchsorted(np.sort(tail), ks, side="right") / len(tail)
    model = 1.0 - special.zeta(alpha, ks + 1) / special.zeta(alpha, kmin)
    return float(np.abs(emp - model).max())


def select_kmin(degrees: np.ndarray, min_tail: int = 25) -> tuple[int, float, float]:
    """Return (k_min, alpha, KS) minimising the KS distance of the power law.

    ``min_tail`` keeps at least that many observations in the tail so the
    estimate is not driven by a handful of points.
    """
    k = np.asarray(degrees, dtype=int)
    k = k[k >= 1]
    best = (1, np.nan, np.inf)
    for kmin in np.unique(k):
        tail = k[k >= kmin]
        if len(tail) < min_tail or len(np.unique(tail)) < 3:
            break
        fit = fit_powerlaw(tail, int(kmin))
        ks = _ks_powerlaw(tail, fit.params["alpha"], int(kmin))
        if ks < best[2]:
            best = (int(kmin), fit.params["alpha"], ks)
    return best


def _sample_powerlaw(alpha: float, kmin: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-transform sampling from the discrete power law."""
    u = rng.random(size)
    kmax = kmin
    z0 = special.zeta(alpha, kmin)
    # grow the lookup table until it covers the largest quantile requested
    while special.zeta(alpha, kmax + 1) / z0 > (1 - u.max()) and kmax < 10**7:
        kmax = kmax * 2 + 10
    ks = np.arange(kmin, kmax + 2)
    cdf = 1.0 - special.zeta(alpha, ks + 1) / z0
    return ks[np.searchsorted(cdf, u, side="left").clip(max=len(ks) - 1)]


def powerlaw_gof(degrees: np.ndarray, n_boot: int = 300, min_tail: int = 25, seed: int = 0) -> dict:
    """Semi-parametric bootstrap goodness-of-fit test (CSN 2009, Sec. 4).

    p-value = fraction of synthetic data sets whose KS distance to their own
    best-fit power law exceeds the observed one.  p < 0.1 rejects the power law.
    """
    rng = np.random.default_rng(seed)
    k = np.asarray(degrees, dtype=int)
    k = k[k >= 1]
    kmin, alpha, ks_obs = select_kmin(k, min_tail)
    if not np.isfinite(ks_obs):
        return {"kmin": None, "alpha": np.nan, "ks": np.nan, "p_value": np.nan, "n_tail": 0}
    body = k[k < kmin]
    n, n_tail = len(k), int((k >= kmin).sum())
    exceed = 0
    for _ in range(n_boot):
        from_tail = rng.random(n) < n_tail / n
        n_t = int(from_tail.sum())
        synth_tail = _sample_powerlaw(alpha, kmin, n_t, rng)
        synth_body = rng.choice(body, size=n - n_t, replace=True) if len(body) and n - n_t > 0 else np.array([], int)
        synth = np.concatenate([synth_body, synth_tail])
        _, _, ks_b = select_kmin(synth, min_tail)
        exceed += ks_b >= ks_obs
    return {"kmin": kmin, "alpha": float(alpha), "ks": float(ks_obs), "p_value": exceed / n_boot,
            "n_tail": n_tail, "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
def vuong(f1: Fit, f2: Fit) -> tuple[float, float]:
    """Vuong's test.  Returns (R, p): R > 0 favours f1; p is two-sided."""
    d = f1.pointwise - f2.pointwise
    r = float(d.sum())
    sd = float(d.std(ddof=0)) * np.sqrt(len(d))
    if sd == 0:
        return r, 1.0
    return r, float(special.erfc(abs(r) / (np.sqrt(2) * sd)))


def fit_tail(degrees: np.ndarray, kmin: int) -> dict[str, Fit]:
    """Fit all four candidates to the tail k >= kmin."""
    k = np.asarray(degrees, dtype=int)
    tail = k[k >= kmin]
    return {
        "power law": fit_powerlaw(tail, kmin),
        "exponential": fit_exponential(tail, kmin),
        "log-normal": fit_lognormal(tail, kmin),
        "power law + cut-off": fit_truncated_powerlaw(tail, kmin),
    }


def comparison_table(fits: dict[str, Fit]):
    """One row per model: parameters, log-likelihood, AIC and the Vuong test
    of the pure power law against that model."""
    import pandas as pd

    pl = fits["power law"]
    best_aic = min(f.aic for f in fits.values())
    rows = []
    for name, f in fits.items():
        r, p = (np.nan, np.nan) if name == "power law" else vuong(pl, f)
        rows.append(
            {
                "model": name,
                "parameters": ", ".join(f"{k}={v:.3f}" for k, v in f.params.items()),
                "loglik": f.loglik,
                "aic": f.aic,
                "delta_aic": f.aic - best_aic,
                "llr_powerlaw_vs_model": r,
                "vuong_p": p,
            }
        )
    return pd.DataFrame(rows).set_index("model")


def tail_ccdf(fit: Fit, ks: np.ndarray) -> np.ndarray:
    """Model CCDF P(K >= k | K >= kmin) on the integer grid ``ks``."""
    kmin = fit.kmin
    ks = np.asarray(ks, dtype=int)
    if fit.name == "power law":
        a = fit.params["alpha"]
        return special.zeta(a, ks) / special.zeta(a, kmin)
    if fit.name == "exponential":
        return np.exp(-fit.params["lambda"] * (ks - kmin))
    if fit.name == "log-normal":
        mu, s = fit.params["mu"], fit.params["sigma"]
        return stats.norm.sf((np.log(ks - 0.5) - mu) / s) / stats.norm.sf((np.log(kmin - 0.5) - mu) / s)
    grid = np.arange(kmin, _KCAP, dtype=float)
    w = np.exp(-fit.params["alpha"] * np.log(grid) - fit.params["lambda"] * grid)
    cc = w[::-1].cumsum()[::-1] / w.sum()
    return cc[(ks - kmin).clip(0, len(cc) - 1)]


def naive_loglog_slope(degrees: np.ndarray, kmin: int = 1, bins_per_decade: int = 8) -> dict:
    """Ordinary least squares on the log-binned log-log histogram.

    This is the fit that a straight-edge on a log-log plot produces.  It is
    reported only so the write-up can compare it with the maximum-likelihood
    estimate: OLS on binned data has no statistical justification here (the
    bins are not independent, the errors are not Gaussian, and the fit is
    dominated by the many low-degree bins), and it routinely returns a
    plausible-looking exponent for data that are not power-law distributed.
    """
    from .metrics import log_binned_pdf

    lb = log_binned_pdf(np.asarray(degrees, dtype=int), bins_per_decade)
    lb = lb[(lb["k_mid"] >= kmin) & (lb["density"] > 0)]
    if len(lb) < 3:
        return {"slope": np.nan, "r_squared": np.nan, "n_bins": len(lb)}
    x = np.log10(lb["k_mid"].to_numpy())
    y = np.log10(lb["density"].to_numpy())
    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "alpha_implied": float(-res.slope),
        "intercept": float(res.intercept),
        "r_squared": float(res.rvalue**2),
        "n_bins": int(len(lb)),
        "kmin": int(kmin),
    }
