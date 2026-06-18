"""topics/portfolio_mgmt.py — Portfolio Construction & Active Risk (CFA L2).

  * compute_beta(...)        — market-model OLS: beta, alpha, R²
  * idiosyncratic_risk(...)  — annualized residual (firm-specific) volatility
  * tracking_error(...)      — annualized std of active returns vs a benchmark
  * information_ratio(...)   — active return / tracking error
  * sharpe_ratio(...)        — excess return / total volatility
  * two_factor_model(...)    — market + size factor regression (multifactor)
  * risk_summary(...)        — one call that assembles the whole risk picture
  * portfolio_active_stats() — holdings-weighted active risk vs a benchmark

Returns over a window come from data_engine; the math is plain numpy so it's
easy to unit-test with synthetic series.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import TRADING_DAYS
from data_engine import get_returns, get_risk_free_rate


# --------------------------------------------------------------------------
# core regression building block
# --------------------------------------------------------------------------
def _align(a: pd.Series, b: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    if df.empty:
        return np.array([]), np.array([])
    return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()


@dataclass
class BetaResult:
    beta: float
    alpha_daily: float
    alpha_annual: float
    r_squared: float
    n_obs: int


def compute_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> BetaResult:
    """OLS of asset on benchmark daily returns: r_a = alpha + beta·r_m + e."""
    y, x = _align(asset_returns, benchmark_returns)
    if y.size < 3:
        return BetaResult(float("nan"), float("nan"), float("nan"), float("nan"), int(y.size))
    var_m = np.var(x, ddof=1)
    if var_m == 0:
        return BetaResult(float("nan"), float("nan"), float("nan"), float("nan"), int(y.size))
    beta = np.cov(y, x, ddof=1)[0, 1] / var_m
    alpha = y.mean() - beta * x.mean()
    corr = np.corrcoef(y, x)[0, 1]
    return BetaResult(float(beta), float(alpha), float(alpha * TRADING_DAYS),
                      float(corr ** 2), int(y.size))


def idiosyncratic_risk(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized std of regression residuals (firm-specific / diversifiable risk)."""
    y, x = _align(asset_returns, benchmark_returns)
    if y.size < 3:
        return float("nan")
    var_m = np.var(x, ddof=1)
    if var_m == 0:
        return float("nan")
    beta = np.cov(y, x, ddof=1)[0, 1] / var_m
    alpha = y.mean() - beta * x.mean()
    resid = y - (alpha + beta * x)
    return float(np.std(resid, ddof=1) * np.sqrt(TRADING_DAYS))


def tracking_error(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized std of active returns (asset − benchmark)."""
    y, x = _align(asset_returns, benchmark_returns)
    if y.size < 2:
        return float("nan")
    active = y - x
    return float(np.std(active, ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized active return / tracking error."""
    y, x = _align(asset_returns, benchmark_returns)
    if y.size < 2:
        return float("nan")
    active = y - x
    te = np.std(active, ddof=1) * np.sqrt(TRADING_DAYS)
    if te == 0:
        return float("nan")
    return float(active.mean() * TRADING_DAYS / te)


def sharpe_ratio(asset_returns: pd.Series, rf_annual: Optional[float] = None) -> float:
    """Annualized Sharpe ratio."""
    r = asset_returns.dropna().to_numpy()
    if r.size < 2:
        return float("nan")
    rf = get_risk_free_rate() if rf_annual is None else rf_annual
    excess_daily = r.mean() - rf / TRADING_DAYS
    vol = np.std(r, ddof=1)
    if vol == 0:
        return float("nan")
    return float(excess_daily / vol * np.sqrt(TRADING_DAYS))


def annualized_volatility(asset_returns: pd.Series) -> float:
    r = asset_returns.dropna().to_numpy()
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))


# --------------------------------------------------------------------------
# Multifactor (market + size)
# --------------------------------------------------------------------------
@dataclass
class FactorResult:
    factor_betas: Dict[str, float]
    alpha_annual: float
    r_squared: float
    n_obs: int


def two_factor_model(asset_returns: pd.Series,
                     factor_returns: Dict[str, pd.Series]) -> FactorResult:
    """Generic multi-factor OLS via least squares. Pass any number of factors
    (e.g. {'MKT': spy_ret, 'SIZE': iwm_minus_spy})."""
    names = list(factor_returns.keys())
    frames = [asset_returns.rename("y")] + [
        factor_returns[n].rename(n) for n in names]
    df = pd.concat(frames, axis=1, join="inner").dropna()
    if df.shape[0] < len(names) + 2:
        return FactorResult({n: float("nan") for n in names}, float("nan"),
                            float("nan"), int(df.shape[0]))
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[n].to_numpy() for n in names])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    betas = {n: float(coef[i + 1]) for i, n in enumerate(names)}
    return FactorResult(betas, float(coef[0] * TRADING_DAYS), float(r2), int(df.shape[0]))


# --------------------------------------------------------------------------
# Convenience: full single-name risk picture
# --------------------------------------------------------------------------
@dataclass
class RiskSummary:
    ticker: str
    benchmark: str
    period: str
    beta: float = float("nan")
    alpha_annual: float = float("nan")
    r_squared: float = float("nan")
    idiosyncratic_vol: float = float("nan")
    total_vol: float = float("nan")
    tracking_error: float = float("nan")
    information_ratio: float = float("nan")
    sharpe: float = float("nan")
    n_obs: int = 0


def risk_summary(symbol: str, benchmark: str = "SPY",
                 period: str = "1y") -> RiskSummary:
    asset = get_returns(symbol, period=period)
    bench = get_returns(benchmark, period=period)
    b = compute_beta(asset, bench)
    return RiskSummary(
        ticker=symbol.strip().upper(), benchmark=benchmark.upper(), period=period,
        beta=b.beta, alpha_annual=b.alpha_annual, r_squared=b.r_squared,
        idiosyncratic_vol=idiosyncratic_risk(asset, bench),
        total_vol=annualized_volatility(asset),
        tracking_error=tracking_error(asset, bench),
        information_ratio=information_ratio(asset, bench),
        sharpe=sharpe_ratio(asset),
        n_obs=b.n_obs,
    )


def portfolio_active_stats(weights: Dict[str, float], benchmark: str = "SPY",
                           period: str = "1y") -> Dict[str, float]:
    """Build a weighted portfolio return series and measure active risk vs bench.
    weights: {ticker: weight}; weights are renormalized to sum to 1."""
    tickers = [t for t in weights if weights[t] > 0]
    if not tickers:
        return {}
    total_w = sum(weights[t] for t in tickers)
    series = []
    used = []
    for t in tickers:
        r = get_returns(t, period=period)
        if not r.empty:
            series.append(r.rename(t) * (weights[t] / total_w))
            used.append(t)
    if not series:
        return {}
    port = pd.concat(series, axis=1, join="inner").dropna().sum(axis=1)
    bench = get_returns(benchmark, period=period)
    return {
        "beta": compute_beta(port, bench).beta,
        "tracking_error": tracking_error(port, bench),
        "information_ratio": information_ratio(port, bench),
        "total_vol": annualized_volatility(port),
        "sharpe": sharpe_ratio(port),
        "n_holdings": float(len(used)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0.0004, 0.01, 252))
    a = 1.2 * m + pd.Series(rng.normal(0.0001, 0.005, 252))
    print("beta:", round(compute_beta(a, m).beta, 3))
    print("TE:", round(tracking_error(a, m), 4))
    print("IR:", round(information_ratio(a, m), 3))
