"""topics/derivatives.py — Black-Scholes-Merton pricing and the Greeks.

Covers the 'greeks' the user asked to track. Greeks are computed analytically
from the BSM model with a continuous dividend yield q:

    d1 = [ln(S/K) + (r - q + 0.5*sigma^2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

Conventions returned (the way desks actually quote them):
    delta        : per $1 move in the underlying          (raw, -1..1)
    gamma        : change in delta per $1 move            (raw)
    vega_per_1pct: P&L per +1 percentage-point in IV      (vega/100)
    theta_per_day: time decay per calendar day            (annual theta/365)
    rho_per_1pct : P&L per +1 percentage-point in rates   (rho/100)

`implied_greeks_from_chain` pulls the live option chain (which already carries
Yahoo's impliedVolatility) and computes greeks for the near-ATM contract, so the
tracker can show a compact greeks readout per ticker without manual input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

from data_engine import OptionChain, get_option_chain, get_quote_metrics, get_risk_free_rate


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega_per_1pct: float
    theta_per_day: float
    rho_per_1pct: float
    # context
    spot: float
    strike: float
    days_to_expiry: float
    iv: float
    option_type: str  # "call" | "put"


def _years(days: float) -> float:
    return max(days, 0.0) / 365.0


def bsm_price(S: float, K: float, T: float, r: float, sigma: float,
              q: float = 0.0, option_type: str = "call") -> float:
    """Black-Scholes-Merton price. Falls back to discounted intrinsic at T<=0
    or sigma<=0 so callers never get a NaN/exception for edge inputs."""
    option_type = option_type.lower()
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return float(intrinsic)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return float(S * math.exp(-q * T) * norm.cdf(d1)
                     - K * math.exp(-r * T) * norm.cdf(d2))
    return float(K * math.exp(-r * T) * norm.cdf(-d2)
                 - S * math.exp(-q * T) * norm.cdf(-d1))


def bsm_greeks(S: float, K: float, T: float, r: float, sigma: float,
               q: float = 0.0, option_type: str = "call") -> Greeks:
    """Full greek set. T in YEARS, sigma & r & q as fractions."""
    option_type = option_type.lower()
    price = bsm_price(S, K, T, r, sigma, q, option_type)

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # Degenerate: delta is 0/1 by moneyness, other greeks ~0.
        itm = (S > K) if option_type == "call" else (S < K)
        delta = (1.0 if option_type == "call" else -1.0) if itm else 0.0
        return Greeks(price, delta, 0.0, 0.0, 0.0, 0.0, S, K,
                      T * 365.0, sigma, option_type)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = norm.pdf(d1)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT                      # per 1.00 vol

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-(S * disc_q * pdf_d1 * sigma) / (2 * sqrtT)
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1))           # per year
        rho = K * T * disc_r * norm.cdf(d2)                 # per 1.00 rate
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-(S * disc_q * pdf_d1 * sigma) / (2 * sqrtT)
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)

    return Greeks(
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        vega_per_1pct=float(vega / 100.0),
        theta_per_day=float(theta / 365.0),
        rho_per_1pct=float(rho / 100.0),
        spot=float(S), strike=float(K), days_to_expiry=float(T * 365.0),
        iv=float(sigma), option_type=option_type,
    )


def _nearest_atm_row(chain_df, spot: float):
    """Pick the row whose strike is closest to spot, with a usable IV."""
    if chain_df is None or chain_df.empty or "strike" not in chain_df:
        return None
    df = chain_df.copy()
    if "impliedVolatility" in df:
        df = df[df["impliedVolatility"].fillna(0) > 0]
    if df.empty:
        return None
    idx = (df["strike"] - spot).abs().idxmin()
    return df.loc[idx]


def implied_greeks_from_chain(symbol: str, option_type: str = "call",
                              expiry: Optional[str] = None) -> Optional[Greeks]:
    """Live near-ATM greeks straight from Yahoo's option chain.
    Returns None if no chain / IV is available."""
    q = get_quote_metrics(symbol)
    spot = q.last_price
    if not np.isfinite(spot) or spot <= 0:
        return None

    chain: OptionChain = get_option_chain(symbol, expiry)
    if chain.is_empty:
        return None
    df = chain.calls if option_type.lower() == "call" else chain.puts
    row = _nearest_atm_row(df, spot)
    if row is None:
        return None

    try:
        strike = float(row["strike"])
        iv = float(row["impliedVolatility"])
    except (KeyError, TypeError, ValueError):
        return None

    # days to expiry from the expiry string (YYYY-MM-DD)
    import datetime as _dt
    try:
        exp = _dt.date.fromisoformat(chain.expiry)
        days = max((exp - _dt.date.today()).days, 1)
    except (ValueError, TypeError):
        days = 30

    r = get_risk_free_rate()
    return bsm_greeks(spot, strike, _years(days), r, iv, q=0.0,
                      option_type=option_type)


if __name__ == "__main__":
    g = bsm_greeks(S=100, K=100, T=0.5, r=0.04, sigma=0.25, option_type="call")
    print(g)
