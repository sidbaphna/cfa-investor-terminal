"""topics/fixed_income.py — Fixed Income & Macro (CFA L2).

Three things the spec asks for:

  1. Bond mechanics: price, Macaulay/modified duration, convexity, and the
     duration+convexity estimate of price change for a yield shock.
  2. Arbitrage-free valuation: bootstrap spot rates from par yields and value
     a cash-flow stream off the spot curve (no-arbitrage price).
  3. Corporate mapping: take a company's balance-sheet debt + interest expense,
     infer an average coupon/maturity, and translate a rate move into an
     estimated change in the value of its debt (duration risk), plus simple
     credit metrics (interest coverage, debt/EBITDA) and a rough rating bucket.

Pure functions; the corporate mapper pulls from data_engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from data_engine import (Statements, get_fundamentals, get_statements,
                         latest_value)


# --------------------------------------------------------------------------
# Bond mechanics
# --------------------------------------------------------------------------
def bond_price(face: float, coupon_rate: float, ytm: float, years: float,
               freq: int = 2) -> float:
    """Price of a level-coupon bond. Rates are annual fractions."""
    if years <= 0 or freq < 1:
        return float("nan")
    n = int(round(years * freq))
    c = face * coupon_rate / freq
    y = ytm / freq
    if y == 0:
        return c * n + face
    pv_coupons = c * (1 - (1 + y) ** -n) / y
    pv_face = face / (1 + y) ** n
    return pv_coupons + pv_face


@dataclass
class DurationResult:
    price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float

    def price_change_pct(self, dy: float) -> float:
        """ΔP/P ≈ −D_mod·Δy + 0.5·Convexity·Δy²  (Δy in decimal, e.g. 0.01)."""
        return -self.modified_duration * dy + 0.5 * self.convexity * dy ** 2


def bond_duration(face: float, coupon_rate: float, ytm: float, years: float,
                  freq: int = 2) -> DurationResult:
    """Macaulay & modified duration plus convexity from discounted cash flows."""
    if years <= 0 or freq < 1:
        return DurationResult(float("nan"), float("nan"), float("nan"), float("nan"))
    n = int(round(years * freq))
    c = face * coupon_rate / freq
    y = ytm / freq

    price = 0.0
    weighted_t = 0.0      # Σ t·PV(cf)   (t in periods)
    conv_sum = 0.0        # Σ t(t+1)·PV(cf)
    for t in range(1, n + 1):
        cf = c + (face if t == n else 0.0)
        pv = cf / (1 + y) ** t
        price += pv
        weighted_t += t * pv
        conv_sum += t * (t + 1) * pv

    if price <= 0:
        return DurationResult(price, float("nan"), float("nan"), float("nan"))

    mac_periods = weighted_t / price
    mac_years = mac_periods / freq
    mod_dur = mac_years / (1 + y)                       # modified duration (years)
    convexity = conv_sum / (price * (1 + y) ** 2 * freq ** 2)  # annualized
    return DurationResult(price, mac_years, mod_dur, convexity)


# --------------------------------------------------------------------------
# Arbitrage-free valuation off the spot curve
# --------------------------------------------------------------------------
def bootstrap_spot_rates(par_yields: List[float], freq: int = 1) -> List[float]:
    """Bootstrap annual spot (zero) rates from a list of par yields.

    par_yields[i] is the par yield for maturity (i+1) periods (annual fractions).
    Assumes par bonds priced at 100. Returns spot rates per period-index.
    """
    spots: List[float] = []
    for i, c in enumerate(par_yields):
        n = i + 1
        coupon = c / freq * 100.0
        if n == 1:
            spots.append((coupon + 100.0) / 100.0 - 1.0)
            continue
        pv_coupons = sum(coupon / (1 + spots[k]) ** (k + 1) for k in range(n - 1))
        residual = 100.0 - pv_coupons
        if residual <= 0:
            spots.append(spots[-1])           # degrade gracefully
            continue
        spot = ((coupon + 100.0) / residual) ** (1.0 / n) - 1.0
        spots.append(spot)
    return spots


def arbitrage_free_value(cashflows: List[float], spot_rates: List[float]) -> float:
    """PV a cash-flow stream off the spot curve. cashflows[t] occurs at period t+1.
    The no-arbitrage price: any deviation implies a riskless profit."""
    if not cashflows or len(spot_rates) < len(cashflows):
        return float("nan")
    pv = 0.0
    for t, cf in enumerate(cashflows):
        pv += cf / (1 + spot_rates[t]) ** (t + 1)
    return pv


# --------------------------------------------------------------------------
# Credit analysis
# --------------------------------------------------------------------------
def credit_rating_bucket(interest_coverage: float, debt_to_ebitda: float) -> str:
    """Very rough mapping of two ratios to an indicative rating band.
    Heuristic only — not a substitute for an agency model."""
    ic, de = interest_coverage, debt_to_ebitda
    if not np.isfinite(ic) or not np.isfinite(de):
        return "n/a"
    if ic >= 12 and de <= 1.5:
        return "AAA/AA (investment grade — very strong)"
    if ic >= 6 and de <= 2.5:
        return "A (investment grade — strong)"
    if ic >= 3 and de <= 3.5:
        return "BBB (investment grade — adequate)"
    if ic >= 2 and de <= 4.5:
        return "BB (speculative)"
    if ic >= 1 and de <= 6:
        return "B (highly speculative)"
    return "CCC or below (substantial credit risk)"


# --------------------------------------------------------------------------
# Corporate balance-sheet -> rate risk mapper
# --------------------------------------------------------------------------
@dataclass
class DebtRiskProfile:
    ticker: str
    total_debt: float = float("nan")
    interest_expense: float = float("nan")
    implied_coupon: float = float("nan")
    assumed_maturity_years: float = float("nan")
    modified_duration: float = float("nan")
    convexity: float = float("nan")
    dv01: float = float("nan")            # $ change in debt value per 1bp move
    value_change_on_shock: float = float("nan")  # $ change for the given shock
    ebit: float = float("nan")
    ebitda: float = float("nan")
    interest_coverage: float = float("nan")
    debt_to_ebitda: float = float("nan")
    rating_bucket: str = "n/a"
    notes: str = ""


def analyze_debt_structure(symbol: str, rate_shock_bps: float = 100.0,
                           assumed_maturity_years: float = 7.0,
                           statements: Optional[Statements] = None
                           ) -> DebtRiskProfile:
    """Map a company's debt to interest-rate sensitivity + credit metrics.

    Approach: treat total debt as one synthetic bullet bond. Imply its coupon
    from interest_expense/total_debt, assume a maturity, compute modified
    duration & convexity, and translate the rate shock into a $ value change.
    """
    s = statements or get_statements(symbol)
    f = get_fundamentals(symbol)

    total_debt = f.total_debt
    interest_expense = latest_value(s.income, "Interest Expense",
                                    "InterestExpense") or float("nan")
    if np.isfinite(interest_expense):
        interest_expense = abs(interest_expense)
    ebit = latest_value(s.income, "EBIT", "Operating Income",
                        "OperatingIncome") or float("nan")
    ebitda = f.ebitda if np.isfinite(f.ebitda) else latest_value(s.income, "EBITDA")

    profile = DebtRiskProfile(
        ticker=symbol.strip().upper(),
        total_debt=total_debt,
        interest_expense=interest_expense,
        assumed_maturity_years=assumed_maturity_years,
        ebit=ebit if ebit is not None else float("nan"),
        ebitda=ebitda if ebitda is not None else float("nan"),
    )

    if not np.isfinite(total_debt) or total_debt <= 0:
        profile.notes = "No usable total-debt figure from yfinance."
        return profile

    implied_coupon = (interest_expense / total_debt
                      if np.isfinite(interest_expense) and interest_expense > 0
                      else 0.05)
    implied_coupon = float(min(max(implied_coupon, 0.005), 0.20))
    profile.implied_coupon = implied_coupon

    # Synthetic bond: par = total_debt, priced at its own coupon (≈par), shocked.
    dur = bond_duration(face=total_debt, coupon_rate=implied_coupon,
                        ytm=implied_coupon, years=assumed_maturity_years, freq=2)
    profile.modified_duration = dur.modified_duration
    profile.convexity = dur.convexity

    if np.isfinite(dur.modified_duration) and np.isfinite(dur.price):
        dy = rate_shock_bps / 10_000.0
        profile.value_change_on_shock = dur.price * dur.price_change_pct(dy)
        profile.dv01 = dur.price * dur.price_change_pct(0.0001)

    # Credit metrics
    if profile.ebit is not None and np.isfinite(profile.ebit) \
            and np.isfinite(interest_expense) and interest_expense > 0:
        profile.interest_coverage = profile.ebit / interest_expense
    if np.isfinite(profile.ebitda) and profile.ebitda > 0:
        profile.debt_to_ebitda = total_debt / profile.ebitda
    profile.rating_bucket = credit_rating_bucket(profile.interest_coverage,
                                                 profile.debt_to_ebitda)
    return profile


if __name__ == "__main__":
    d = bond_duration(face=100, coupon_rate=0.05, ytm=0.06, years=10, freq=2)
    print(f"price={d.price:.2f} macD={d.macaulay_duration:.2f} "
          f"modD={d.modified_duration:.2f} conv={d.convexity:.2f}")
    print("ΔP/P for +100bp:", round(d.price_change_pct(0.01) * 100, 2), "%")
    spots = bootstrap_spot_rates([0.03, 0.035, 0.04])
    print("spot rates:", [round(x, 4) for x in spots])
