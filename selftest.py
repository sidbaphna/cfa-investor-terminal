"""selftest.py — deterministic, network-free checks of the quant engines.

Run:  python selftest.py
Every assertion uses closed-form / hand-computed expected values so a regression
in any formula fails loudly. No yfinance calls.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from topics import derivatives as dv
from topics import equity_valuation as ev
from topics import fixed_income as fi
from topics import fsa
from topics import portfolio_mgmt as pm
from data_engine import Statements


def approx(a: float, b: float, tol: float = 1e-2) -> bool:
    return abs(a - b) <= tol


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'✅' if cond else '❌'} {name}" + (f"  {detail}" if detail else ""))
    assert cond, f"FAILED: {name} {detail}"


# --- Equity Valuation -----------------------------------------------------
def test_equity():
    ri = ev.residual_income_constant_growth(10.0, 0.15, 0.10, 0.04)
    check("RI constant growth = 18.33", approx(ri.intrinsic_value, 18.3333),
          f"got {ri.intrinsic_value:.4f}")

    # persistence=1, g=0  ->  perpetuity:  V0 = B0 + RI/r = 10 + 0.5/0.10 = 15.0
    # (this is the case that catches a continuing-value off-by-one)
    rim = ev.residual_income_multistage(10.0, 0.15, 0.10, g_high=0.0,
                                        n_high=5, persistence=1.0)
    check("RI multistage perpetuity = 15.00", approx(rim.intrinsic_value, 15.0),
          f"got {rim.intrinsic_value:.4f}")

    check("Gordon growth = 40", approx(ev.gordon_growth_value(2.0, 0.10, 0.05), 40.0))

    # constant-growth DDM split into explicit+terminal must equal Gordon (D1/(r-g))
    dd = ev.two_stage_ddm(2.0, 0.10, 0.05, 5, 0.05)
    check("Two-stage DDM telescopes to Gordon = 42",
          approx(dd.intrinsic_value, 2.0 * 1.05 / 0.05), f"got {dd.intrinsic_value:.4f}")

    # FCFF: firm value minus debt over shares
    f = ev.two_stage_fcff(100.0, 0.08, 0.05, 5, 0.03, total_debt=200.0, shares=50.0)
    check("FCFF returns finite per-share value", math.isfinite(f.intrinsic_value),
          f"got {f.intrinsic_value:.4f}")


# --- Fixed Income ---------------------------------------------------------
def test_fixed_income():
    check("Par bond prices to 100", approx(fi.bond_price(100, 0.05, 0.05, 10, 2), 100.0))
    d = fi.bond_duration(100, 0.05, 0.06, 10, 2)
    check("Macaulay > Modified duration", d.macaulay_duration > d.modified_duration)
    check("Convexity positive", d.convexity > 0)
    check("+100bp -> price falls", d.price_change_pct(0.01) < 0)
    check("-100bp -> price rises", d.price_change_pct(-0.01) > 0)

    spots = fi.bootstrap_spot_rates([0.03, 0.035, 0.04])
    check("Spot[0] == par[0] (1-period)", approx(spots[0], 0.03, 1e-6))
    check("Spot curve upward -> spot > par at long end", spots[-1] > 0.04)


# --- Derivatives / Greeks -------------------------------------------------
def test_greeks():
    S, K, T, r, sig = 100, 100, 0.5, 0.04, 0.25
    c = dv.bsm_price(S, K, T, r, sig, 0.0, "call")
    p = dv.bsm_price(S, K, T, r, sig, 0.0, "put")
    parity = c - p - (S - K * math.exp(-r * T))
    check("Put-call parity holds", approx(parity, 0.0, 1e-6), f"residual {parity:.2e}")

    gc = dv.bsm_greeks(S, K, T, r, sig, 0.0, "call")
    gp = dv.bsm_greeks(S, K, T, r, sig, 0.0, "put")
    check("ATM call delta in (0.4,0.7)", 0.4 < gc.delta < 0.7, f"{gc.delta:.3f}")
    check("Put delta negative", gp.delta < 0)
    check("Gamma positive & equal call/put", gc.gamma > 0 and approx(gc.gamma, gp.gamma, 1e-9))
    check("Vega positive", gc.vega_per_1pct > 0)
    check("Call theta/day negative", gc.theta_per_day < 0)
    # delta_call - delta_put == e^{-qT} == 1 (q=0)
    check("delta_call - delta_put == 1", approx(gc.delta - gp.delta, 1.0, 1e-6))


# --- Portfolio risk -------------------------------------------------------
def test_portfolio():
    rng = np.random.default_rng(42)
    m = pd.Series(rng.normal(0.0004, 0.01, 500))
    a = 1.2 * m + pd.Series(rng.normal(0.0, 0.004, 500))
    b = pm.compute_beta(a, m)
    check("Beta recovers ~1.2", 1.1 < b.beta < 1.3, f"{b.beta:.3f}")
    check("R^2 in (0,1)", 0 < b.r_squared < 1)
    check("Tracking error positive", pm.tracking_error(a, m) > 0)
    check("Idiosyncratic vol positive", pm.idiosyncratic_risk(a, m) > 0)
    check("Information ratio finite", math.isfinite(pm.information_ratio(a, m)))


# --- Beneish M-Score ------------------------------------------------------
def _flat_statements():
    """Two identical years -> all 8 indices = 1, TATA = 0 -> M = -2.48."""
    cols = ["2025", "2024"]
    income = pd.DataFrame({
        cols[0]: [1000, 600, 100, 120],
        cols[1]: [1000, 600, 100, 120],
    }, index=["Total Revenue", "Cost Of Revenue",
              "Selling General And Administration", "Net Income"])
    balance = pd.DataFrame({
        cols[0]: [200, 500, 300, 1500, 250, 400],
        cols[1]: [200, 500, 300, 1500, 250, 400],
    }, index=["Accounts Receivable", "Current Assets", "Net PPE",
              "Total Assets", "Current Liabilities", "Long Term Debt"])
    cashflow = pd.DataFrame({
        cols[0]: [50, 120],   # depreciation, operating cash flow (== net income)
        cols[1]: [50, 120],
    }, index=["Depreciation And Amortization", "Operating Cash Flow"])
    return Statements(income, balance, cashflow)


def test_beneish():
    res = fsa.beneish_m_score("TEST", statements=_flat_statements())
    expected = (-4.84 + 0.92 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 + 0 - 0.327)
    check("Beneish all-indices-1 => M = -2.48", approx(res.m_score, expected, 1e-6),
          f"got {res.m_score:.4f} vs {expected:.4f}")
    check("SGI == 1 for flat sales", approx(res.components["SGI"], 1.0, 1e-9))
    check("TATA == 0 when NI == CFO", approx(res.components["TATA"], 0.0, 1e-9))


# --- FX translation -------------------------------------------------------
def test_fx():
    fx = fsa.fx_translate(1000, 200, 400, 1.10, 1.12, 1.20, "temporal")
    # remeasurement = monetary*(current - historical) = 400*(1.10-1.20) = -40
    check("Temporal remeasurement = -40", approx(fx.adjustment, -40.0, 1e-6),
          f"got {fx.adjustment:.4f}")


if __name__ == "__main__":
    for fn in (test_equity, test_fixed_income, test_greeks, test_portfolio,
               test_beneish, test_fx):
        fn()
    print("\n🎉 All self-tests passed.")
