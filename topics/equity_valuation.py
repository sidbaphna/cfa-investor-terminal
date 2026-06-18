"""topics/equity_valuation.py — Intrinsic-value models (CFA L2 Equity).

Implements four model families, all returning a uniform ValuationResult so the
UI can render any of them identically:

  * Gordon growth & two-stage DDM        (dividends)
  * Two-stage FCFE                        (cash flow to equity)
  * Two-stage FCFF                        (firm value -> equity per share)
  * Residual Income (single & multistage) (book value + economic profit)

Residual Income — the headline model — proves out the spec's formula:

    V0 = B0 + Σ_{t=1..∞} (E_t − r·B_{t-1}) / (1+r)^t

where RI_t = E_t − r·B_{t-1} = (ROE_t − r)·B_{t-1}. The constant-growth closed
form is V0 = B0 + (ROE − r)·B0 / (r − g); the multistage form forecasts RI
explicitly for T years and applies a persistence factor ω to the continuing
residual income.

All inputs are PER SHARE where it makes a per-share value comparable to price.
Functions are pure; `extract_equity_inputs` does best-effort data harvesting so
the UI can pre-fill (and let the user override) assumptions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from data_engine import Fundamentals, get_fundamentals, get_quote_metrics


# --------------------------------------------------------------------------
# Uniform result type
# --------------------------------------------------------------------------
@dataclass
class ValuationResult:
    model: str
    intrinsic_value: float                       # per share (NaN if not computable)
    current_price: float = float("nan")
    assumptions: Dict[str, float] = field(default_factory=dict)
    schedule: Optional[List[Dict[str, float]]] = None   # per-period detail
    notes: str = ""

    @property
    def upside_pct(self) -> float:
        if not np.isfinite(self.intrinsic_value) or not np.isfinite(self.current_price) \
                or self.current_price <= 0:
            return float("nan")
        return self.intrinsic_value / self.current_price - 1.0

    @property
    def ok(self) -> bool:
        return np.isfinite(self.intrinsic_value)


# --------------------------------------------------------------------------
# Dividend Discount Models
# --------------------------------------------------------------------------
def gordon_growth_value(d1: float, r: float, g: float) -> float:
    """V0 = D1 / (r - g). Requires r > g."""
    if not all(np.isfinite(x) for x in (d1, r, g)) or r <= g:
        return float("nan")
    return d1 / (r - g)


def two_stage_ddm(d0: float, r: float, g_high: float, n_high: int,
                  g_term: float, price: float = float("nan")) -> ValuationResult:
    """High-growth dividends for n_high years, then Gordon terminal at g_term."""
    if not all(np.isfinite(x) for x in (d0, r, g_high, g_term)) or r <= g_term \
            or n_high < 1:
        return ValuationResult("Two-Stage DDM", float("nan"), price,
                               notes="Invalid inputs (need r > terminal g, n≥1).")
    pv = 0.0
    schedule: List[Dict[str, float]] = []
    d = d0
    for t in range(1, n_high + 1):
        d *= (1 + g_high)
        disc = d / (1 + r) ** t
        pv += disc
        schedule.append({"t": t, "dividend": d, "pv": disc})
    d_term = d * (1 + g_term)
    tv = d_term / (r - g_term)
    pv_tv = tv / (1 + r) ** n_high
    schedule.append({"t": float(n_high), "terminal_value": tv, "pv": pv_tv})
    v0 = pv + pv_tv
    return ValuationResult(
        "Two-Stage DDM", v0, price,
        assumptions={"D0": d0, "r": r, "g_high": g_high,
                     "n_high": float(n_high), "g_term": g_term},
        schedule=schedule,
    )


# --------------------------------------------------------------------------
# FCFE / FCFF
# --------------------------------------------------------------------------
def two_stage_fcfe(fcfe0_ps: float, r: float, g_high: float, n_high: int,
                   g_term: float, price: float = float("nan")) -> ValuationResult:
    """Two-stage Free-Cash-Flow-to-Equity, per share. Discount at cost of equity r."""
    if not all(np.isfinite(x) for x in (fcfe0_ps, r, g_high, g_term)) \
            or r <= g_term or n_high < 1:
        return ValuationResult("Two-Stage FCFE", float("nan"), price,
                               notes="Invalid inputs (need r > terminal g, n≥1).")
    pv = 0.0
    schedule: List[Dict[str, float]] = []
    f = fcfe0_ps
    for t in range(1, n_high + 1):
        f *= (1 + g_high)
        disc = f / (1 + r) ** t
        pv += disc
        schedule.append({"t": t, "fcfe": f, "pv": disc})
    tv = f * (1 + g_term) / (r - g_term)
    pv_tv = tv / (1 + r) ** n_high
    schedule.append({"t": float(n_high), "terminal_value": tv, "pv": pv_tv})
    return ValuationResult(
        "Two-Stage FCFE", pv + pv_tv, price,
        assumptions={"FCFE0_ps": fcfe0_ps, "r": r, "g_high": g_high,
                     "n_high": float(n_high), "g_term": g_term},
        schedule=schedule,
    )


def two_stage_fcff(fcff0: float, wacc: float, g_high: float, n_high: int,
                   g_term: float, total_debt: float, shares: float,
                   price: float = float("nan")) -> ValuationResult:
    """Two-stage FCFF -> firm value -> equity value -> per share.
    fcff0/total_debt are firm-level $; result is per share."""
    if not all(np.isfinite(x) for x in (fcff0, wacc, g_high, g_term, shares)) \
            or wacc <= g_term or n_high < 1 or shares <= 0:
        return ValuationResult("Two-Stage FCFF", float("nan"), price,
                               notes="Invalid inputs (need WACC > terminal g, shares>0).")
    pv = 0.0
    schedule: List[Dict[str, float]] = []
    f = fcff0
    for t in range(1, n_high + 1):
        f *= (1 + g_high)
        disc = f / (1 + wacc) ** t
        pv += disc
        schedule.append({"t": t, "fcff": f, "pv": disc})
    tv = f * (1 + g_term) / (wacc - g_term)
    pv_tv = tv / (1 + wacc) ** n_high
    schedule.append({"t": float(n_high), "terminal_value": tv, "pv": pv_tv})
    firm_value = pv + pv_tv
    equity_value = firm_value - (total_debt if np.isfinite(total_debt) else 0.0)
    v0_ps = equity_value / shares
    return ValuationResult(
        "Two-Stage FCFF", v0_ps, price,
        assumptions={"FCFF0": fcff0, "WACC": wacc, "g_high": g_high,
                     "n_high": float(n_high), "g_term": g_term,
                     "total_debt": total_debt, "shares": shares,
                     "firm_value": firm_value, "equity_value": equity_value},
        schedule=schedule,
    )


# --------------------------------------------------------------------------
# Residual Income  (the spec's headline model)
# --------------------------------------------------------------------------
def residual_income_constant_growth(book0: float, roe: float, r: float,
                                    g: float, price: float = float("nan")
                                    ) -> ValuationResult:
    """Single-stage RI:  V0 = B0 + (ROE − r)·B0 / (r − g).  Requires r > g."""
    if not all(np.isfinite(x) for x in (book0, roe, r, g)) or r <= g:
        return ValuationResult("Residual Income (constant growth)", float("nan"),
                               price, notes="Invalid inputs (need r > g).")
    v0 = book0 + (roe - r) * book0 / (r - g)
    return ValuationResult(
        "Residual Income (constant growth)", v0, price,
        assumptions={"B0": book0, "ROE": roe, "r": r, "g": g,
                     "RI1": (roe - r) * book0},
        notes="V0 = B0 + (ROE − r)·B0 / (r − g)",
    )


def residual_income_multistage(book0: float, roe: float, r: float,
                               g_high: float, n_high: int,
                               persistence: float = 0.0,
                               price: float = float("nan")) -> ValuationResult:
    """Multistage RI with a persistence factor ω (0..1).

    Forecast clean-surplus book and residual income for n_high years:
        B_t  = B_{t-1}·(1 + g_high)          (clean-surplus, no new equity)
        RI_t = (ROE − r)·B_{t-1}
    Continuing value at year n uses ω:  TV = RI_{n+1} / (1 + r − ω),
    discounted n−1 periods (CFA convention: the year-n RI persists).
    ω=0 → residual income stops after the forecast; ω=1 → it persists forever.
    """
    if not all(np.isfinite(x) for x in (book0, roe, r, g_high)) or n_high < 1 \
            or not (0.0 <= persistence <= 1.0) or (1 + r - persistence) <= 0:
        return ValuationResult("Residual Income (multistage)", float("nan"),
                               price, notes="Invalid inputs.")
    pv_ri = 0.0
    schedule: List[Dict[str, float]] = []
    b_prev = book0
    last_ri = 0.0
    for t in range(1, n_high + 1):
        ri = (roe - r) * b_prev
        disc = ri / (1 + r) ** t
        pv_ri += disc
        schedule.append({"t": t, "book_begin": b_prev, "RI": ri, "pv": disc})
        last_ri = ri
        b_prev = b_prev * (1 + g_high)

    # Continuing residual income (persistence-weighted).
    # We forecast n_high explicit years above, so the continuing value begins at
    # RI_{n+1}, is valued as of time n, and is discounted by (1+r)^n.
    ri_next = last_ri * (1 + g_high)
    cont_value = ri_next / (1 + r - persistence)
    pv_cont = cont_value / (1 + r) ** n_high
    schedule.append({"t": float(n_high), "continuing_value": cont_value,
                     "pv": pv_cont})
    v0 = book0 + pv_ri + pv_cont
    return ValuationResult(
        "Residual Income (multistage)", v0, price,
        assumptions={"B0": book0, "ROE": roe, "r": r, "g_high": g_high,
                     "n_high": float(n_high), "persistence": persistence},
        schedule=schedule,
        notes="V0 = B0 + Σ RI_t/(1+r)^t + continuing value (ω-weighted)",
    )


# --------------------------------------------------------------------------
# Best-effort input harvesting (pre-fills the UI; user can override)
# --------------------------------------------------------------------------
@dataclass
class EquityInputs:
    ticker: str
    price: float
    book_value_ps: float
    eps_ttm: float
    dps: float
    roe: float
    fcfe_ps: float
    fcff: float
    total_debt: float
    shares: float
    g_high: float
    g_term: float
    r: float


def extract_equity_inputs(symbol: str, r: float = 0.09,
                          g_term: float = 0.025) -> EquityInputs:
    """Pull fundamentals and assemble sensible defaults for every model."""
    q = get_quote_metrics(symbol)
    f: Fundamentals = get_fundamentals(symbol)

    g_high = f.sustainable_growth
    if not np.isfinite(g_high):
        g_high = f.earnings_growth if np.isfinite(f.earnings_growth) else 0.08
    g_high = float(min(max(g_high, 0.0), 0.30))   # clamp to a believable band

    return EquityInputs(
        ticker=symbol.strip().upper(),
        price=q.last_price,
        book_value_ps=f.book_value_ps,
        eps_ttm=f.eps_ttm,
        dps=f.dps if np.isfinite(f.dps) else 0.0,
        roe=f.roe,
        fcfe_ps=f.fcfe_ps,
        fcff=f.free_cashflow,
        total_debt=f.total_debt if np.isfinite(f.total_debt) else 0.0,
        shares=f.shares_out,
        g_high=g_high,
        g_term=g_term,
        r=r,
    )


def value_all(inp: EquityInputs, n_high: int = 5,
              persistence: float = 0.6) -> Dict[str, ValuationResult]:
    """Run every model from one EquityInputs bundle (skipping the impossible)."""
    out: Dict[str, ValuationResult] = {}

    # DDM only meaningful for dividend payers
    if np.isfinite(inp.dps) and inp.dps > 0:
        out["ddm"] = two_stage_ddm(inp.dps, inp.r, inp.g_high, n_high,
                                   inp.g_term, inp.price)
    if np.isfinite(inp.fcfe_ps):
        out["fcfe"] = two_stage_fcfe(inp.fcfe_ps, inp.r, inp.g_high, n_high,
                                     inp.g_term, inp.price)
    if np.isfinite(inp.fcff) and np.isfinite(inp.shares):
        out["fcff"] = two_stage_fcff(inp.fcff, inp.r, inp.g_high, n_high,
                                     inp.g_term, inp.total_debt, inp.shares,
                                     inp.price)
    if np.isfinite(inp.book_value_ps) and np.isfinite(inp.roe):
        out["ri"] = residual_income_multistage(inp.book_value_ps, inp.roe, inp.r,
                                               inp.g_high, n_high, persistence,
                                               inp.price)
    return out


if __name__ == "__main__":
    ri = residual_income_constant_growth(book0=10.0, roe=0.15, r=0.10, g=0.04)
    print(ri.model, round(ri.intrinsic_value, 2))   # expect 18.33
    dd = two_stage_ddm(d0=2.0, r=0.10, g_high=0.08, n_high=5, g_term=0.03)
    print(dd.model, round(dd.intrinsic_value, 2))
