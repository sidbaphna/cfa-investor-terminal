"""topics/fsa.py — Financial Statement Analysis (CFA L2).

Spec asks for Pension, Intercorporate Investments, and Foreign Currency
Translation. We add the Beneish M-Score (earnings-manipulation detector) because
the ELI5 examples reference its DSRI component and it is fully computable from
two years of yfinance statements.

  * beneish_m_score(symbol)      — 8-factor forensic accounting model
  * fx_translate(...)            — current-rate vs temporal method + CTA
  * pension_funded_status(...)   — funded status & net periodic pension cost
  * classify_investment(pct)     — equity method vs consolidation thresholds

The M-Score is the data-driven one; the others are teaching calculators that
take explicit inputs (Study Mode), since multi-currency subsidiary data and
pension footnotes are not exposed by yfinance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_engine import Statements, get_statements


# --------------------------------------------------------------------------
# helpers: pull a line item for a given period index (0 = latest, 1 = prior)
# --------------------------------------------------------------------------
def _period_value(df: pd.DataFrame, period_idx: int, *candidates: str
                  ) -> Optional[float]:
    if df is None or df.empty or df.shape[1] <= period_idx:
        return None
    idx = {str(i).lower(): i for i in df.index}
    row_label = None
    for cand in candidates:
        if cand.lower() in idx:
            row_label = idx[cand.lower()]
            break
    if row_label is None:
        for cand in candidates:
            for low, original in idx.items():
                if cand.lower() in low:
                    row_label = original
                    break
            if row_label is not None:
                break
    if row_label is None:
        return None
    try:
        val = df.loc[row_label].iloc[period_idx]
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError, IndexError):
        return None


# --------------------------------------------------------------------------
# Beneish M-Score
# --------------------------------------------------------------------------
@dataclass
class MScoreResult:
    ticker: str
    m_score: float = float("nan")
    components: Dict[str, float] = field(default_factory=dict)
    likely_manipulator: bool = False
    notes: str = ""

    THRESHOLD: float = -1.78   # M > -1.78 → flagged


def beneish_m_score(symbol: str, statements: Optional[Statements] = None
                    ) -> MScoreResult:
    """8-variable Beneish model. Needs two fiscal years of data."""
    s = statements or get_statements(symbol)
    inc, bal, cf = s.income, s.balance, s.cashflow
    res = MScoreResult(ticker=symbol.strip().upper())

    if inc.empty or bal.empty or inc.shape[1] < 2 or bal.shape[1] < 2:
        res.notes = "Need ≥2 years of income + balance sheet data."
        return res

    def pv(df, i, *c):
        return _period_value(df, i, *c)

    # Period t (0) and t-1 (1)
    sales_t = pv(inc, 0, "Total Revenue", "TotalRevenue", "Revenue")
    sales_p = pv(inc, 1, "Total Revenue", "TotalRevenue", "Revenue")
    cogs_t = pv(inc, 0, "Cost Of Revenue", "Cost of Revenue", "CostOfRevenue")
    cogs_p = pv(inc, 1, "Cost Of Revenue", "Cost of Revenue", "CostOfRevenue")
    sga_t = pv(inc, 0, "Selling General And Administration",
               "Selling General And Administrative", "SellingGeneralAndAdministration")
    sga_p = pv(inc, 1, "Selling General And Administration",
               "Selling General And Administrative", "SellingGeneralAndAdministration")
    ni_t = pv(inc, 0, "Net Income", "NetIncome",
              "Net Income Continuous Operations")

    rec_t = pv(bal, 0, "Accounts Receivable", "Net Receivables", "Receivables")
    rec_p = pv(bal, 1, "Accounts Receivable", "Net Receivables", "Receivables")
    ca_t = pv(bal, 0, "Current Assets", "Total Current Assets")
    ca_p = pv(bal, 1, "Current Assets", "Total Current Assets")
    ppe_t = pv(bal, 0, "Net PPE", "Net Property Plant And Equipment",
               "Property Plant And Equipment Net")
    ppe_p = pv(bal, 1, "Net PPE", "Net Property Plant And Equipment",
               "Property Plant And Equipment Net")
    ta_t = pv(bal, 0, "Total Assets", "TotalAssets")
    ta_p = pv(bal, 1, "Total Assets", "TotalAssets")
    cl_t = pv(bal, 0, "Current Liabilities", "Total Current Liabilities")
    cl_p = pv(bal, 1, "Current Liabilities", "Total Current Liabilities")
    ltd_t = pv(bal, 0, "Long Term Debt", "LongTermDebt") or 0.0
    ltd_p = pv(bal, 1, "Long Term Debt", "LongTermDebt") or 0.0

    dep_t = pv(cf, 0, "Depreciation And Amortization", "Depreciation",
               "DepreciationAndAmortization")
    dep_p = pv(cf, 1, "Depreciation And Amortization", "Depreciation",
               "DepreciationAndAmortization")
    cfo_t = pv(cf, 0, "Operating Cash Flow", "Total Cash From Operating Activities",
               "OperatingCashFlow")

    def ratio(a, b):
        if a is None or b is None or b == 0:
            return None
        return a / b

    comp: Dict[str, float] = {}

    # DSRI — Days Sales in Receivables Index
    rs_t, rs_p = ratio(rec_t, sales_t), ratio(rec_p, sales_p)
    comp["DSRI"] = ratio(rs_t, rs_p) if rs_t and rs_p else None
    # GMI — Gross Margin Index
    gm_t = ratio(sales_t - cogs_t, sales_t) if sales_t and cogs_t is not None else None
    gm_p = ratio(sales_p - cogs_p, sales_p) if sales_p and cogs_p is not None else None
    comp["GMI"] = ratio(gm_p, gm_t) if gm_t and gm_p else None
    # AQI — Asset Quality Index
    aq_t = 1 - ratio((ca_t or 0) + (ppe_t or 0), ta_t) if ta_t else None
    aq_p = 1 - ratio((ca_p or 0) + (ppe_p or 0), ta_p) if ta_p else None
    comp["AQI"] = ratio(aq_t, aq_p) if aq_t is not None and aq_p else None
    # SGI — Sales Growth Index
    comp["SGI"] = ratio(sales_t, sales_p)
    # DEPI — Depreciation Index
    dr_t = ratio(dep_t, (dep_t or 0) + (ppe_t or 0)) if dep_t is not None else None
    dr_p = ratio(dep_p, (dep_p or 0) + (ppe_p or 0)) if dep_p is not None else None
    comp["DEPI"] = ratio(dr_p, dr_t) if dr_t and dr_p else None
    # SGAI — SG&A Index
    sg_t, sg_p = ratio(sga_t, sales_t), ratio(sga_p, sales_p)
    comp["SGAI"] = ratio(sg_t, sg_p) if sg_t and sg_p else None
    # LVGI — Leverage Index
    lev_t = ratio((cl_t or 0) + ltd_t, ta_t) if ta_t else None
    lev_p = ratio((cl_p or 0) + ltd_p, ta_p) if ta_p else None
    comp["LVGI"] = ratio(lev_t, lev_p) if lev_t and lev_p else None
    # TATA — Total Accruals to Total Assets
    comp["TATA"] = (ratio(ni_t - cfo_t, ta_t)
                    if ni_t is not None and cfo_t is not None and ta_t else None)

    # Defaults for missing indices: index variables → 1.0 (no change), TATA → 0.
    defaults = {"DSRI": 1.0, "GMI": 1.0, "AQI": 1.0, "SGI": 1.0, "DEPI": 1.0,
                "SGAI": 1.0, "LVGI": 1.0, "TATA": 0.0}
    missing = [k for k, v in comp.items() if v is None]
    filled = {k: (comp[k] if comp[k] is not None else defaults[k]) for k in comp}

    m = (-4.84 + 0.92 * filled["DSRI"] + 0.528 * filled["GMI"]
         + 0.404 * filled["AQI"] + 0.892 * filled["SGI"]
         + 0.115 * filled["DEPI"] - 0.172 * filled["SGAI"]
         + 4.679 * filled["TATA"] - 0.327 * filled["LVGI"])

    res.m_score = float(m)
    res.components = {k: (round(v, 4) if v is not None else float("nan"))
                     for k, v in comp.items()}
    res.likely_manipulator = m > res.THRESHOLD
    if missing:
        res.notes = ("Some inputs unavailable; defaulted "
                     + ", ".join(missing) + ". Interpret with caution.")
    return res


# --------------------------------------------------------------------------
# Foreign currency translation (Study Mode calculator)
# --------------------------------------------------------------------------
@dataclass
class FXTranslationResult:
    method: str
    translated: Dict[str, float]
    adjustment: float       # CTA (current-rate) or remeasurement gain (temporal)
    notes: str = ""


def fx_translate(net_assets_local: float, net_income_local: float,
                 monetary_net_assets_local: float,
                 current_rate: float, average_rate: float,
                 historical_rate: float, method: str = "current_rate"
                 ) -> FXTranslationResult:
    """Translate a subsidiary's results. All *_local in local currency.

    current_rate method (functional currency = local): assets/liabilities at the
    current rate, income at the average rate; the plug is the CTA in equity.

    temporal method (functional currency = parent): monetary items at the
    current rate, non-monetary at historical; the plug hits the income statement
    as a remeasurement gain/loss.
    """
    method = method.lower()
    if method == "current_rate":
        translated = {
            "net_assets": net_assets_local * current_rate,
            "net_income": net_income_local * average_rate,
        }
        # CTA = (NA·current) − (NA·historical) approximation for the period plug
        cta = net_assets_local * current_rate - net_assets_local * historical_rate
        return FXTranslationResult("Current-Rate (CTA in OCI)", translated, cta,
                                   notes="Exposure = entire net assets.")
    # temporal
    nonmonetary = net_assets_local - monetary_net_assets_local
    translated = {
        "monetary_net_assets": monetary_net_assets_local * current_rate,
        "nonmonetary_net_assets": nonmonetary * historical_rate,
        "net_income": net_income_local * average_rate,
    }
    remeasurement = (monetary_net_assets_local * current_rate
                     - monetary_net_assets_local * historical_rate)
    return FXTranslationResult("Temporal (remeasurement in NI)", translated,
                               remeasurement,
                               notes="Exposure = net monetary assets only.")


# --------------------------------------------------------------------------
# Pension
# --------------------------------------------------------------------------
@dataclass
class PensionResult:
    funded_status: float
    status_label: str
    net_periodic_cost: float = float("nan")


def pension_funded_status(pbo: float, plan_assets: float,
                          service_cost: float = float("nan"),
                          interest_cost: float = float("nan"),
                          expected_return: float = float("nan"),
                          amortization: float = 0.0) -> PensionResult:
    """Funded status = plan assets − PBO. Negative = underfunded (a liability)."""
    funded = plan_assets - pbo
    label = "Overfunded (net asset)" if funded >= 0 else "Underfunded (net liability)"
    npc = float("nan")
    if all(np.isfinite(x) for x in (service_cost, interest_cost, expected_return)):
        npc = service_cost + interest_cost - expected_return + amortization
    return PensionResult(funded, label, npc)


# --------------------------------------------------------------------------
# Intercorporate investments
# --------------------------------------------------------------------------
def classify_investment(ownership_pct: float, control: bool = False) -> str:
    """Standard accounting classification by ownership stake."""
    if control or ownership_pct > 0.50:
        return "Consolidation (control) — line-by-line, with NCI"
    if ownership_pct >= 0.20:
        return "Equity method (significant influence) — one-line consolidation"
    return "Financial asset (FVPL/FVOCI) — fair value, no significant influence"


if __name__ == "__main__":
    fx = fx_translate(1000, 200, 400, current_rate=1.10, average_rate=1.12,
                      historical_rate=1.20, method="temporal")
    print(fx.method, fx.translated, "remeasurement:", round(fx.adjustment, 2))
    print(classify_investment(0.25))
