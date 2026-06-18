"""topics/research.py — financial-statement transforms for the Research tab.

Pure functions over yfinance statement DataFrames (columns = periods, newest
first). Streamlit-free and unit-testable.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from data_engine import latest_value


def yoy_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Period-over-period % change for each line item. yfinance columns are
    newest-first, so column i grows relative to column i+1 (the older period)."""
    if df is None or df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    cols = list(df.columns)
    out: Dict[str, pd.Series] = {}
    for i in range(len(cols) - 1):
        newer, older = cols[i], cols[i + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = df[newer] / df[older].replace(0, np.nan) - 1
        out[str(newer)[:10]] = growth
    return pd.DataFrame(out, index=df.index)


def common_size(df: pd.DataFrame, *base_labels: str) -> pd.DataFrame:
    """Express each line as a fraction of a base row (Total Revenue / Total
    Assets), per period. Returns empty if no base row is found."""
    if df is None or df.empty:
        return pd.DataFrame()
    idx = {str(i).lower(): i for i in df.index}
    base = None
    for cand in base_labels:                       # exact match first
        if cand.lower() in idx:
            base = df.loc[idx[cand.lower()]]
            break
    if base is None:                               # then contains-match
        for cand in base_labels:
            for low, orig in idx.items():
                if cand.lower() in low:
                    base = df.loc[orig]
                    break
            if base is not None:
                break
    if base is None:
        return pd.DataFrame()
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = df.divide(base.replace(0, np.nan), axis=1)
    cs.columns = [str(c)[:10] for c in cs.columns]
    return cs


def key_ratios(income: pd.DataFrame, balance: pd.DataFrame,
               cashflow: pd.DataFrame = None) -> Dict[str, float]:
    """Headline profitability / liquidity / leverage ratios from the latest period."""
    def L(df, *cands) -> float:
        v = latest_value(df, *cands)
        return v if v is not None else float("nan")

    def div(a: float, b: float) -> float:
        return a / b if (np.isfinite(a) and np.isfinite(b) and b != 0) else float("nan")

    rev = L(income, "Total Revenue", "Revenue")
    cogs = L(income, "Cost Of Revenue", "Cost of Revenue")
    opinc = L(income, "Operating Income", "EBIT")
    ni = L(income, "Net Income")
    ta = L(balance, "Total Assets")
    ca = L(balance, "Current Assets", "Total Current Assets")
    cl = L(balance, "Current Liabilities", "Total Current Liabilities")
    ltd = L(balance, "Long Term Debt")
    eq = L(balance, "Stockholders Equity", "Total Stockholder Equity",
           "Common Stock Equity")

    gross = div(rev - cogs, rev) if (np.isfinite(rev) and np.isfinite(cogs)) else float("nan")
    return {
        "Gross margin": gross,
        "Operating margin": div(opinc, rev),
        "Net margin": div(ni, rev),
        "Current ratio": div(ca, cl),
        "Debt/Equity": div(ltd, eq),
        "ROA": div(ni, ta),
        "ROE": div(ni, eq),
    }


if __name__ == "__main__":
    cols = ["2025", "2024"]
    inc = pd.DataFrame({cols[0]: [1000, 600, 200, 150], cols[1]: [800, 500, 150, 100]},
                       index=["Total Revenue", "Cost Of Revenue",
                              "Operating Income", "Net Income"])
    print("YoY:\n", yoy_growth(inc).round(3))
    print("common-size:\n", common_size(inc, "Total Revenue").round(3))
