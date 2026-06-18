"""data_engine.py — Unified market-data harvester (yfinance).

One job: turn a ticker string into clean, typed Python objects/DataFrames that
every CFA subengine and the UI can consume. Nothing here imports Streamlit or
any topics/* module, so it stays a reusable, testable data layer.

Resilience contract
-------------------
* Every public function catches network/parse errors and returns an *empty but
  well-typed* result (empty DataFrame, None, or a QuoteMetrics with NaNs) so the
  UI can render "n/a" instead of crashing.
* A small in-process TTL cache avoids hammering Yahoo when the app re-runs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - import-time guard
    yf = None  # type: ignore

from config import FALLBACK_RISK_FREE, RISK_FREE_TICKER, TRADING_DAYS


# --------------------------------------------------------------------------
# Tiny TTL cache (keeps data_engine self-contained; app.py adds st.cache_data
# on top for the UI layer).
# --------------------------------------------------------------------------
_CACHE_REGISTRY: List[Dict[Tuple[Any, ...], Tuple[float, Any]]] = []


def clear_caches() -> None:
    """Drop every TTL cache so the next call re-fetches from the source.
    The UI's 'Refresh' button calls this together with st.cache_data.clear()."""
    for store in _CACHE_REGISTRY:
        store.clear()


def ttl_cache(seconds: int = 300) -> Callable:
    def decorator(fn: Callable) -> Callable:
        store: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}
        _CACHE_REGISTRY.append(store)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in store:
                ts, val = store[key]
                if now - ts < seconds:
                    return val
            val = fn(*args, **kwargs)
            store[key] = (now, val)
            return val
        return wrapper
    return decorator


# --------------------------------------------------------------------------
# Quote metrics — the row behind the tracker (your shared image)
# --------------------------------------------------------------------------
@dataclass
class QuoteMetrics:
    ticker: str
    last_price: float = float("nan")
    target_mean: float = float("nan")     # 1-Yr analyst target (mean)
    target_high: float = float("nan")
    target_low: float = float("nan")
    num_analysts: int = 0
    beta: float = float("nan")
    trailing_pe: float = float("nan")
    forward_pe: float = float("nan")
    profit_margin: float = float("nan")   # net margin (fraction)
    market_cap: float = float("nan")
    name: str = ""
    sector: str = ""
    currency: str = "USD"

    @property
    def expected_pct(self) -> float:
        """(target / price) - 1 — the green/red 'Expected %' in the tracker."""
        if not np.isfinite(self.last_price) or not np.isfinite(self.target_mean) \
                or self.last_price <= 0:
            return float("nan")
        return self.target_mean / self.last_price - 1.0


def _ticker(symbol: str):  # -> yf.Ticker | None
    if yf is None:
        return None
    try:
        return yf.Ticker(symbol.strip().upper())
    except Exception:
        return None


def _info(symbol: str) -> Dict[str, Any]:
    tk = _ticker(symbol)
    if tk is None:
        return {}
    try:
        info = tk.info or {}
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def _safe(d: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    val = d.get(key, default)
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


@ttl_cache(seconds=180)
def get_quote_metrics(symbol: str) -> QuoteMetrics:
    """Everything the tracker needs in one shot."""
    info = _info(symbol)
    if not info:
        return QuoteMetrics(ticker=symbol.strip().upper())

    last = _safe(info, "currentPrice")
    if not np.isfinite(last):
        last = _safe(info, "regularMarketPrice", _safe(info, "previousClose"))

    return QuoteMetrics(
        ticker=symbol.strip().upper(),
        last_price=last,
        target_mean=_safe(info, "targetMeanPrice"),
        target_high=_safe(info, "targetHighPrice"),
        target_low=_safe(info, "targetLowPrice"),
        num_analysts=int(_safe(info, "numberOfAnalystOpinions", 0) or 0),
        beta=_safe(info, "beta"),
        trailing_pe=_safe(info, "trailingPE"),
        forward_pe=_safe(info, "forwardPE"),
        profit_margin=_safe(info, "profitMargins"),
        market_cap=_safe(info, "marketCap"),
        name=str(info.get("shortName") or info.get("longName") or symbol).strip(),
        sector=str(info.get("sector") or "").strip(),
        currency=str(info.get("currency") or "USD").strip(),
    )


# --------------------------------------------------------------------------
# Financial statements
# --------------------------------------------------------------------------
@dataclass
class Statements:
    income: pd.DataFrame
    balance: pd.DataFrame
    cashflow: pd.DataFrame

    @property
    def is_empty(self) -> bool:
        return self.income.empty and self.balance.empty and self.cashflow.empty


@ttl_cache(seconds=600)
def get_statements(symbol: str, quarterly: bool = False) -> Statements:
    """Income, balance sheet, and cash flow as DataFrames (cols = periods)."""
    tk = _ticker(symbol)
    empty = pd.DataFrame()
    if tk is None:
        return Statements(empty, empty.copy(), empty.copy())

    def _grab(annual_attr: str, q_attr: str) -> pd.DataFrame:
        try:
            df = getattr(tk, q_attr if quarterly else annual_attr)
            return df if isinstance(df, pd.DataFrame) else empty.copy()
        except Exception:
            return empty.copy()

    return Statements(
        income=_grab("income_stmt", "quarterly_income_stmt"),
        balance=_grab("balance_sheet", "quarterly_balance_sheet"),
        cashflow=_grab("cashflow", "quarterly_cashflow"),
    )


def statement_row(df: pd.DataFrame, *candidates: str) -> Optional[pd.Series]:
    """Look up a line item by any of several yfinance label spellings.
    Returns the row (Series indexed by period) or None."""
    if df is None or df.empty:
        return None
    idx = {str(i).lower(): i for i in df.index}
    for cand in candidates:
        key = cand.lower()
        if key in idx:
            return df.loc[idx[key]]
    # loose contains-match fallback
    for cand in candidates:
        for low, original in idx.items():
            if cand.lower() in low:
                return df.loc[original]
    return None


def latest_value(df: pd.DataFrame, *candidates: str) -> Optional[float]:
    """Most-recent period's value for a line item, as a float."""
    row = statement_row(df, *candidates)
    if row is None or row.empty:
        return None
    try:
        val = row.dropna().iloc[0]   # yfinance orders newest-first
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError, IndexError):
        return None


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
@ttl_cache(seconds=300)
def get_prices(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Rolling-window OHLCV history. Empty DataFrame on failure."""
    tk = _ticker(symbol)
    if tk is None:
        return pd.DataFrame()
    try:
        df = tk.history(period=period, interval=interval, auto_adjust=True)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@ttl_cache(seconds=300)
def get_close_series(symbol: str, period: str = "1y", interval: str = "1d") -> pd.Series:
    """Just the Close column (named by ticker) — convenient for returns math."""
    df = get_prices(symbol, period, interval)
    if df.empty or "Close" not in df:
        return pd.Series(dtype=float, name=symbol.strip().upper())
    s = df["Close"].dropna()
    s.name = symbol.strip().upper()
    return s


def get_returns(symbol: str, period: str = "1y", interval: str = "1d") -> pd.Series:
    """Daily simple returns."""
    return get_close_series(symbol, period, interval).pct_change().dropna()


@ttl_cache(seconds=900)
def get_risk_free_rate() -> float:
    """Annualized risk-free rate from the 13-week T-bill (^IRX), as a fraction.
    Falls back to a sane constant if the fetch fails."""
    s = get_close_series(RISK_FREE_TICKER, period="5d", interval="1d")
    if s.empty:
        return FALLBACK_RISK_FREE
    try:
        return float(s.iloc[-1]) / 100.0   # ^IRX is quoted in percent
    except (TypeError, ValueError, IndexError):
        return FALLBACK_RISK_FREE


# --------------------------------------------------------------------------
# Options (for greeks)
# --------------------------------------------------------------------------
@dataclass
class OptionChain:
    expiry: str
    calls: pd.DataFrame
    puts: pd.DataFrame

    @property
    def is_empty(self) -> bool:
        return self.calls.empty and self.puts.empty


def list_expiries(symbol: str) -> List[str]:
    tk = _ticker(symbol)
    if tk is None:
        return []
    try:
        return list(tk.options or [])
    except Exception:
        return []


@ttl_cache(seconds=300)
def get_option_chain(symbol: str, expiry: Optional[str] = None) -> OptionChain:
    """Option chain for one expiry. If expiry is None, uses the nearest one."""
    tk = _ticker(symbol)
    if tk is None:
        return OptionChain("", pd.DataFrame(), pd.DataFrame())
    expiries = list_expiries(symbol)
    if not expiries:
        return OptionChain("", pd.DataFrame(), pd.DataFrame())
    chosen = expiry if expiry in expiries else expiries[0]
    try:
        oc = tk.option_chain(chosen)
        return OptionChain(chosen, oc.calls.copy(), oc.puts.copy())
    except Exception:
        return OptionChain(chosen, pd.DataFrame(), pd.DataFrame())


# --------------------------------------------------------------------------
# Fundamentals (per-share inputs for the valuation models)
# --------------------------------------------------------------------------
@dataclass
class Fundamentals:
    ticker: str
    book_value_ps: float = float("nan")   # book value per share
    eps_ttm: float = float("nan")         # trailing EPS
    dps: float = float("nan")             # annual dividend per share
    roe: float = float("nan")             # return on equity (fraction)
    payout_ratio: float = float("nan")    # dividends / earnings (fraction)
    shares_out: float = float("nan")
    total_debt: float = float("nan")
    free_cashflow: float = float("nan")   # total FCF (firm level, $)
    ebitda: float = float("nan")
    earnings_growth: float = float("nan") # yoy (fraction)
    revenue_growth: float = float("nan")  # yoy (fraction)

    @property
    def fcfe_ps(self) -> float:
        """Free cash flow to equity per share (rough: uses FCF / shares)."""
        if np.isfinite(self.free_cashflow) and np.isfinite(self.shares_out) \
                and self.shares_out > 0:
            return self.free_cashflow / self.shares_out
        return float("nan")

    @property
    def retention(self) -> float:
        """Plowback ratio b = 1 - payout (literal CFA; may be <0 if payout>1).
        The valuation default in extract_equity_inputs clamps g to [0, 0.30]."""
        if np.isfinite(self.payout_ratio):
            return 1.0 - self.payout_ratio
        return float("nan")

    @property
    def sustainable_growth(self) -> float:
        """g = ROE * retention — a sane default for terminal/explicit growth."""
        if np.isfinite(self.roe) and np.isfinite(self.retention):
            return self.roe * self.retention
        return float("nan")


@ttl_cache(seconds=300)
def get_fundamentals(symbol: str) -> Fundamentals:
    """Best-effort per-share fundamentals from the .info dict (NaN where absent)."""
    info = _info(symbol)
    if not info:
        return Fundamentals(ticker=symbol.strip().upper())
    return Fundamentals(
        ticker=symbol.strip().upper(),
        book_value_ps=_safe(info, "bookValue"),
        eps_ttm=_safe(info, "trailingEps"),
        dps=_safe(info, "dividendRate", _safe(info, "trailingAnnualDividendRate")),
        roe=_safe(info, "returnOnEquity"),
        payout_ratio=_safe(info, "payoutRatio"),
        shares_out=_safe(info, "sharesOutstanding"),
        total_debt=_safe(info, "totalDebt"),
        free_cashflow=_safe(info, "freeCashflow"),
        ebitda=_safe(info, "ebitda"),
        earnings_growth=_safe(info, "earningsGrowth"),
        revenue_growth=_safe(info, "revenueGrowth"),
    )


if __name__ == "__main__":  # quick manual check (needs internet)
    q = get_quote_metrics("AAPL")
    print(f"{q.ticker} last={q.last_price} target={q.target_mean} "
          f"expected={q.expected_pct:.2%} beta={q.beta}")
    print("risk-free:", get_risk_free_rate())
