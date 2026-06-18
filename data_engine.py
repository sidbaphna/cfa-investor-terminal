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

import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

import numpy as np
import pandas as pd
import requests

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
    quote_epoch: float = float("nan")     # last quote update (epoch seconds)
    exchange_tz: str = ""                  # IANA tz, e.g. "America/New_York"

    @property
    def expected_pct(self) -> float:
        """(target / price) - 1 — the green/red 'Expected %' in the tracker."""
        if not np.isfinite(self.last_price) or not np.isfinite(self.target_mean) \
                or self.last_price <= 0:
            return float("nan")
        return self.target_mean / self.last_price - 1.0

    @property
    def as_of(self) -> str:
        """Date/time of the last price update, in the exchange's local zone.
        Yahoo quotes are delayed, so this is the 'as of' stamp, not real-time."""
        if not np.isfinite(self.quote_epoch) or self.quote_epoch <= 0:
            return ""
        try:
            dt = datetime.fromtimestamp(self.quote_epoch, tz=timezone.utc)
            if self.exchange_tz and ZoneInfo is not None:
                try:
                    dt = dt.astimezone(ZoneInfo(self.exchange_tz))
                except Exception:
                    dt = dt.astimezone()
            else:
                dt = dt.astimezone()
            return dt.strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            return ""


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
        quote_epoch=_safe(info, "regularMarketTime",
                          _safe(info, "postMarketTime",
                                _safe(info, "preMarketTime"))),
        exchange_tz=str(info.get("exchangeTimezoneName") or "").strip(),
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


# --------------------------------------------------------------------------
# News (free, no-key — Yahoo Finance via yfinance)
# --------------------------------------------------------------------------
@dataclass
class NewsItem:
    title: str
    publisher: str
    url: str
    published: str          # formatted local date/time ("" if unknown)
    summary: str = ""


# GICS-ish sectors → (sector ETF used for headlines, broad search query)
SECTOR_QUERIES: Dict[str, Tuple[str, str]] = {
    "Technology": ("XLK", "technology sector stocks"),
    "Financials": ("XLF", "bank stocks financial sector"),
    "Health Care": ("XLV", "healthcare pharma sector stocks"),
    "Energy": ("XLE", "energy oil gas sector stocks"),
    "Consumer Discretionary": ("XLY", "consumer discretionary retail stocks"),
    "Consumer Staples": ("XLP", "consumer staples sector stocks"),
    "Industrials": ("XLI", "industrials sector stocks"),
    "Utilities": ("XLU", "utilities sector stocks"),
    "Real Estate": ("XLRE", "real estate REIT sector"),
    "Materials": ("XLB", "materials sector mining chemicals stocks"),
    "Communication Services": ("XLC", "communication services media telecom stocks"),
}


def _fmt_news_time(value: Any) -> str:
    """Format a news timestamp (ISO string or epoch seconds) to local time."""
    if value in (None, "", 0):
        return ""
    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            iso = str(value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError, OverflowError, OSError):
        return ""


def _parse_news_item(raw: Dict[str, Any]) -> NewsItem:
    """Normalize yfinance's news dict (new nested or legacy flat schema)."""
    c = raw.get("content", raw) if isinstance(raw, dict) else {}
    title = (c.get("title") or raw.get("title") or "").strip()
    summary = (c.get("summary") or c.get("description") or "").strip()
    prov = c.get("provider") or {}
    publisher = (prov.get("displayName") if isinstance(prov, dict) else "") \
        or raw.get("publisher") or ""
    url = ""
    for key in ("canonicalUrl", "clickThroughUrl"):
        u = c.get(key) or {}
        if isinstance(u, dict) and u.get("url"):
            url = u["url"]
            break
    url = url or raw.get("link") or ""
    pub = (c.get("pubDate") or c.get("displayTime")
           or raw.get("providerPublishTime") or "")
    return NewsItem(title, str(publisher).strip(), url, _fmt_news_time(pub), summary)


def _clean_news(raw_list: Any, limit: int) -> List[NewsItem]:
    items = [_parse_news_item(r) for r in (raw_list or []) if isinstance(r, dict)]
    seen, out = set(), []
    for it in items:
        if it.title and it.url and it.url not in seen:
            seen.add(it.url)
            out.append(it)
    return out[:limit]


@ttl_cache(seconds=300)
def get_ticker_news(symbol: str, limit: int = 8) -> List[NewsItem]:
    tk = _ticker(symbol)
    if tk is None:
        return []
    try:
        return _clean_news(tk.news, limit)
    except Exception:
        return []


@ttl_cache(seconds=300)
def search_news(query: str, limit: int = 8) -> List[NewsItem]:
    if yf is None:
        return []
    try:
        return _clean_news(yf.Search(query, news_count=max(limit, 8)).news, limit)
    except Exception:
        return []


@ttl_cache(seconds=300)
def get_sector_news(sector: str, limit: int = 8) -> List[NewsItem]:
    etf, query = SECTOR_QUERIES.get(sector, ("", sector))
    combined: List[NewsItem] = []
    if etf:
        combined += get_ticker_news(etf, limit)
    combined += search_news(query, limit)
    seen, out = set(), []
    for it in combined:
        if it.url not in seen:
            seen.add(it.url)
            out.append(it)
    return out[:limit]


def _num(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _date_str(v: Any) -> str:
    """Format a date/datetime/epoch into YYYY-MM-DD ('' if unusable)."""
    if v in (None, "", 0):
        return ""
    try:
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(float(v), tz=timezone.utc).astimezone().strftime("%Y-%m-%d")
        return str(v)[:10]
    except (ValueError, TypeError, OverflowError, OSError):
        return ""


# --------------------------------------------------------------------------
# Comparables (peer comparison)
# --------------------------------------------------------------------------
@dataclass
class CompMetrics:
    ticker: str
    name: str = ""
    price: float = float("nan")
    market_cap: float = float("nan")
    trailing_pe: float = float("nan")
    forward_pe: float = float("nan")
    price_to_book: float = float("nan")
    ev_to_ebitda: float = float("nan")
    gross_margin: float = float("nan")
    operating_margin: float = float("nan")
    profit_margin: float = float("nan")
    revenue_growth: float = float("nan")
    earnings_growth: float = float("nan")
    dividend_yield: float = float("nan")
    beta: float = float("nan")


@ttl_cache(seconds=300)
def get_comp_metrics(symbol: str) -> CompMetrics:
    info = _info(symbol)
    if not info:
        return CompMetrics(symbol.strip().upper())
    price = _safe(info, "currentPrice", _safe(info, "regularMarketPrice"))
    return CompMetrics(
        ticker=symbol.strip().upper(),
        name=str(info.get("shortName") or symbol).strip(),
        price=price,
        market_cap=_safe(info, "marketCap"),
        trailing_pe=_safe(info, "trailingPE"),
        forward_pe=_safe(info, "forwardPE"),
        price_to_book=_safe(info, "priceToBook"),
        ev_to_ebitda=_safe(info, "enterpriseToEbitda"),
        gross_margin=_safe(info, "grossMargins"),
        operating_margin=_safe(info, "operatingMargins"),
        profit_margin=_safe(info, "profitMargins"),
        revenue_growth=_safe(info, "revenueGrowth"),
        earnings_growth=_safe(info, "earningsGrowth"),
        dividend_yield=_safe(info, "trailingAnnualDividendYield",
                             _safe(info, "dividendYield")),
        beta=_safe(info, "beta"),
    )


# --------------------------------------------------------------------------
# Calendar (earnings + dividend dates)
# --------------------------------------------------------------------------
@dataclass
class CalendarInfo:
    ticker: str
    next_earnings: str = ""
    ex_dividend: str = ""
    dividend_rate: float = float("nan")
    dividend_yield: float = float("nan")
    eps_estimate: float = float("nan")


@ttl_cache(seconds=600)
def get_calendar(symbol: str) -> CalendarInfo:
    tk = _ticker(symbol)
    out = CalendarInfo(ticker=symbol.strip().upper())
    if tk is None:
        return out
    try:
        cal = tk.calendar or {}
    except Exception:
        cal = {}
    if isinstance(cal, dict):
        ed = cal.get("Earnings Date")
        if isinstance(ed, (list, tuple)) and ed:
            out.next_earnings = _date_str(ed[0])
        elif ed:
            out.next_earnings = _date_str(ed)
        out.ex_dividend = _date_str(cal.get("Ex-Dividend Date"))
        out.eps_estimate = _num(cal.get("Earnings Average"))
    info = _info(symbol)
    if not out.next_earnings:
        out.next_earnings = _date_str(info.get("earningsTimestamp"))
    if not out.ex_dividend:
        out.ex_dividend = _date_str(info.get("exDividendDate"))
    out.dividend_rate = _safe(info, "dividendRate",
                              _safe(info, "trailingAnnualDividendRate"))
    out.dividend_yield = _safe(info, "trailingAnnualDividendYield",
                               _safe(info, "dividendYield"))
    return out


# --------------------------------------------------------------------------
# Macro snapshot (FRED — free, no API key)
# --------------------------------------------------------------------------
@dataclass
class MacroIndicator:
    label: str
    value: float
    unit: str
    as_of: str
    change: float = float("nan")   # vs prior reading (level series only)


# (label, FRED series id, unit, compute_year_over_year)
FRED_SERIES: List[Tuple[str, str, str, bool]] = [
    ("Fed Funds", "DFF", "%", False),
    ("2Y Treasury", "DGS2", "%", False),
    ("10Y Treasury", "DGS10", "%", False),
    ("Unemployment", "UNRATE", "%", False),
    ("CPI YoY", "CPIAUCSL", "%", True),
    ("Core CPI YoY", "CPILFESL", "%", True),
    ("Real GDP SAAR", "A191RL1Q225SBEA", "%", False),
]


@ttl_cache(seconds=21600)
def get_fred_series(series_id: str, yoy: bool = False) -> Tuple[float, str, float]:
    """Return (latest_value, as_of_date, change_vs_prior) from FRED's free CSV.
    For yoy=True the value is the 12-period % change of the index."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception:
        return (float("nan"), "", float("nan"))
    if df.empty or df.shape[1] < 2:
        return (float("nan"), "", float("nan"))
    date_col, val_col = df.columns[0], df.columns[1]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])
    if df.empty:
        return (float("nan"), "", float("nan"))
    latest = df.iloc[-1]
    as_of = str(latest[date_col])[:10]
    if yoy:
        if len(df) > 12:
            prior = float(df[val_col].iloc[-13])
            value = (float(latest[val_col]) / prior - 1) * 100 if prior else float("nan")
        else:
            value = float("nan")
        return (value, as_of, float("nan"))
    value = float(latest[val_col])
    change = value - float(df[val_col].iloc[-2]) if len(df) > 1 else float("nan")
    return (value, as_of, change)


@ttl_cache(seconds=21600)
def get_macro_snapshot() -> List[MacroIndicator]:
    out: List[MacroIndicator] = []
    for label, sid, unit, yoy in FRED_SERIES:
        value, as_of, change = get_fred_series(sid, yoy)
        out.append(MacroIndicator(label, value, unit, as_of, change))
    return out


if __name__ == "__main__":  # quick manual check (needs internet)
    q = get_quote_metrics("AAPL")
    print(f"{q.ticker} last={q.last_price} target={q.target_mean} "
          f"expected={q.expected_pct:.2%} beta={q.beta}")
    print("risk-free:", get_risk_free_rate())
    print("news:", [n.title[:50] for n in get_ticker_news("AAPL", 3)])
    print("macro:", [(m.label, round(m.value, 2)) for m in get_macro_snapshot()])
