"""app.py — The Universal CFA Investor Terminal (Streamlit dashboard).

Layout
------
* Top: global "Learning vs Earning" mode toggle.
* Sidebar: 💡 ELI5 Concept Decoder + portfolio/watchlist manager + settings.
* Tabs: Tracker | Equity Valuation | Fixed Income | Forensics (FSA) |
        Portfolio Risk | Greeks.

Every heavy data call is wrapped in st.cache_data; the topic math lives in
topics/* and the data in data_engine — app.py only orchestrates and renders.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data_engine as de
import llm_orchestrator as llm
from config import (DEFAULT_BENCHMARK, DEFAULT_DISCOUNT_RATE, LLM_MODEL,
                    MODE_ACTIVE, MODE_STUDY)
from importers import tickers_from_csv
from storage import PortfolioStore
from topics import derivatives as dv
from topics import equity_valuation as ev
from topics import fixed_income as fi
from topics import fsa
from topics import portfolio_mgmt as pm
from topics import research

st.set_page_config(page_title="CFA Investor Terminal", page_icon="📊",
                   layout="wide")


# ==========================================================================
# Config + login gate (for remote/phone access)
# ==========================================================================
def _conf(name: str, default=None):
    """Read a setting from Streamlit secrets first, then the environment."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def _require_login() -> None:
    """Single-password gate. If no password is configured, the app is open
    (convenient for local dev). Set CFA_APP_PASSWORD (env or Streamlit secret)
    to protect the deployed app."""
    password = _conf("CFA_APP_PASSWORD")
    if not password or st.session_state.get("authed"):
        return
    st.title("🔒 CFA Investor Terminal")
    st.caption("Enter the access password to continue.")
    pw = st.text_input("Password", type="password", key="login_pw")
    if st.button("Enter", key="login_btn"):
        if pw == str(password):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


_require_login()


# ==========================================================================
# Cached data wrappers (UI-layer cache on top of data_engine's TTL cache)
# ==========================================================================
@st.cache_data(ttl=180, show_spinner=False)
def c_quote(sym: str) -> de.QuoteMetrics:
    return de.get_quote_metrics(sym)


@st.cache_data(ttl=300, show_spinner=False)
def c_close(sym: str, period: str = "1y") -> pd.Series:
    return de.get_close_series(sym, period=period)


@st.cache_data(ttl=600, show_spinner=False)
def c_statements(sym: str) -> de.Statements:
    return de.get_statements(sym)


@st.cache_data(ttl=300, show_spinner=False)
def c_inputs(sym: str, r: float, g_term: float) -> ev.EquityInputs:
    return ev.extract_equity_inputs(sym, r=r, g_term=g_term)


@st.cache_data(ttl=900, show_spinner=False)
def c_riskfree() -> float:
    return de.get_risk_free_rate()


# ==========================================================================
# Session / store
# ==========================================================================
@st.cache_resource
def get_store(db_url: str = "") -> PortfolioStore:
    return PortfolioStore(db_url=db_url or None)


store = get_store(_conf("CFA_DB_URL", "") or "")
st.session_state.setdefault("mode", MODE_STUDY)


def fmt(x: float, pct: bool = False, money: bool = False, dp: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if pct:
        return f"{x * 100:.1f}%"
    if money:
        return f"{x:,.{dp}f}"
    return f"{x:,.{dp}f}"


def deconstruct_button(key: str, topic: str, ticker: str, payload: str) -> None:
    """Contextual 'Deconstruct Topic' control rendered next to any output."""
    cols = st.columns([1, 3])
    with cols[0]:
        clicked = st.button("🧩 Deconstruct", key=f"btn_{key}",
                            width="stretch")
    if clicked:
        with st.spinner("Consulting the CFA orchestrator…"):
            st.session_state[f"deep_{key}"] = llm.deep_analysis(
                topic, ticker, payload, st.session_state["mode"])
    if st.session_state.get(f"deep_{key}"):
        with st.expander("🧠 Deconstruction", expanded=True):
            st.markdown(st.session_state[f"deep_{key}"])


def picker(label: str, key: str, default: str = "AAPL") -> str:
    """Ticker selector seeded from the watchlist/holdings, with free entry."""
    known = store.all_tickers()
    options = known + ["✏️ Type a ticker…"]
    idx = 0 if known else len(options) - 1
    choice = st.selectbox(label, options, index=idx, key=f"sel_{key}")
    if choice == "✏️ Type a ticker…" or not known:
        return st.text_input("Ticker", value=default, key=f"txt_{key}").strip().upper()
    return choice


def _momentum(sym: str) -> Dict[str, float]:
    """Active-mode price markers from the 1y close series."""
    s = c_close(sym, "1y")
    if s.empty:
        return {}
    last = float(s.iloc[-1])
    return {
        "last": last,
        "vs_ma50": last / s.tail(50).mean() - 1 if len(s) >= 50 else float("nan"),
        "vs_ma200": last / s.tail(200).mean() - 1 if len(s) >= 200 else float("nan"),
        "from_high": last / s.max() - 1,
        "from_low": last / s.min() - 1,
        "r3m": last / s.iloc[-63] - 1 if len(s) > 63 else float("nan"),
        "r6m": last / s.iloc[-126] - 1 if len(s) > 126 else float("nan"),
    }


# ==========================================================================
# Header + mode toggle
# ==========================================================================
st.title("📊 The Universal CFA Investor Terminal")
top = st.columns([3, 2])
with top[0]:
    st.caption("Manual local watchlists · yfinance data · CFA L2 cross-topic engine")
with top[1]:
    st.session_state["mode"] = st.radio(
        "Framework", [MODE_STUDY, MODE_ACTIVE],
        index=0 if st.session_state["mode"] == MODE_STUDY else 1,
        horizontal=True, label_visibility="collapsed")

MODE = st.session_state["mode"]
STUDY = MODE == MODE_STUDY
st.caption("🎓 **Study lens** — formula derivations, IFRS vs US GAAP notes, and "
           "practice prompts." if STUDY else
           "💰 **Active lens** — momentum, trailing multiples, margins, and an "
           "actionable risk read.")


# ==========================================================================
# Sidebar — ELI5 decoder, portfolio manager, settings
# ==========================================================================
with st.sidebar:
    st.header("💡 The ELI5 Concept Decoder")
    common = ["Option-Adjusted Spread", "Temporal Method Cumulative Translation "
              "Adjustment", "Beneish M-Score DSRI metric", "Residual Income",
              "Modified Duration", "Information Ratio", "Tracking Error",
              "Vega", "Clean Surplus Relation", "Persistence Factor (ω)"]
    quick = st.selectbox("Quick picks", ["—"] + common, index=0)
    term = st.text_input("…or type any phrase on screen",
                         value="" if quick == "—" else quick)
    ctx = st.text_input("Context (optional)", value="")
    if st.button("🔍 Decode", width="stretch"):
        with st.spinner("Decoding…"):
            st.session_state["eli5"] = llm.eli5_explain(term, ctx)
    if st.session_state.get("eli5"):
        st.markdown(st.session_state["eli5"])
        if st.button("Clear", key="clr_eli5"):
            st.session_state.pop("eli5", None)

    st.divider()
    st.subheader("⚙️ Settings")
    bench = st.text_input("Benchmark", value=store.get_setting("benchmark",
                          DEFAULT_BENCHMARK)).strip().upper()
    disc = st.number_input("Discount rate r (cost of equity)", min_value=0.01,
                           max_value=0.40,
                           value=float(store.get_setting("discount_rate",
                                       DEFAULT_DISCOUNT_RATE)), step=0.005,
                           format="%.3f")
    if bench != store.get_setting("benchmark") or disc != store.get_setting("discount_rate"):
        store.set_setting("benchmark", bench)
        store.set_setting("discount_rate", disc)
    if llm.is_available():
        st.caption(f"LLM: `{llm.active_model()}` · 🟢 {llm.provider_name()}")
    else:
        st.caption("LLM: 🔴 offline · ELI5 uses the built-in glossary")

    if st.button("🔄 Refresh market data", width="stretch"):
        de.clear_caches()          # bust data_engine's internal TTL cache
        st.cache_data.clear()      # bust the Streamlit UI cache
        st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()
    st.caption(f"Prices via yfinance (Yahoo, delayed). Risk-free (^IRX): "
               f"{fmt(c_riskfree(), pct=True)}")
    st.caption(f"Last manual refresh: {st.session_state.get('last_refresh', '—')} "
               "· auto-expires ~3 min")
    st.caption(f"Storage: {store.backend_description()}")

    st.divider()
    st.subheader("📋 Watchlist & Holdings")
    with st.expander("📤 Import watchlist from CSV"):
        up = st.file_uploader("CSV with a Symbol/Ticker column", type=["csv"],
                              key="wl_csv")
        if up is not None:
            syms: List[str] = []
            err = None
            try:
                syms = tickers_from_csv(up)
            except Exception as exc:                      # noqa: BLE001
                err = str(exc)
            if err:
                st.error(f"Couldn't read that CSV: {err}")
            elif syms:
                preview = ", ".join(syms[:8]) + (" …" if len(syms) > 8 else "")
                st.caption(f"Found {len(syms)} tickers: {preview}")
                if st.button(f"➕ Add {len(syms)} to watchlist", key="wl_csv_btn",
                             width="stretch"):
                    existing = {w.ticker for w in store.list_watchlist()}
                    for s in syms:
                        store.add_watch(s)               # upsert; no strike
                    added = sum(1 for s in syms if s not in existing)
                    st.success(f"Added {added} new · {len(syms) - added} already there.")
                    st.rerun()
            else:
                st.warning("No tickers found — need a 'Symbol' or 'Ticker' column.")
    with st.expander("➕ Add to watchlist"):
        wt = st.text_input("Ticker", key="wt_add").strip().upper()
        wstrike = st.number_input("Target strike (your thesis price)",
                                  min_value=0.0, value=0.0, key="wt_strike")
        if st.button("Add watch", key="wt_btn") and wt:
            store.add_watch(wt, wstrike if wstrike > 0 else None)
            st.success(f"Added {wt}")
            st.rerun()
    with st.expander("➕ Add holding"):
        ht = st.text_input("Ticker", key="ht_add").strip().upper()
        hcb = st.number_input("Cost basis", min_value=0.0, value=0.0, key="ht_cb")
        hsz = st.number_input("Size (shares)", min_value=0.0, value=0.0, key="ht_sz")
        if st.button("Add holding", key="ht_btn") and ht:
            store.add_holding(ht, hcb, hsz)
            st.success(f"Added {ht}")
            st.rerun()
    rm = st.selectbox("Remove ticker", ["—"] + store.all_tickers())
    if st.button("Remove", key="rm_btn") and rm != "—":
        store.remove_watch(rm)
        store.remove_holding(rm)
        st.rerun()


# ==========================================================================
# Tabs
# ==========================================================================
tab_track, tab_news, tab_research, tab_eq, tab_fi, tab_fsa, tab_pm, tab_grk = st.tabs(
    ["📡 Tracker", "📰 News & Sectors", "🔭 Research", "🧮 Equity Valuation",
     "🏦 Fixed Income", "🔬 Forensics (FSA)", "📈 Portfolio Risk", "🎲 Greeks"])


# --------------------------------------------------------------------------
# TRACKER (the shared image)
# --------------------------------------------------------------------------
def _color_expected(val: float) -> str:
    if pd.isna(val):
        return ""
    intensity = min(abs(val) / 0.25, 1.0)
    if val >= 0:
        return f"background-color: rgba(122,170,135,{0.20 + 0.55 * intensity}); color:#0a2e16;"
    return f"background-color: rgba(198,121,103,{0.20 + 0.55 * intensity}); color:#3d0f08;"


with tab_track:
    st.subheader("Current Price vs. 1-Yr Analyst Target")
    tickers = store.all_tickers()
    show_greeks = st.checkbox("Include ATM greeks (slower — fetches option chains)",
                              value=False)
    if not tickers:
        st.info("Add tickers in the sidebar to populate your tracker.")
    else:
        rows: List[Dict] = []
        for t in tickers:
            q = c_quote(t)
            exp = q.expected_pct
            prev = store.previous_expected_pct(t)
            delta = (exp - prev) if (prev is not None and np.isfinite(exp)) else float("nan")
            if np.isfinite(exp) and np.isfinite(q.last_price) and np.isfinite(q.target_mean):
                store.record_snapshot(t, q.last_price, q.target_mean, exp)
            row = {
                "Ticker": t,
                "Last Price": q.last_price,
                "1 Yr Target Est": q.target_mean,
                "As of": q.as_of,
                "Expected %": exp,
                "Δ Expected %": delta,
                "Beta": q.beta,
                "Fwd P/E": q.forward_pe,
                "# Analysts": q.num_analysts,
            }
            if show_greeks:
                g = dv.implied_greeks_from_chain(t, "call")
                row["ATM Δ"] = g.delta if g else float("nan")
                row["IV"] = g.iv if g else float("nan")
            rows.append(row)
        store.save()

        df = pd.DataFrame(rows).set_index("Ticker")
        pct_cols = ["Expected %", "Δ Expected %"] + (["IV"] if show_greeks else [])
        styler = (df.style
                  .map(_color_expected, subset=["Expected %", "Δ Expected %"])
                  .format({"Last Price": "{:,.2f}", "1 Yr Target Est": "{:,.2f}",
                           "Expected %": "{:.1%}", "Δ Expected %": "{:+.2%}",
                           "Beta": "{:.2f}", "Fwd P/E": "{:.1f}",
                           **({"ATM Δ": "{:.2f}", "IV": "{:.1%}"} if show_greeks else {})},
                          na_rep="—"))
        st.dataframe(styler, width="stretch")
        st.caption("Expected % = (Target / Price) − 1. ‘As of’ is each quote’s last "
                   "Yahoo update (delayed) in the exchange’s local time. Δ Expected % "
                   "compares to the last saved reading.")

        # Mode lens
        if not df.empty:
            best = df["Expected %"].idxmax()
            payload = df.round(4).to_string()
            deconstruct_button("track", "Analyst-target tracker", best, payload)


# --------------------------------------------------------------------------
# NEWS & SECTORS
# --------------------------------------------------------------------------
def _render_news(items, empty_msg: str = "No headlines available right now.") -> None:
    if not items:
        st.caption(empty_msg)
        return
    for n in items:
        meta = " · ".join(x for x in [n.publisher, n.published] if x)
        line = f"- **[{n.title}]({n.url})**"
        if meta:
            line += f"  \n  <sub>{meta}</sub>"
        st.markdown(line, unsafe_allow_html=True)


with tab_news:
    st.subheader("News & Sector Trends")
    st.caption("Free Yahoo Finance headlines (no API key needed) — links open the "
               "full article.")

    section = st.radio("Feed", ["📈 Markets", "🇺🇸 U.S. economy", "🌍 World",
                                "🏭 Sector", "📌 My watchlist"], horizontal=True)

    items: List = []
    context_label = "the broad market"

    if section == "📈 Markets":
        items = de.search_news("stock market today earnings movers", 12)
    elif section == "🇺🇸 U.S. economy":
        items = de.search_news("US economy Federal Reserve inflation interest rates jobs", 12)
        context_label = "the U.S. macro economy"
    elif section == "🌍 World":
        items = de.search_news("global markets world economy geopolitics commodities", 12)
        context_label = "global markets"
    elif section == "🏭 Sector":
        sector = st.selectbox("Sector", list(de.SECTOR_QUERIES.keys()))
        etf = de.SECTOR_QUERIES[sector][0]
        mo = _momentum(etf)
        if mo:
            cc = st.columns(4)
            cc[0].metric(f"{etf} vs 50D", fmt(mo.get("vs_ma50"), pct=True))
            cc[1].metric(f"{etf} vs 200D", fmt(mo.get("vs_ma200"), pct=True))
            cc[2].metric("3M return", fmt(mo.get("r3m"), pct=True))
            cc[3].metric("From 52w high", fmt(mo.get("from_high"), pct=True))
        items = de.get_sector_news(sector, 12)
        context_label = f"the {sector} sector"
    else:  # My watchlist
        tks = store.all_tickers()
        if not tks:
            st.info("Add tickers (sidebar or CSV import) to see their news.")
        else:
            pick = st.selectbox("Ticker", tks)
            items = de.get_ticker_news(pick, 12)
            context_label = pick

    _render_news(items)

    if items:
        st.divider()
        if llm.is_available():
            if st.button("🧠 How does this affect my investments?", key="news_ai"):
                heads = "\n".join(f"- {n.title} ({n.publisher})" for n in items)
                with st.spinner("Analyzing…"):
                    st.session_state["news_impact"] = llm.news_impact(
                        heads, ", ".join(store.all_tickers()[:40]),
                        st.session_state["mode"])
            if st.session_state.get("news_impact"):
                with st.expander("🧠 Investment impact", expanded=True):
                    st.markdown(st.session_state["news_impact"])
        else:
            st.caption(f"💡 Reading {context_label}. Add an `ANTHROPIC_API_KEY` for an "
                       "AI read on how these headlines affect your holdings.")


# --------------------------------------------------------------------------
# RESEARCH (financials · peers · calendar & macro)
# --------------------------------------------------------------------------
with tab_research:
    st.subheader("Research")
    rsec = st.radio("View", ["🧾 Financials", "⚖️ Peer comparison",
                             "📅 Calendar & Macro", "📈 Technicals"],
                    horizontal=True, key="research_view")

    if rsec == "🧾 Financials":
        sym = picker("Company", "rfin")
        if sym:
            s = c_statements(sym)
            which = st.selectbox("Statement", ["Income", "Balance Sheet", "Cash Flow"])
            df = {"Income": s.income, "Balance Sheet": s.balance,
                  "Cash Flow": s.cashflow}[which]
            view = st.radio("Show", ["Values", "YoY growth", "Common-size %"],
                            horizontal=True)
            if df is None or df.empty:
                st.warning("No statement data available for this name.")
            else:
                if view == "Values":
                    show = df.copy()
                    show.columns = [str(c)[:10] for c in show.columns]
                    st.dataframe(show.style.format("{:,.0f}", na_rep="—"),
                                 width="stretch")
                elif view == "YoY growth":
                    g = research.yoy_growth(df)
                    st.dataframe(g.style.format("{:+.1%}", na_rep="—")
                                 .map(_color_expected), width="stretch")
                else:
                    cs = research.common_size(df, "Total Revenue", "Total Assets")
                    if cs.empty:
                        st.caption("Common-size needs a Total Revenue / Total Assets "
                                   "base row — not available for this statement.")
                    else:
                        st.dataframe(cs.style.format("{:.1%}", na_rep="—"),
                                     width="stretch")
                rr = research.key_ratios(s.income, s.balance, s.cashflow)
                cols = st.columns(len(rr))
                pct_keys = {"Gross margin", "Operating margin", "Net margin",
                            "ROA", "ROE"}
                for (k, v), c in zip(rr.items(), cols):
                    c.metric(k, fmt(v, pct=k in pct_keys))
                if STUDY:
                    with st.expander("📗 IFRS vs US GAAP — statement presentation"):
                        st.markdown(
                            "- **Cash flow classification**: under US GAAP interest "
                            "paid/received and dividends received are operating, "
                            "dividends paid are financing; **IFRS lets you choose** "
                            "(operating or investing/financing) — compare CFO carefully.\n"
                            "- **Income statement**: IFRS allows expenses by *nature* "
                            "or *function*; US GAAP is by function. No 'extraordinary "
                            "items' under either today.\n"
                            "- **Inventory/PP&E**: LIFO (US GAAP only) and IFRS PP&E "
                            "revaluation make cross-border comparisons non-trivial.")
                else:
                    g = research.yoy_growth(df)
                    rev = g.loc[[i for i in g.index
                                 if "revenue" in str(i).lower()][:1]] if not g.empty else pd.DataFrame()
                    if not rev.empty and rev.shape[1] >= 1:
                        latest_rev = rev.iloc[0, 0]
                        st.caption(f"📈 Latest revenue YoY: **{fmt(latest_rev, pct=True)}** "
                                   "— scan the YoY view for margin and cash-flow trend.")
                deconstruct_button("rfin", "Financial Statement Analysis", sym,
                                   f"{which} statement; key ratios = {rr}")

    elif rsec == "⚖️ Peer comparison":
        base_sym = picker("Primary ticker", "rpeer")
        peers_txt = st.text_input("Peers (comma-separated)", value="MSFT, GOOG, AMZN")
        syms = [base_sym] + [p.strip().upper() for p in peers_txt.split(",") if p.strip()]
        syms = list(dict.fromkeys([s for s in syms if s]))   # dedupe, keep order
        if syms:
            rows = []
            for s in syms:
                m = de.get_comp_metrics(s)
                rows.append({
                    "Ticker": m.ticker, "Price": m.price, "Mkt Cap": m.market_cap,
                    "P/E (ttm)": m.trailing_pe, "Fwd P/E": m.forward_pe,
                    "P/B": m.price_to_book, "EV/EBITDA": m.ev_to_ebitda,
                    "Gross M": m.gross_margin, "Op M": m.operating_margin,
                    "Net M": m.profit_margin, "Rev Gr": m.revenue_growth,
                    "EPS Gr": m.earnings_growth, "Div Yld": m.dividend_yield,
                    "Beta": m.beta})
            cdf = pd.DataFrame(rows).set_index("Ticker")
            st.dataframe(cdf.style.format(
                {"Price": "{:,.2f}", "Mkt Cap": "{:,.0f}", "P/E (ttm)": "{:.1f}",
                 "Fwd P/E": "{:.1f}", "P/B": "{:.2f}", "EV/EBITDA": "{:.1f}",
                 "Gross M": "{:.1%}", "Op M": "{:.1%}", "Net M": "{:.1%}",
                 "Rev Gr": "{:+.1%}", "EPS Gr": "{:+.1%}", "Div Yld": "{:.2%}",
                 "Beta": "{:.2f}"}, na_rep="—"), width="stretch")
            deconstruct_button("rpeer", "Peer / Comparable Analysis", base_sym,
                               cdf.round(3).to_string())

    elif rsec == "📅 Calendar & Macro":
        st.markdown("**Earnings & dividend dates**")
        sym = picker("Company", "rcal")
        if sym:
            cal = de.get_calendar(sym)
            cc = st.columns(4)
            cc[0].metric("Next earnings", cal.next_earnings or "—")
            cc[1].metric("Ex-dividend", cal.ex_dividend or "—")
            cc[2].metric("Dividend rate", fmt(cal.dividend_rate))
            cc[3].metric("Dividend yield", fmt(cal.dividend_yield, pct=True))
        if store.all_tickers() and st.button("📋 Load watchlist earnings dates (slow)"):
            wrows = []
            for t in store.all_tickers()[:40]:
                c = de.get_calendar(t)
                if c.next_earnings:
                    wrows.append({"Ticker": t, "Next earnings": c.next_earnings,
                                  "Ex-dividend": c.ex_dividend})
            if wrows:
                st.dataframe(pd.DataFrame(wrows).set_index("Ticker")
                             .sort_values("Next earnings"), width="stretch")
            else:
                st.caption("No upcoming dates found for the watchlist.")

        st.divider()
        st.markdown("**Macro snapshot** — Federal Reserve Economic Data (free)")
        macro = de.get_macro_snapshot()
        if macro:
            mcols = st.columns(len(macro))
            for ind, c in zip(macro, mcols):
                vs = f"{ind.value:.2f}{ind.unit}" if np.isfinite(ind.value) else "—"
                delta = f"{ind.change:+.2f}" if np.isfinite(ind.change) else None
                c.metric(ind.label, vs, delta)
            st.caption("Rates/levels in %, CPI shown year-over-year. As-of dates vary "
                       "by series (daily/monthly/quarterly).")
            deconstruct_button("rmacro", "Macro Snapshot", "Macro",
                               "; ".join(f"{i.label}={i.value:.2f}{i.unit}"
                                         for i in macro if np.isfinite(i.value)))

    else:  # 📈 Technicals
        sym = picker("Company", "rtech")
        period = st.selectbox("Window", ["6mo", "1y", "2y", "5y"], index=1,
                              key="tech_period")
        overlays = st.multiselect(
            "Overlays", ["SMA 20", "SMA 50", "SMA 200", "Bollinger (20, 2σ)"],
            default=["SMA 50", "SMA 200"])
        px = de.get_prices(sym, period) if sym else pd.DataFrame()
        if px.empty or "Close" not in px:
            st.warning("No price history available for this name.")
        else:
            close = px["Close"].dropna()
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03,
                                row_heights=[0.52, 0.12, 0.18, 0.18],
                                subplot_titles=("Price", "Volume", "RSI (14)",
                                                "MACD (12, 26, 9)"))
            fig.add_trace(go.Candlestick(
                x=px.index, open=px["Open"], high=px["High"], low=px["Low"],
                close=px["Close"], name="Price", showlegend=False), row=1, col=1)
            for label, win in [("SMA 20", 20), ("SMA 50", 50), ("SMA 200", 200)]:
                if label in overlays:
                    fig.add_trace(go.Scatter(x=px.index, y=research.sma(close, win),
                                  name=label, line=dict(width=1)), row=1, col=1)
            if "Bollinger (20, 2σ)" in overlays:
                _mid, _up, _lo = research.bollinger(close, 20, 2.0)
                fig.add_trace(go.Scatter(x=px.index, y=_up, name="BB upper",
                              line=dict(width=1, dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=px.index, y=_lo, name="BB lower",
                              line=dict(width=1, dash="dot"), fill="tonexty",
                              fillcolor="rgba(120,120,160,0.08)"), row=1, col=1)
            if "Volume" in px:
                fig.add_trace(go.Bar(x=px.index, y=px["Volume"], name="Vol",
                              marker_color="rgba(120,140,170,0.5)",
                              showlegend=False), row=2, col=1)
            r = research.rsi(close)
            fig.add_trace(go.Scatter(x=close.index, y=r, name="RSI",
                          line=dict(width=1), showlegend=False), row=3, col=1)
            fig.add_hline(y=70, line=dict(width=1, dash="dot", color="#c67967"),
                          row=3, col=1)
            fig.add_hline(y=30, line=dict(width=1, dash="dot", color="#7aaa87"),
                          row=3, col=1)
            macd_line, signal_line, hist = research.macd(close)
            fig.add_trace(go.Bar(x=close.index, y=hist, name="Hist",
                          marker_color="rgba(140,140,170,0.5)", showlegend=False),
                          row=4, col=1)
            fig.add_trace(go.Scatter(x=close.index, y=macd_line, name="MACD",
                          line=dict(width=1)), row=4, col=1)
            fig.add_trace(go.Scatter(x=close.index, y=signal_line, name="Signal",
                          line=dict(width=1)), row=4, col=1)
            fig.update_layout(height=720, margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_rangeslider_visible=False,
                              legend=dict(orientation="h", y=1.02, x=0))
            fig.update_yaxes(range=[0, 100], row=3, col=1)
            st.plotly_chart(fig, width="stretch")
            rsi_clean = r.dropna()
            last_rsi = float(rsi_clean.iloc[-1]) if not rsi_clean.empty else float("nan")
            tag = ("overbought" if last_rsi >= 70 else
                   "oversold" if last_rsi <= 30 else "neutral")
            st.caption(f"Latest RSI {fmt(last_rsi)} ({tag}). Candles use "
                       "auto-adjusted prices.")
            deconstruct_button("rtech", "Technical Analysis", sym,
                               f"period={period}; last RSI={last_rsi:.1f}; "
                               f"overlays={overlays}")


# --------------------------------------------------------------------------
# EQUITY VALUATION
# --------------------------------------------------------------------------
with tab_eq:
    st.subheader("Intrinsic Value — DDM · FCFE · FCFF · Residual Income")
    sym = picker("Ticker to value", "eq")
    if sym:
        inp = c_inputs(sym, disc, 0.025)   # g_term default 2.5%
        c1, c2, c3 = st.columns(3)
        with c1:
            r = st.number_input("r (cost of equity)", 0.01, 0.40, float(inp.r), 0.005, format="%.3f")
            g_high = st.number_input("g (high growth)", -0.10, 0.40,
                                     float(inp.g_high if np.isfinite(inp.g_high) else 0.08),
                                     0.005, format="%.3f")
        with c2:
            g_term = st.number_input("g (terminal)", 0.0, 0.10, 0.025, 0.0025, format="%.4f")
            n_high = st.number_input("High-growth years", 1, 15, 5)
        with c3:
            persistence = st.slider("RI persistence factor ω", 0.0, 1.0, 0.6, 0.05)
            book = st.number_input("Book value / share",
                                   value=float(inp.book_value_ps) if np.isfinite(inp.book_value_ps) else 0.0,
                                   format="%.2f")

        with st.expander("Per-share inputs (override the auto-pulled fundamentals)"):
            cc = st.columns(4)
            roe = cc[0].number_input("ROE", value=float(inp.roe) if np.isfinite(inp.roe) else 0.0, format="%.3f")
            dps = cc[1].number_input("Dividend / share", value=float(inp.dps), format="%.2f")
            fcfe = cc[2].number_input("FCFE / share",
                                      value=float(inp.fcfe_ps) if np.isfinite(inp.fcfe_ps) else 0.0, format="%.2f")
            price = cc[3].number_input("Current price",
                                       value=float(inp.price) if np.isfinite(inp.price) else 0.0, format="%.2f")

        results: Dict[str, ev.ValuationResult] = {}
        if np.isfinite(book) and np.isfinite(roe) and roe != 0:
            results["Residual Income"] = ev.residual_income_multistage(
                book, roe, r, g_high, int(n_high), persistence, price)
        if dps > 0:
            results["Two-Stage DDM"] = ev.two_stage_ddm(dps, r, g_high, int(n_high), g_term, price)
        if fcfe != 0:
            results["Two-Stage FCFE"] = ev.two_stage_fcfe(fcfe, r, g_high, int(n_high), g_term, price)
        if np.isfinite(inp.fcff) and np.isfinite(inp.shares):
            results["Two-Stage FCFF"] = ev.two_stage_fcff(
                inp.fcff, r, g_high, int(n_high), g_term, inp.total_debt, inp.shares, price)

        if results:
            table = pd.DataFrame([{
                "Model": k, "Intrinsic Value": v.intrinsic_value,
                "Price": v.current_price, "Upside %": v.upside_pct,
            } for k, v in results.items()]).set_index("Model")
            st.dataframe(table.style.format(
                {"Intrinsic Value": "{:,.2f}", "Price": "{:,.2f}", "Upside %": "{:+.1%}"},
                na_rep="—").map(_color_expected, subset=["Upside %"]),
                width="stretch")

            if STUDY:
                ri = results.get("Residual Income")
                if ri and ri.schedule:
                    with st.expander("📘 Residual Income schedule & formula"):
                        st.latex(r"V_0 = B_0 + \sum_{t=1}^{\infty}\frac{E_t - r\,B_{t-1}}{(1+r)^t}")
                        st.dataframe(pd.DataFrame(ri.schedule), width="stretch")
                with st.expander("📗 IFRS vs US GAAP — what shifts these inputs"):
                    st.markdown(
                        "- **R&D**: US GAAP expenses it; IFRS *capitalizes* qualifying "
                        "development costs → higher book value/assets, lower current "
                        "earnings → moves B₀ and ROE in the RI model.\n"
                        "- **Inventory**: LIFO is allowed under US GAAP, **banned** under "
                        "IFRS → COGS, margins, and equity differ (watch the LIFO reserve).\n"
                        "- **PP&E**: IFRS permits upward **revaluation**; US GAAP is "
                        "historical cost → affects book value and clean surplus.\n"
                        "- **Clean surplus**: items routed through OCI (FX, some "
                        "remeasurements) violate Bₜ = Bₜ₋₁ + Eₜ − Dₜ and bias RI.")
            else:
                mo = _momentum(sym)
                if mo:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("vs 50D MA", fmt(mo["vs_ma50"], pct=True))
                    m2.metric("vs 200D MA", fmt(mo["vs_ma200"], pct=True))
                    m3.metric("3M return", fmt(mo["r3m"], pct=True))
                    m4.metric("From 52w high", fmt(mo["from_high"], pct=True))
                cm = de.get_comp_metrics(sym)
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("P/E (ttm)", fmt(cm.trailing_pe, dp=1))
                q2.metric("Fwd P/E", fmt(cm.forward_pe, dp=1))
                q3.metric("Net margin", fmt(cm.profit_margin, pct=True))
                q4.metric("EPS growth", fmt(cm.earnings_growth, pct=True))

            payload = table.round(3).to_string() + f"\nAssumptions: r={r}, g_high={g_high}, g_term={g_term}, n={n_high}, ω={persistence}"
            deconstruct_button("eq", "Equity Valuation", sym, payload)
        else:
            st.warning("Not enough data to value this name. Enter inputs manually above.")


# --------------------------------------------------------------------------
# FIXED INCOME
# --------------------------------------------------------------------------
with tab_fi:
    st.subheader("Bond Analytics, Spot Curve & Corporate Rate Risk")

    left, right = st.columns(2)
    with left:
        st.markdown("**Bond calculator**")
        face = st.number_input("Face", 100.0, key="fi_face")
        cpn = st.number_input("Coupon rate", 0.0, 0.20, 0.05, 0.0025, format="%.4f", key="fi_cpn")
        ytm = st.number_input("YTM", 0.0, 0.30, 0.06, 0.0025, format="%.4f", key="fi_ytm")
        yrs = st.number_input("Years to maturity", 0.5, 50.0, 10.0, 0.5, key="fi_yrs")
        freq = st.selectbox("Coupon freq", [1, 2, 4], index=1, key="fi_freq")
        d = fi.bond_duration(face, cpn, ytm, yrs, freq)
        st.metric("Price", fmt(d.price))
        cA, cB, cC = st.columns(3)
        cA.metric("Macaulay Dur", fmt(d.macaulay_duration))
        cB.metric("Modified Dur", fmt(d.modified_duration))
        cC.metric("Convexity", fmt(d.convexity))
        shock = st.slider("Yield shock (bps)", -300, 300, 100, 25, key="fi_shock")
        chg = d.price_change_pct(shock / 10_000.0)
        st.metric(f"Est. ΔPrice for {shock:+d}bp", fmt(chg, pct=True))

    with right:
        st.markdown("**Arbitrage-free spot curve**")
        par_txt = st.text_input("Par yields (comma %, by maturity)",
                                value="3.0, 3.3, 3.6, 3.8, 4.0", key="fi_par")
        try:
            pars = [float(x) / 100 for x in par_txt.split(",") if x.strip()]
            spots = fi.bootstrap_spot_rates(pars)
            curve = pd.DataFrame({
                "Maturity": list(range(1, len(spots) + 1)),
                "Par yield": pars, "Spot rate": spots})
            st.dataframe(curve.style.format({"Par yield": "{:.2%}", "Spot rate": "{:.2%}"}),
                         width="stretch")
        except ValueError:
            st.warning("Enter comma-separated numbers, e.g. 3.0, 3.3, 3.6")

    st.divider()
    st.markdown("**Corporate debt → interest-rate & credit risk**")
    sym = picker("Company", "fi")
    if sym:
        rs = st.slider("Rate shock (bps)", -300, 300, 100, 25, key="fi_corp_shock")
        mat = st.slider("Assumed avg maturity (yrs)", 1.0, 20.0, 7.0, 0.5, key="fi_mat")
        prof = fi.analyze_debt_structure(sym, rate_shock_bps=rs,
                                         assumed_maturity_years=mat,
                                         statements=c_statements(sym))
        g = st.columns(4)
        g[0].metric("Total debt", fmt(prof.total_debt, money=True, dp=0))
        g[1].metric("Implied coupon", fmt(prof.implied_coupon, pct=True))
        g[2].metric("Modified Dur", fmt(prof.modified_duration))
        g[3].metric(f"Δ debt value ({rs:+d}bp)", fmt(prof.value_change_on_shock, money=True, dp=0))
        g2 = st.columns(3)
        g2[0].metric("Interest coverage", fmt(prof.interest_coverage))
        g2[1].metric("Debt / EBITDA", fmt(prof.debt_to_ebitda))
        g2[2].metric("Credit bucket", prof.rating_bucket if prof.rating_bucket else "—")
        if prof.notes:
            st.caption(prof.notes)
        payload = (f"Total debt={prof.total_debt}, implied_coupon={prof.implied_coupon}, "
                   f"modD={prof.modified_duration}, Δvalue@{rs}bp={prof.value_change_on_shock}, "
                   f"int_cov={prof.interest_coverage}, debt/EBITDA={prof.debt_to_ebitda}, "
                   f"bucket={prof.rating_bucket}")
        deconstruct_button("fi", "Fixed Income & Credit", sym, payload)


# --------------------------------------------------------------------------
# FORENSICS / FSA
# --------------------------------------------------------------------------
with tab_fsa:
    st.subheader("Forensic Accounting & Statement Analysis")
    sym = picker("Company", "fsa")
    if sym:
        st.markdown("**Beneish M-Score** (earnings-manipulation detector)")
        m = fsa.beneish_m_score(sym, statements=c_statements(sym))
        if np.isfinite(m.m_score):
            verdict = ("🚩 Above −1.78 → elevated manipulation risk"
                       if m.likely_manipulator else "✅ Below −1.78 → low manipulation flag")
            st.metric("M-Score", fmt(m.m_score), help="M > −1.78 flags likely manipulators")
            st.write(verdict)
            comp = pd.DataFrame([m.components]).T.rename(columns={0: "Value"})
            st.dataframe(comp.style.format({"Value": "{:.4f}"}, na_rep="—"),
                         width="stretch")
            if m.notes:
                st.caption(m.notes)
            deconstruct_button("fsa", "Beneish M-Score / FSA", sym,
                               f"M-Score={m.m_score:.3f}; components={m.components}; {m.notes}")
        else:
            st.warning(m.notes or "Insufficient statement history for M-Score.")

        st.divider()
        cset = st.columns(2)
        with cset[0]:
            st.markdown("**FX translation** (Study calculator)")
            na = st.number_input("Net assets (local)", value=1000.0)
            ni = st.number_input("Net income (local)", value=200.0)
            mna = st.number_input("Net monetary assets (local)", value=400.0)
            cr = st.number_input("Current rate", value=1.10, format="%.4f")
            ar = st.number_input("Average rate", value=1.12, format="%.4f")
            hr = st.number_input("Historical rate", value=1.20, format="%.4f")
            method = st.radio("Method", ["current_rate", "temporal"], horizontal=True)
            fx = fsa.fx_translate(na, ni, mna, cr, ar, hr, method)
            st.write(f"**{fx.method}** — adjustment: `{fx.adjustment:,.2f}`")
            st.json(fx.translated)
        with cset[1]:
            st.markdown("**Pension funded status**")
            pbo = st.number_input("PBO", value=1200.0)
            assets = st.number_input("Plan assets", value=1000.0)
            p = fsa.pension_funded_status(pbo, assets)
            st.metric("Funded status", fmt(p.funded_status, money=True, dp=0))
            st.write(p.status_label)
            st.markdown("**Intercorporate investment**")
            own = st.slider("Ownership %", 0, 100, 25) / 100
            st.info(fsa.classify_investment(own))


# --------------------------------------------------------------------------
# PORTFOLIO RISK
# --------------------------------------------------------------------------
with tab_pm:
    st.subheader(f"Active Risk vs. {bench}")
    sym = picker("Ticker", "pm")
    period = st.selectbox("Window", ["6mo", "1y", "2y", "5y"], index=1)
    if sym:
        rsum = pm.risk_summary(sym, bench, period)
        g = st.columns(4)
        g[0].metric("Beta", fmt(rsum.beta))
        g[1].metric("Alpha (ann.)", fmt(rsum.alpha_annual, pct=True))
        g[2].metric("R²", fmt(rsum.r_squared))
        g[3].metric("Sharpe", fmt(rsum.sharpe))
        g2 = st.columns(4)
        g2[0].metric("Total vol (ann.)", fmt(rsum.total_vol, pct=True))
        g2[1].metric("Idiosyncratic vol", fmt(rsum.idiosyncratic_vol, pct=True))
        g2[2].metric("Tracking error", fmt(rsum.tracking_error, pct=True))
        g2[3].metric("Information ratio", fmt(rsum.information_ratio))
        st.caption(f"{rsum.n_obs} observations over {period}.")

        holdings = store.list_holdings()
        if holdings:
            with st.expander("📦 Portfolio-level active risk (from your holdings)"):
                weights = {}
                for h in holdings:
                    q = c_quote(h.ticker)
                    mv = (q.last_price * h.size) if np.isfinite(q.last_price) else 0.0
                    weights[h.ticker] = mv
                stats = pm.portfolio_active_stats(weights, bench, period)
                if stats:
                    cc = st.columns(4)
                    cc[0].metric("Portfolio beta", fmt(stats.get("beta")))
                    cc[1].metric("Tracking error", fmt(stats.get("tracking_error"), pct=True))
                    cc[2].metric("Information ratio", fmt(stats.get("information_ratio")))
                    cc[3].metric("Sharpe", fmt(stats.get("sharpe")))

        payload = (f"beta={rsum.beta}, alpha_ann={rsum.alpha_annual}, R2={rsum.r_squared}, "
                   f"idio_vol={rsum.idiosyncratic_vol}, TE={rsum.tracking_error}, "
                   f"IR={rsum.information_ratio}, sharpe={rsum.sharpe}, n={rsum.n_obs}")
        deconstruct_button("pm", "Portfolio Construction / Active Risk", sym, payload)


# --------------------------------------------------------------------------
# GREEKS
# --------------------------------------------------------------------------
with tab_grk:
    st.subheader("Option Greeks (Black-Scholes-Merton)")
    mode_grk = st.radio("Source", ["Live chain (ATM)", "Manual calculator"],
                        horizontal=True)
    if mode_grk == "Live chain (ATM)":
        sym = picker("Underlying", "grk")
        otype = st.radio("Type", ["call", "put"], horizontal=True)
        expiries = de.list_expiries(sym) if sym else []
        exp = st.selectbox("Expiry", expiries) if expiries else None
        if sym:
            g = dv.implied_greeks_from_chain(sym, otype, exp)
            if g is None:
                st.warning("No option chain / implied vol available for this name.")
            else:
                c = st.columns(3)
                c[0].metric("Spot", fmt(g.spot))
                c[1].metric("Strike (ATM)", fmt(g.strike))
                c[2].metric("IV", fmt(g.iv, pct=True))
                c2 = st.columns(5)
                c2[0].metric("Delta", fmt(g.delta, dp=3))
                c2[1].metric("Gamma", fmt(g.gamma, dp=4))
                c2[2].metric("Vega /1%", fmt(g.vega_per_1pct, dp=3))
                c2[3].metric("Theta /day", fmt(g.theta_per_day, dp=3))
                c2[4].metric("Rho /1%", fmt(g.rho_per_1pct, dp=3))
                st.caption(f"~{g.days_to_expiry:.0f} days to expiry · BSM price {g.price:.2f}")
                deconstruct_button("grk", "Option Greeks", sym,
                                   f"{otype} ATM K={g.strike} S={g.spot} IV={g.iv} "
                                   f"Δ={g.delta} Γ={g.gamma} V={g.vega_per_1pct} "
                                   f"Θ/day={g.theta_per_day} ρ={g.rho_per_1pct}")
    else:
        c = st.columns(3)
        S = c[0].number_input("Spot S", value=100.0)
        K = c[1].number_input("Strike K", value=100.0)
        Tdays = c[2].number_input("Days to expiry", value=30.0)
        c2 = st.columns(3)
        rr = c2[0].number_input("Risk-free r", value=float(c_riskfree()), format="%.4f")
        sig = c2[1].number_input("Volatility σ", value=0.25, format="%.4f")
        ot = c2[2].radio("Type", ["call", "put"], horizontal=True)
        g = dv.bsm_greeks(S, K, Tdays / 365.0, rr, sig, 0.0, ot)
        cc = st.columns(6)
        cc[0].metric("Price", fmt(g.price))
        cc[1].metric("Delta", fmt(g.delta, dp=3))
        cc[2].metric("Gamma", fmt(g.gamma, dp=4))
        cc[3].metric("Vega /1%", fmt(g.vega_per_1pct, dp=3))
        cc[4].metric("Theta /day", fmt(g.theta_per_day, dp=3))
        cc[5].metric("Rho /1%", fmt(g.rho_per_1pct, dp=3))
