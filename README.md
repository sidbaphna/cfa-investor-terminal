# 📊 The Universal CFA Investor Terminal

A local Streamlit app that doubles as **(1)** a CFA Level-2 study tool and
**(2)** a real-world watchlist/portfolio tracker. It reads market data from
yfinance, persists your watchlists/holdings to a local `portfolio_data.json`,
and can switch its whole lens between **🎓 CFA Study Mode** and
**💰 Active Investor Mode**.

## Features

| Tab | What it does |
|-----|--------------|
| 📡 **Tracker** | Your shared image: Last Price vs 1-Yr analyst target, **Expected %** (green/red heatmap), **Δ Expected %** vs the last saved reading, and an **As of** quote-time column. Optional ATM greeks. |
| 📰 **News & Sectors** | Free Yahoo Finance headlines (no key) — Markets / U.S. economy / World / per-**Sector** (with sector-ETF momentum) / per-watchlist-ticker, with article links. Optional AI "how it affects my investments" read when a key is set. |
| 🔭 **Research** | **Financials explorer** (statements with YoY growth, common-size %, key ratios), **peer comparison** (P/E, P/B, EV/EBITDA, margins, growth vs peers), and **Calendar & Macro** (earnings/ex-div dates + a free **FRED** macro snapshot: Fed funds, 2Y/10Y, unemployment, CPI YoY, real GDP). |
| 🧮 **Equity Valuation** | DDM, FCFE, FCFF, and **Residual Income** (single + multistage with persistence ω). Auto-pulls fundamentals; every assumption is editable. |
| 🏦 **Fixed Income** | Bond price / Macaulay & modified duration / convexity, a yield-shock estimator, spot-curve bootstrapping (arbitrage-free), and a corporate **debt → rate-risk + credit** mapper. |
| 🔬 **Forensics (FSA)** | **Beneish M-Score** (8-factor manipulation detector), FX translation (current-rate vs temporal + CTA), pension funded status, intercorporate-investment classification. |
| 📈 **Portfolio Risk** | Beta, alpha, R², idiosyncratic vol, **tracking error**, **information ratio**, Sharpe — single name and portfolio-weighted vs a benchmark (default SPY). |
| 🎲 **Greeks** | Black-Scholes-Merton greeks from the live option chain (ATM) or a manual calculator. |

Everywhere: a **💡 ELI5 Concept Decoder** (sidebar) and a **🧩 Deconstruct**
button that explain any term/output in two tiers — a 5-year-old analogy, then
the exact CFA reality.

**ELI5 without an API key:** common terms (OAS, Residual Income, Duration, Vega,
Beneish M-Score, …) are answered instantly from a built-in **offline glossary**
(`glossary.py`) — no key required, works on the deployed app. With a key, the LLM
handles *any* term contextually. For free dynamic answers on any term locally,
point it at a local model (e.g. Ollama) — see below.

## Install

```bash
cd cfa_investor_terminal
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the ELI5 / Deconstruct features, set an Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# optional: trade cost for speed
export CFA_LLM_MODEL="claude-opus-4-8"   # or claude-sonnet-4-6 / claude-haiku-4-5
export CFA_LLM_EFFORT="medium"           # low | medium | high | max
```

## Run

```bash
streamlit run app.py
```

It opens at http://localhost:8501. Add tickers in the sidebar to populate the
tracker, or seed it with the screenshot's names in one step:

```bash
cp portfolio_data.example.json portfolio_data.json   # APP, FICO, ORCL, SMCI
```

## Architecture

```
config.py            shared constants (model id, benchmark, rates)
storage.py           pluggable persistence (local JSON OR SQL/Postgres): holdings, watchlist, snapshots
data_engine.py       yfinance harvester: quotes, statements, prices, options, fundamentals
topics/
  equity_valuation.py   DDM / FCFE / FCFF / Residual Income
  fixed_income.py       duration, convexity, spot curve, corporate credit
  fsa.py                Beneish M-Score, FX translation, pension, intercorporate
  portfolio_mgmt.py     beta, idiosyncratic risk, tracking error, info ratio
  derivatives.py        Black-Scholes greeks (the 'greeks' tracker feature)
llm_orchestrator.py  Anthropic prompt routers: ELI5 + Deep Analysis
app.py               Streamlit UI (tabs, sidebar, mode toggle)
```

`topics/*` and `data_engine.py` never import Streamlit, so each engine is
reusable and unit-testable on its own. To add a CFA topic, drop a new module in
`topics/` and add a tab in `app.py` — nothing else needs to change.

## Deploy for remote / phone access (Streamlit Community Cloud)

1. Push to GitHub, then sign in at **https://share.streamlit.io** with GitHub.
2. **Create app** → your repo, branch `main`, **Main file path = `app.py`**
   (not the default `streamlit_app.py`), then Deploy.
3. In the app's **Settings → Secrets**, paste (TOML):

   ```toml
   CFA_APP_PASSWORD = "choose-a-strong-password"   # turns on the login gate
   CFA_DB_URL = "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"  # durable storage
   ANTHROPIC_API_KEY = "sk-ant-..."                 # optional: ELI5 / Deconstruct
   ```

**Login gate** — set `CFA_APP_PASSWORD` and the app shows a password prompt before
anything else (good for a public URL you open from your phone). Unset = open.

**Durable storage (`CFA_DB_URL`)** — Streamlit Cloud's filesystem is *ephemeral*:
a local `portfolio_data.json` is wiped on every restart/redeploy. Point
`CFA_DB_URL` at a **free hosted Postgres** so your watchlist/holdings survive:

- **Neon** (https://neon.tech) or **Supabase** (https://supabase.com) → create a
  free project → copy the connection string → ensure the scheme is
  `postgresql+psycopg2://…`. The app auto-creates its single table on first run.

The store auto-selects: `CFA_DB_URL` set → SQL backend; otherwise the JSON file.
The active backend is shown in the sidebar ("Storage: …"). Locally you need
neither — it uses the JSON file with no login.

## Notes & caveats

- Data is delayed/though-the-day from Yahoo; analyst targets & fundamentals come
  from `.info`, which is occasionally sparse — fields render as `—` when missing.
- The corporate debt model treats total debt as one synthetic bullet bond; the
  credit-rating bucket is a heuristic, **not** an agency rating.
- Greeks use Yahoo's posted implied vol; for thin options chains that can be
  noisy. Not investment advice.
