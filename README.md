# 📊 The Universal CFA Investor Terminal

A local Streamlit app that doubles as **(1)** a CFA Level-2 study tool and
**(2)** a real-world watchlist/portfolio tracker. It reads market data from
yfinance, persists your watchlists/holdings to a local `portfolio_data.json`,
and can switch its whole lens between **🎓 CFA Study Mode** and
**💰 Active Investor Mode**.

## Features

| Tab | What it does |
|-----|--------------|
| 📡 **Tracker** | Your shared image: Last Price vs 1-Yr analyst target, **Expected %** (green/red heatmap), and **Δ Expected %** vs the last saved reading. Optional ATM greeks column. |
| 🧮 **Equity Valuation** | DDM, FCFE, FCFF, and **Residual Income** (single + multistage with persistence ω). Auto-pulls fundamentals; every assumption is editable. |
| 🏦 **Fixed Income** | Bond price / Macaulay & modified duration / convexity, a yield-shock estimator, spot-curve bootstrapping (arbitrage-free), and a corporate **debt → rate-risk + credit** mapper. |
| 🔬 **Forensics (FSA)** | **Beneish M-Score** (8-factor manipulation detector), FX translation (current-rate vs temporal + CTA), pension funded status, intercorporate-investment classification. |
| 📈 **Portfolio Risk** | Beta, alpha, R², idiosyncratic vol, **tracking error**, **information ratio**, Sharpe — single name and portfolio-weighted vs a benchmark (default SPY). |
| 🎲 **Greeks** | Black-Scholes-Merton greeks from the live option chain (ATM) or a manual calculator. |

Everywhere: a **💡 ELI5 Concept Decoder** (sidebar) and a **🧩 Deconstruct**
button that explain any term/output in two tiers — a 5-year-old analogy, then
the exact CFA reality. These use the Anthropic API (the rest works offline).

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
storage.py           JSON CRUD: holdings, watchlist, snapshots (Δ Expected %)
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

## Notes & caveats

- Data is delayed/though-the-day from Yahoo; analyst targets & fundamentals come
  from `.info`, which is occasionally sparse — fields render as `—` when missing.
- The corporate debt model treats total debt as one synthetic bullet bond; the
  credit-rating bucket is a heuristic, **not** an agency rating.
- Greeks use Yahoo's posted implied vol; for thin options chains that can be
  noisy. Not investment advice.
