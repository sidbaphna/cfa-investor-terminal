"""glossary.py — offline ELI5 explanations for common terms.

Lets the ELI5 Decoder work with NO API key for curated terms (same two-tier
format the LLM uses). Streamlit-free; pure data + a fuzzy lookup.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

# Each entry uses the same two-tier contract as the LLM path.
GLOSSARY: Dict[str, str] = {
    "option-adjusted spread": """### Level 1 — The 5-Year-Old Analogy
Imagine two lemonade stands promise to pay you back with interest. One stand
can decide to stop paying early if it feels like it (an "option"). To compare
them fairly, you first subtract the value of that sneaky early-stop trick — what's
left is the *real* extra reward you're getting.

### Level 2 — The CFA Reality
The **Option-Adjusted Spread (OAS)** is the constant spread added to the
benchmark (Treasury) spot curve that makes a bond's model price equal its market
price, *after* removing the value of embedded options (call/put/prepayment) via a
binomial tree or Monte Carlo. OAS = Z-spread − option cost, so it isolates
compensation for credit/liquidity risk and is comparable across bonds with
different optionality.""",

    "residual income": """### Level 1 — The 5-Year-Old Analogy
You lend your friend $10 to run a lemonade stand and expect $1 back as a fair
rent for your money. If the stand earns $1.50, the extra **$0.50** is true
"bonus" profit — value created above what you required.

### Level 2 — The CFA Reality
**Residual Income** RIₜ = Eₜ − r·Bₜ₋₁ = (ROE − r)·Bₜ₋₁ — earnings minus a charge
for equity capital. Intrinsic value V₀ = B₀ + Σ RIₜ/(1+r)ᵗ. A firm only adds
value when ROE > r; the model relies on the clean-surplus relation
(Bₜ = Bₜ₋₁ + Eₜ − Dₜ).""",

    "modified duration": """### Level 1 — The 5-Year-Old Analogy
A seesaw: when interest rates (one side) go up a little, the bond's price (other
side) goes down. Modified duration tells you *how much* the price end drops for
each small push on the rate end.

### Level 2 — The CFA Reality
**Modified Duration** ≈ −(1/P)·(dP/dy), the % price change for a 1% (100bp) yield
move: ModDur = MacaulayDuration / (1 + y/k). First-order (linear) estimate;
pair with **convexity** for the curvature correction:
ΔP/P ≈ −ModDur·Δy + ½·Convexity·Δy².""",

    "convexity": """### Level 1 — The 5-Year-Old Analogy
The seesaw isn't a straight plank — it's curved. So the simple "how much it
tips" rule is a little off for big pushes. Convexity is that curve, and it's good
for you: prices fall less and rise more than the straight rule predicts.

### Level 2 — The CFA Reality
**Convexity** is the second derivative of price w.r.t. yield, scaled by price. It
corrects duration's linear approximation: positive convexity means a bond gains
more when yields fall than it loses when yields rise by the same amount — a
desirable, priced-in property (callable bonds can show negative convexity).""",

    "tracking error": """### Level 1 — The 5-Year-Old Analogy
You're trying to walk in step with a friend (the benchmark). Tracking error is
how much your steps wander away from theirs — wobbly = high, glued-to-their-side
= low.

### Level 2 — The CFA Reality
**Tracking Error** = the annualized standard deviation of active returns
(portfolio − benchmark): σ(Rₚ − R_b)·√252. It measures how tightly a portfolio
hugs its benchmark; the numerator of the Information Ratio.""",

    "information ratio": """### Level 1 — The 5-Year-Old Analogy
For every step you wander off your friend's path (risk), how much faster did you
get to the ice-cream truck (reward)? More ice cream per wobble = better.

### Level 2 — The CFA Reality
**Information Ratio** = active return / tracking error =
mean(Rₚ − R_b)·252 / [σ(Rₚ − R_b)·√252]. It's risk-adjusted active performance —
the manager's skill per unit of benchmark-relative risk.""",

    "vega": """### Level 1 — The 5-Year-Old Analogy
A weather forecast that says "storms might come" makes umbrella prices jump. Vega
is how much an option's price changes when people get more or less nervous about
future swings — not the weather itself, just the *expected wildness*.

### Level 2 — The CFA Reality
**Vega** = ∂(option price)/∂σ — sensitivity to implied volatility. In
Black-Scholes, Vega = S·e^(−qT)·φ(d₁)·√T, identical for calls and puts, largest
for at-the-money options with more time to expiry. Usually quoted per 1-point
(1%) change in IV.""",

    "delta": """### Level 1 — The 5-Year-Old Analogy
If a toy's price goes up by $1, how much does a "coupon for that toy" go up? Delta
is that follow-along amount — near $1 if you're sure to use the coupon, near $0 if
you probably won't.

### Level 2 — The CFA Reality
**Delta** = ∂(option price)/∂S, the hedge ratio. For a call, Δ = e^(−qT)·N(d₁)
(0→1); for a put it's negative (−1→0). Also a rough risk-neutral probability the
option finishes in the money; the basis of delta-hedging.""",

    "gamma": """### Level 1 — The 5-Year-Old Analogy
Delta is your speed; gamma is how fast your speed itself changes. A twitchy car
where a tiny tap on the gas changes your speed a lot has high gamma.

### Level 2 — The CFA Reality
**Gamma** = ∂²(option price)/∂S² = ∂Δ/∂S — the rate of change of delta. Highest
for at-the-money, near-expiry options. High gamma means a delta hedge needs
frequent rebalancing.""",

    "clean surplus relation": """### Level 1 — The 5-Year-Old Analogy
Your piggy bank's new total = old total + money you earned − money you spent.
Nothing magically appears or disappears outside that rule.

### Level 2 — The CFA Reality
**Clean Surplus Relation**: Bₜ = Bₜ₋₁ + Eₜ − Dₜ — ending book equity equals
beginning plus earnings minus dividends, with *all* gains/losses flowing through
the income statement. It's the assumption that makes the Residual Income model
internally consistent; violations (items booked directly to equity / OCI under
IFRS or US GAAP) break it.""",

    "persistence factor": """### Level 1 — The 5-Year-Old Analogy
A really good lemonade stand makes "bonus" profit, but copycats move in and the
bonus fades over time. The persistence factor (ω) is how slowly that bonus fades —
1 means it lasts forever, 0 means it vanishes next year.

### Level 2 — The CFA Reality
The **persistence factor (ω, 0–1)** scales how long residual income endures in a
multistage RI model. Continuing value = RIₜ₊₁ / (1 + r − ω): ω→1 implies RI
persists indefinitely (strong moat), ω→0 implies it disappears after the forecast
horizon as competition erodes excess ROE toward the cost of equity.""",

    "beneish m-score": """### Level 1 — The 5-Year-Old Analogy
A teacher has 8 little "fishy behavior" checks for a student's homework (erasing
a lot, answers changing overnight…). Add them with a recipe; a high score means
"please double-check this work."

### Level 2 — The CFA Reality
The **Beneish M-Score** combines 8 ratios (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI,
TATA) measuring receivables, margins, asset quality, growth, depreciation, SG&A,
leverage, and accruals: M = −4.84 + 0.92·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
+ 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI. **M > −1.78** flags a
probable earnings manipulator. **DSRI** specifically = (Receivablesₜ/Salesₜ) /
(Receivablesₜ₋₁/Salesₜ₋₁); a big jump can signal revenue inflated via lax credit.""",

    "temporal method cumulative translation adjustment": """### Level 1 — The 5-Year-Old Analogy
Your friend in another country keeps a piggy bank in their money. To write its
value in *your* money you use exchange rates — but cash uses today's rate while
the toy they bought long ago keeps its old-day rate. The leftover difference is a
plug number.

### Level 2 — The CFA Reality
Under the **temporal method** (functional currency = parent's), monetary
items translate at the current rate and non-monetary items at historical rates;
the remeasurement gain/loss hits **net income**. The **Cumulative Translation
Adjustment (CTA)** is instead the plug under the **current-rate method**
(functional = local), where all assets/liabilities use the current rate and the
adjustment accumulates in **OCI/equity**, not income. (So a "temporal CTA" is a
bit of a contradiction — temporal produces a P&L remeasurement, current-rate
produces the CTA.)""",

    "sharpe ratio": """### Level 1 — The 5-Year-Old Analogy
For every bit of stomach-churn on the roller coaster (risk), how much fun did you
get (return above just sitting still)? More fun per churn = a better ride.

### Level 2 — The CFA Reality
**Sharpe Ratio** = (Rₚ − R_f) / σₚ — excess return over the risk-free rate per
unit of total volatility. Annualized here as (mean daily excess)·252 / (σ·√252).
Uses total risk (vs the Information Ratio's active risk).""",
}

# Aliases → canonical keys (so "OAS", "DSRI", "duration" resolve too)
_ALIASES = {
    "oas": "option-adjusted spread",
    "option adjusted spread": "option-adjusted spread",
    "ri": "residual income",
    "duration": "modified duration",
    "moddur": "modified duration",
    "ir": "information ratio",
    "dsri": "beneish m-score",
    "m-score": "beneish m-score",
    "m score": "beneish m-score",
    "beneish": "beneish m-score",
    "cta": "temporal method cumulative translation adjustment",
    "cumulative translation adjustment": "temporal method cumulative translation adjustment",
    "temporal method": "temporal method cumulative translation adjustment",
    "omega": "persistence factor",
    "persistence factor (ω)": "persistence factor",
    "clean surplus": "clean surplus relation",
    "sharpe": "sharpe ratio",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").strip().lower()).strip()


def lookup(term: str) -> Optional[str]:
    """Return the offline two-tier explanation for a term, or None if unknown."""
    key = _norm(term)
    if not key:
        return None
    if key in GLOSSARY:
        return GLOSSARY[key]
    if key in _ALIASES:
        return GLOSSARY[_ALIASES[key]]
    # substring / contains match (e.g. "modified duration of a bond")
    for k in GLOSSARY:
        if k in key or key in k:
            return GLOSSARY[k]
    for alias, canonical in _ALIASES.items():
        if alias in key:
            return GLOSSARY[canonical]
    return None


def available_terms() -> list:
    return sorted(GLOSSARY.keys())
