"""llm_orchestrator.py — Master prompt routers for ELI5 + CFA Deep Analysis.

Two entry points the UI calls:

  * eli5_explain(term, context)      → the two-tier decoder (analogy → CFA reality)
  * deep_analysis(topic, ...)        → a structured CFA-grade breakdown of a chart
                                       or calculation, mode-aware (study vs active)

Uses the Anthropic SDK (claude-opus-4-8 by default, adaptive thinking). If no
API key / SDK is present, returns a clear, self-explanatory fallback so the rest
of the terminal keeps working offline.
"""
from __future__ import annotations

import os
from typing import Optional

import glossary
from config import LLM_EFFORT, LLM_MODEL, MODE_STUDY

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

_CLIENT = None


def is_available() -> bool:
    """True when both the SDK and an API key are present."""
    if anthropic is None:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _client():
    global _CLIENT
    if _CLIENT is None and is_available():
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


def _complete(system: str, user: str, max_tokens: int = 2000) -> str:
    """Single-shot completion with graceful degradation."""
    client = _client()
    if client is None:
        return _offline_notice()
    try:
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": LLM_EFFORT},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "\n".join(parts).strip() or "_(empty response)_"
    except Exception as exc:  # network / auth / quota — never crash the UI
        return (f"⚠️ LLM call failed: `{type(exc).__name__}: {exc}`\n\n"
                "Check your `ANTHROPIC_API_KEY`, model access, and network.")


def _offline_notice() -> str:
    return (
        "🔌 **LLM features are offline.** Set `ANTHROPIC_API_KEY` and "
        "`pip install anthropic` to enable the ELI5 Decoder and Deep Analysis.\n\n"
        f"Configured model: `{LLM_MODEL}` (override with `CFA_LLM_MODEL`)."
    )


# --------------------------------------------------------------------------
# ELI5 Decoder — the two-tier contract
# --------------------------------------------------------------------------
_ELI5_SYSTEM = """You are a CFA charterholder and a brilliant teacher. A learner \
will give you a confusing financial term or phrase. You MUST answer in exactly \
this two-tier structure, using Markdown headers verbatim:

### Level 1 — The 5-Year-Old Analogy
Explain the concept with ONE simple, concrete physical analogy a child would get \
(a lemonade stand, trading cards, borrowing toys, a piggy bank, a seesaw...). \
2–4 sentences. No jargon at all.

### Level 2 — The CFA Reality
Map that exact analogy, piece by piece, onto the precise CFA curriculum \
definition. Name the formula, the standard, or the framework. Be technically \
correct and exam-precise. Note IFRS vs US GAAP differences if relevant.

Keep the whole answer tight. Do not add extra sections or preamble."""


def eli5_explain(term: str, context: str = "") -> str:
    """Decode one term/phrase into the Level 1 / Level 2 structure.

    With an API key → a contextual LLM answer for *any* term. Without one →
    a curated offline glossary answer (no key needed) for common terms.
    """
    term = (term or "").strip()
    if not term:
        return "_Type a term to decode (e.g. “Option-Adjusted Spread”)._"

    if not is_available():
        hit = glossary.lookup(term)
        if hit:
            return hit + "\n\n<sub>📖 Offline glossary — no API key needed.</sub>"
        sample = ", ".join(f"“{t.title()}”" for t in glossary.available_terms()[:6])
        return ("🔌 **No API key**, and “" + term + "” isn't in the offline "
                "glossary yet.\n\nOffline terms include: " + sample + ", …\n\n"
                "For *any* term, set `ANTHROPIC_API_KEY`, or run a free local model "
                "(see README → ELI5 offline).")

    user = f'Term to decode: "{term}"'
    if context.strip():
        user += f"\n\nWhere it appeared on screen (context): {context.strip()}"
    return _complete(_ELI5_SYSTEM, user, max_tokens=1500)


# --------------------------------------------------------------------------
# Deep Analysis — mode-aware "Deconstruct Topic"
# --------------------------------------------------------------------------
def _deep_system(mode: str) -> str:
    base = ("You are an elite quantitative analyst and CFA charterholder "
            "deconstructing a calculation or chart from an investment terminal. "
            "Be precise, cite the relevant CFA framework, and flag key "
            "assumptions and limitations.")
    if mode == MODE_STUDY:
        return (base + " STUDY MODE: emphasize formula derivations, the exact "
                "IFRS vs US GAAP treatment, and finish with ONE short "
                "item-set-style question (with the answer hidden under a "
                "**Answer:** line) based on the data shown.")
    return (base + " ACTIVE INVESTOR MODE: emphasize the practical signal — "
            "what it implies for positioning, momentum, valuation, and risk "
            "management. End with a crisp, actionable takeaway. Not financial "
            "advice.")


def deep_analysis(topic: str, ticker: str, payload: str,
                  mode: str = MODE_STUDY) -> str:
    """Explain a specific terminal output. `payload` is the data/numbers shown."""
    system = _deep_system(mode)
    user = (f"Topic area: {topic}\nTicker: {ticker}\n\n"
            f"Here is exactly what the terminal computed/displayed:\n{payload}\n\n"
            "Deconstruct it for me.")
    return _complete(system, user, max_tokens=3500)


# --------------------------------------------------------------------------
# News impact (key-optional — used by the News tab)
# --------------------------------------------------------------------------
def news_impact(headlines: str, holdings: str, mode: str = MODE_STUDY) -> str:
    """Summarize how a batch of headlines may bear on the user's book."""
    system = ("You are a CFA charterholder and macro strategist. Given recent "
              "headlines and the user's holdings/watchlist, write 3–6 tight "
              "bullets on how the news may affect their investments — sector and "
              "factor exposure, key risks, and what to watch next. End with one "
              "line of caution. This is analysis, not financial advice.")
    if mode == MODE_STUDY:
        system += " Briefly name the macro/finance concepts in play."
    user = f"Holdings / watchlist: {holdings or '(none specified)'}\n\nHeadlines:\n{headlines}"
    return _complete(system, user, max_tokens=1400)


if __name__ == "__main__":
    print("LLM available:", is_available())
    print(eli5_explain("Option-Adjusted Spread"))
