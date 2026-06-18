"""Central configuration and shared constants for the CFA Investor Terminal.

Keeping these in one place lets every subengine and the UI stay decoupled:
topics/* never import from app.py, they import from here (or take plain args).
"""
from __future__ import annotations

import os
from typing import Final

# --- Persistence ----------------------------------------------------------
DATA_FILE: Final[str] = os.environ.get("CFA_DATA_FILE", "portfolio_data.json")

# --- Markets --------------------------------------------------------------
DEFAULT_BENCHMARK: Final[str] = "SPY"          # tracking-error / beta reference
RISK_FREE_TICKER: Final[str] = "^IRX"          # 13-week T-bill yield (percent)
FALLBACK_RISK_FREE: Final[float] = 0.043        # used if ^IRX cannot be fetched
DEFAULT_DISCOUNT_RATE: Final[float] = 0.09      # required return r for valuation
TRADING_DAYS: Final[int] = 252

# --- LLM (Anthropic) ------------------------------------------------------
# Default to the most capable model; override with CFA_LLM_MODEL if you want
# to trade cost for speed (e.g. claude-sonnet-4-6 or claude-haiku-4-5).
LLM_MODEL: Final[str] = os.environ.get("CFA_LLM_MODEL", "claude-opus-4-8")
LLM_EFFORT: Final[str] = os.environ.get("CFA_LLM_EFFORT", "medium")  # low|medium|high|max

# --- App modes ------------------------------------------------------------
MODE_STUDY: Final[str] = "🎓 CFA Study Mode"
MODE_ACTIVE: Final[str] = "💰 Active Investor Mode"

# Two-tier ELI5 output contract (referenced by llm_orchestrator + UI labels)
ELI5_LEVELS: Final[tuple[str, str]] = (
    "Level 1 — The 5-Year-Old Analogy",
    "Level 2 — The CFA Reality",
)
