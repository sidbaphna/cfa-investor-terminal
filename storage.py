"""storage.py — Local JSON persistence for portfolios, watchlists, and snapshots.

Two "tables" per the spec, plus a snapshot log that powers the tracker's
"Change in Expected %" column:

    holdings   : My Portfolio Holdings  (Ticker, Cost Basis, Size, Date Added)
    watchlist  : Watchlist Terminal     (Ticker, Date Added, Target Strike)
    snapshots  : {ticker: [{date, price, target, expected_pct}, ...]}
    settings   : misc UI/engine settings (benchmark, discount rate, ...)

Design notes
------------
* Pure standard-library; no Streamlit import (keeps the layer reusable/testable).
* Writes are atomic (temp file + os.replace) so a crash never corrupts state.
* Every public method has strict type hints and degrades gracefully on bad data.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from config import DATA_FILE, DEFAULT_BENCHMARK, DEFAULT_DISCOUNT_RATE


# --------------------------------------------------------------------------
# Record types
# --------------------------------------------------------------------------
@dataclass
class Holding:
    """A position you actually own."""
    ticker: str
    cost_basis: float          # average price paid per share
    size: float                # number of shares
    date_added: str = field(default_factory=lambda: date.today().isoformat())

    def normalized(self) -> "Holding":
        return Holding(self.ticker.strip().upper(), float(self.cost_basis),
                       float(self.size), self.date_added)


@dataclass
class WatchItem:
    """A ticker you're tracking but don't own. Target Strike = your thesis price."""
    ticker: str
    date_added: str = field(default_factory=lambda: date.today().isoformat())
    target_strike: Optional[float] = None

    def normalized(self) -> "WatchItem":
        strike = None if self.target_strike in (None, "") else float(self.target_strike)
        return WatchItem(self.ticker.strip().upper(), self.date_added, strike)


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
class PortfolioStore:
    """CRUD wrapper over a single JSON file. Instantiate once, reuse."""

    def __init__(self, path: str = DATA_FILE) -> None:
        self.path = path
        self._data: Dict[str, Any] = self._load()

    # ---- low-level I/O ---------------------------------------------------
    def _empty(self) -> Dict[str, Any]:
        return {
            "holdings": [],
            "watchlist": [],
            "snapshots": {},
            "settings": {
                "benchmark": DEFAULT_BENCHMARK,
                "discount_rate": DEFAULT_DISCOUNT_RATE,
            },
        }

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return self._empty()
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable file: start clean rather than crash the app.
            return self._empty()
        # Backfill any missing top-level keys (forward/backward compatibility).
        base = self._empty()
        base.update({k: v for k, v in data.items() if k in base})
        for k in base:
            if k not in data:
                data[k] = base[k]
        return data

    def save(self) -> None:
        """Atomic write: never leaves a half-written file behind."""
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    # ---- Holdings CRUD ---------------------------------------------------
    def list_holdings(self) -> List[Holding]:
        out: List[Holding] = []
        for row in self._data.get("holdings", []):
            try:
                out.append(Holding(**row).normalized())
            except (TypeError, ValueError):
                continue
        return out

    def add_holding(self, ticker: str, cost_basis: float, size: float,
                    date_added: Optional[str] = None) -> Holding:
        h = Holding(ticker, cost_basis, size,
                    date_added or date.today().isoformat()).normalized()
        rows = [r for r in self._data["holdings"]
                if r.get("ticker", "").upper() != h.ticker]   # upsert by ticker
        rows.append(asdict(h))
        self._data["holdings"] = rows
        self.save()
        return h

    def remove_holding(self, ticker: str) -> bool:
        t = ticker.strip().upper()
        before = len(self._data["holdings"])
        self._data["holdings"] = [r for r in self._data["holdings"]
                                  if r.get("ticker", "").upper() != t]
        changed = len(self._data["holdings"]) != before
        if changed:
            self.save()
        return changed

    # ---- Watchlist CRUD --------------------------------------------------
    def list_watchlist(self) -> List[WatchItem]:
        out: List[WatchItem] = []
        for row in self._data.get("watchlist", []):
            try:
                out.append(WatchItem(**row).normalized())
            except (TypeError, ValueError):
                continue
        return out

    def add_watch(self, ticker: str, target_strike: Optional[float] = None,
                  date_added: Optional[str] = None) -> WatchItem:
        w = WatchItem(ticker, date_added or date.today().isoformat(),
                      target_strike).normalized()
        rows = [r for r in self._data["watchlist"]
                if r.get("ticker", "").upper() != w.ticker]   # upsert by ticker
        rows.append(asdict(w))
        self._data["watchlist"] = rows
        self.save()
        return w

    def remove_watch(self, ticker: str) -> bool:
        t = ticker.strip().upper()
        before = len(self._data["watchlist"])
        self._data["watchlist"] = [r for r in self._data["watchlist"]
                                   if r.get("ticker", "").upper() != t]
        changed = len(self._data["watchlist"]) != before
        if changed:
            self.save()
        return changed

    def all_tickers(self) -> List[str]:
        """Unique tickers across both tables — what the tracker iterates over."""
        seen: Dict[str, None] = {}
        for h in self.list_holdings():
            seen[h.ticker] = None
        for w in self.list_watchlist():
            seen[w.ticker] = None
        return list(seen.keys())

    # ---- Snapshots (powers "Change in Expected %") -----------------------
    def record_snapshot(self, ticker: str, price: float, target: float,
                        expected_pct: float, on: Optional[str] = None) -> None:
        """Append today's reading. One row per (ticker, date) — re-running
        the tracker on the same day overwrites that day's row."""
        t = ticker.strip().upper()
        day = on or date.today().isoformat()
        log = self._data["snapshots"].setdefault(t, [])
        log[:] = [r for r in log if r.get("date") != day]
        log.append({"date": day, "price": float(price),
                    "target": float(target), "expected_pct": float(expected_pct)})
        log.sort(key=lambda r: r.get("date", ""))
        self._data["snapshots"][t] = log[-90:]   # keep last ~90 readings

    def previous_expected_pct(self, ticker: str) -> Optional[float]:
        """Most recent expected_pct from a *prior* day, for delta computation."""
        t = ticker.strip().upper()
        today = date.today().isoformat()
        prior = [r for r in self._data["snapshots"].get(t, [])
                 if r.get("date") != today]
        return float(prior[-1]["expected_pct"]) if prior else None

    def snapshot_history(self, ticker: str) -> List[Dict[str, Any]]:
        return list(self._data["snapshots"].get(ticker.strip().upper(), []))

    # ---- Settings --------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        return self._data.get("settings", {}).get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        self._data.setdefault("settings", {})[key] = value
        self.save()


if __name__ == "__main__":  # tiny smoke test
    store = PortfolioStore("portfolio_data.demo.json")
    store.add_holding("AAPL", 150.0, 10)
    store.add_watch("FICO", target_strike=1600.0)
    store.record_snapshot("FICO", 1261.17, 1531.52, 0.214)
    print("Holdings :", store.list_holdings())
    print("Watchlist:", store.list_watchlist())
    print("Tickers  :", store.all_tickers())
    os.remove("portfolio_data.demo.json")
