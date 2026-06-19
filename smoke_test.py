"""smoke_test.py — run app.py end-to-end via Streamlit's AppTest harness.

Covers four paths, asserting no Python exception bubbles up:
  1. Open app (no password)        — every tab renders, both modes.
  2. Login gate (password set)     — app halts at the password prompt.
  3. DB backend (sqlite CFA_DB_URL)— app boots on the durable SQL backend.
Uses isolated temp files so it never touches your real portfolio.
Network calls degrade to empty/NaN, so it passes with or without internet.
"""
import json
import os
import tempfile

from streamlit.testing.v1 import AppTest


def _seed_datafile():
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump({
        "holdings": [{"ticker": "AAPL", "cost_basis": 150.0, "size": 5,
                      "date_added": "2026-06-18"}],
        "watchlist": [{"ticker": "AAPL", "date_added": "2026-06-18",
                       "target_strike": 250.0}],
        "snapshots": {}, "settings": {"benchmark": "SPY", "discount_rate": 0.09},
    }, f)
    f.close()
    return f.name


# 1. Open app — JSON backend, no password ----------------------------------
os.environ.pop("CFA_APP_PASSWORD", None)
os.environ.pop("CFA_DB_URL", None)
os.environ["CFA_DATA_FILE"] = _seed_datafile()
at = AppTest.from_file("app.py", default_timeout=240)
at.run()
assert not at.exception, f"OPEN exception: {at.exception}"
assert len(at.tabs) == 8, f"expected 8 tabs, got {len(at.tabs)}"
at.radio[0].set_value("💰 Active Investor Mode").run()
assert not at.exception, f"ACTIVE exception: {at.exception}"
print(f"✅ Open app — {len(at.tabs)} tabs, both modes, no exceptions")

# Research sub-views — drive the View radio through every option (exercises the
# peer table, calendar/FRED macro, and the Plotly technicals chart).
for opt in ["⚖️ Peer comparison", "📅 Calendar & Macro", "📈 Technicals",
            "🧾 Financials"]:
    rv = next((r for r in at.radio if r.key == "research_view"), None)
    assert rv is not None, "research_view radio not found"
    rv.set_value(opt).run()
    assert not at.exception, f"Research[{opt}] exception: {at.exception}"
print("✅ Research views — financials / peers / calendar / technicals, no exceptions")

# 2. Login gate — password set should halt before the tabs -----------------
os.environ["CFA_APP_PASSWORD"] = "test-secret"
at2 = AppTest.from_file("app.py", default_timeout=60)
at2.run()
assert not at2.exception, f"GATE exception: {at2.exception}"
assert len(at2.tabs) == 0, "gate should stop before tabs render"
assert len(at2.text_input) >= 1, "password field should be present"
print("✅ Login gate — app halts at password prompt (no tabs rendered)")
os.environ.pop("CFA_APP_PASSWORD", None)

# 3. DB backend — boot on a temp SQLite database ---------------------------
_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_db.close()
os.environ["CFA_DB_URL"] = f"sqlite:///{_db.name}"
at3 = AppTest.from_file("app.py", default_timeout=240)
at3.run()
assert not at3.exception, f"DB-BACKEND exception: {at3.exception}"
assert any("database" in c.value for c in at3.caption), \
    "sidebar should report the database backend"
print("✅ DB backend — app boots on SQLite, storage caption reports 'database'")

os.remove(_db.name)
print("🎉 App smoke test passed.")
