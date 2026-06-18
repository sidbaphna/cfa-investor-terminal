"""smoke_test.py — run app.py end-to-end via Streamlit's AppTest harness.

Renders every tab (in both modes) and asserts no Python exception bubbles up.
Uses an isolated temp data file so it never touches your real portfolio.
Network calls degrade to empty/NaN, so this passes with or without internet —
it fails only on a genuine code/integration error.
"""
import json
import os
import tempfile

# Isolate persistence + seed one ticker so the tracker's Styler path runs.
_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
json.dump({
    "holdings": [{"ticker": "AAPL", "cost_basis": 150.0, "size": 5,
                  "date_added": "2026-06-18"}],
    "watchlist": [{"ticker": "AAPL", "date_added": "2026-06-18",
                   "target_strike": 250.0}],
    "snapshots": {}, "settings": {"benchmark": "SPY", "discount_rate": 0.09},
}, _tmp)
_tmp.close()
os.environ["CFA_DATA_FILE"] = _tmp.name

from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file("app.py", default_timeout=240)

# Study mode
at.run()
assert not at.exception, f"STUDY MODE exception: {at.exception}"
print(f"✅ Study mode rendered — {len(at.tabs)} tabs, no exceptions")

# Active mode (flip the top radio and re-run)
at.radio[0].set_value("💰 Active Investor Mode").run()
assert not at.exception, f"ACTIVE MODE exception: {at.exception}"
print("✅ Active mode rendered — no exceptions")

os.remove(_tmp.name)
print("🎉 App smoke test passed.")
