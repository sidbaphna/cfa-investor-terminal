"""importers.py — parse uploaded files into watchlist tickers.

Streamlit-free and pandas-only, so it's unit-testable on its own. Accepts a
file path, a file-like buffer, or a Streamlit UploadedFile.
"""
from __future__ import annotations

from typing import Any, List

import pandas as pd


def tickers_from_csv(file: Any) -> List[str]:
    """Return a clean, de-duplicated, upper-cased ticker list from a CSV.

    Prefers a 'Symbol' or 'Ticker' column (case-insensitive); otherwise falls
    back to the first column. Skips blanks and NaNs, preserves first-seen order.
    """
    # Rewind file-like buffers (a re-uploaded Streamlit file may be at EOF).
    try:
        file.seek(0)
    except (AttributeError, OSError, ValueError):
        pass

    df = pd.read_csv(file)
    if df.empty:
        return []

    col = None
    for cand in ("symbol", "ticker"):
        for c in df.columns:
            if str(c).strip().lower() == cand:
                col = c
                break
        if col is not None:
            break
    if col is None:
        col = df.columns[0]

    out: List[str] = []
    for value in df[col].tolist():
        sym = str(value).strip().upper()
        if sym and sym != "NAN" and sym not in out:
            out.append(sym)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        syms = tickers_from_csv(sys.argv[1])
        print(f"{len(syms)} tickers:", ", ".join(syms))
