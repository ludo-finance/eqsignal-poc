from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

def price_with_ma(meta: pd.DataFrame, symbol: str, ma_fast=10, ma_slow=50) -> str:
    """
    Plot price with moving averages.
    - Uses navy shades
    - Clear legend labels
    - Rectangular aspect ratio
    - Smaller fonts for everything except title
    """
    fig, ax = plt.subplots(figsize=(10, 5))  # rectangular

    # Custom navy shades
    colors = {
        "close": "#1f3b73",       # dark navy
        "ma_fast": "#2d5aa6",     # medium navy
        "ma_slow": "#6c8ebf",     # light navy
    }

    # Explicit labels for each line
    ax.plot(meta.index, meta['close'], label=f"{symbol} Close Price", color=colors["close"], linewidth=1.2)
    ax.plot(meta.index, meta['close'].rolling(ma_fast).mean(), label=f"{ma_fast}-day Moving Average", color=colors["ma_fast"], linewidth=1.2)
    ax.plot(meta.index, meta['close'].rolling(ma_slow).mean(), label=f"{ma_slow}-day Moving Average", color=colors["ma_slow"], linewidth=1.2)

    # Title and labels
    ax.set_title(f"{symbol} — Price with Moving Averages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price", fontsize=9)

    # Legend (smaller font)
    ax.legend(fontsize=8)

    # Tick label sizes
    ax.tick_params(axis="both", labelsize=8)

    out = REPORTS / f"{symbol}_price_ma.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(out)

def oos_prob_timeline(meta: pd.DataFrame, oof_proba) -> str:
    fig, ax = plt.subplots()
    pd.Series(oof_proba, index=meta.index).plot(ax=ax)
    ax.set_title("Out-of-sample predicted probability (class=1)")
    out = REPORTS / "oos_proba.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(out)
