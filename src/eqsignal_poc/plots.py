from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

def price_with_ma(meta: pd.DataFrame, symbol: str, ma_fast=10, ma_slow=50) -> str:
    fig, ax = plt.subplots()
    meta['close'].plot(ax=ax, label="Close")
    meta['close'].rolling(ma_fast).mean().plot(ax=ax, label=f"MA{ma_fast}")
    meta['close'].rolling(ma_slow).mean().plot(ax=ax, label=f"MA{ma_slow}")
    ax.set_title(f"{symbol} — Price with Moving Averages")
    ax.legend()
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
