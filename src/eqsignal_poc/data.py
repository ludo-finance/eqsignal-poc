from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def load_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices and align to business days.
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")
    df = df[['Close']].rename(columns={'Close': 'close'})
    # Align to business days; ffill to avoid gaps (holidays/weekends)
    df = df.asfreq('B').ffill()
    df.index.name = 'date'
    # cache raw
    out = RAW_DIR / f"{symbol.replace('^','_')}.csv"
    df.to_csv(out)
    return df
