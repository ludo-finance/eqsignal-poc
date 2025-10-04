from __future__ import annotations
import pandas as pd
import numpy as np

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(n, min_periods=n).mean()
    ma_down = down.rolling(n, min_periods=n).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def make_feature_table(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create a tidy table with features and targets.
    Target (classification): 5-day forward return > 0.
    """
    df = prices.copy()
    # Base returns
    df['ret_1d']  = df['close'].pct_change()
    df['ret_5d']  = df['close'].pct_change(5)

    # Momentum & MAs
    df['mom_5']   = df['close'].pct_change(5)
    df['mom_20']  = df['close'].pct_change(20)
    df['ma_10']   = df['close'].rolling(10).mean()
    df['ma_50']   = df['close'].rolling(50).mean()
    df['ma_cross'] = (df['ma_10'] > df['ma_50']).astype(int)

    # Volatility proxy
    df['vol_10']  = df['ret_1d'].rolling(10).std()

    # RSI
    df['rsi_14']  = _rsi(df['close'], 14)

    # Forward-looking 5d return (shift negative to avoid look-ahead in features)
    df['fwd_5d_ret'] = df['close'].pct_change(5).shift(-5)
    df['y_cls'] = (df['fwd_5d_ret'] > 0).astype(int)

    # Drop rows with any NaNs (warm-up windows)
    df = df.dropna()

    # Final feature set
    feature_cols = ['mom_5','mom_20','vol_10','ma_10','ma_50','ma_cross','rsi_14']
    X = df[feature_cols].copy()
    y = df['y_cls'].copy()
    meta = df[['close','ret_1d','fwd_5d_ret']].copy()  # handy for plots and PnL-ish stats

    return X, y, meta, feature_cols
