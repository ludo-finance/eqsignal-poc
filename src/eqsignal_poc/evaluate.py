from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from scipy.stats import spearmanr

REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

def classification_report_oos(y_true, oof_pred, oof_proba) -> dict:
    mask = ~np.isnan(oof_pred)
    y = y_true.iloc[mask]
    yhat = oof_pred[mask].astype(int)
    phat = oof_proba[mask]

    return {
        "n_oos": int(mask.sum()),
        "accuracy": float(accuracy_score(y, yhat)),
        "precision_pos": float(precision_score(y, yhat, zero_division=0)),
        "recall_pos": float(recall_score(y, yhat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, phat)) if len(np.unique(y)) > 1 else np.nan,
    }

def simple_signal_pnl(meta: pd.DataFrame, oof_pred, holding_days: int = 5) -> dict:
    """
    A very simple 'hold when positive signal' sketch:
    - When predicted class == 1 at day t, assume equal-weight long for the next 'holding_days'.
    - Compute overlapping-average return approximation using fwd_5d_ret.
    This is intentionally rough—kept to PoC level.
    """
    ser_pred = pd.Series(oof_pred, index=meta.index)
    mask = ~ser_pred.isna()
    # Use the 5-day forward return as a coarse payoff proxy
    realized = meta.loc[mask, 'fwd_5d_ret']
    signal = ser_pred.loc[mask].astype(int)
    # Return of the strategy: only take fwd_5d_ret when signal==1, else 0
    strat = (realized * signal).dropna()
    gross = (1 + strat).prod() - 1
    hit_rate = (realized[signal == 1] > 0).mean() if (signal == 1).any() else np.nan
    ic, _ = spearmanr(realized, signal)  # monotonic association (very rough)

    return {"strat_cum_return": float(gross),
            "hit_rate_when_long": float(hit_rate) if hit_rate==hit_rate else np.nan,
            "spearman_ic_signal_vs_fwd": float(ic) if ic==ic else np.nan}
