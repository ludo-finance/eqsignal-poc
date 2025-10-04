from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

def make_clf_pipeline() -> Pipeline:
    """
    Standard scaler + logistic regression (balanced) as a transparent baseline.
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])

def time_series_cv_predictions(
    X, y, n_splits: int = 5
) -> Dict[str, Any]:
    """
    Expanding-window style CV: each split trains on past, tests on next block.
    Returns out-of-sample predictions stacked across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pipe = make_clf_pipeline()

    oof_pred = np.full(shape=y.shape, fill_value=np.nan, dtype=float)
    oof_proba = np.full(shape=y.shape, fill_value=np.nan, dtype=float)
    fold_idx: List[tuple] = []

    for fold, (tr, te) in enumerate(tscv.split(X, y), 1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof_pred[te]  = pipe.predict(X.iloc[te])
        oof_proba[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        fold_idx.append((tr, te))

    return {
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "fold_idx": fold_idx,
    }
