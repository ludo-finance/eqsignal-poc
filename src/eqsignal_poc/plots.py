from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure reports directory exists
REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helper: prepare data safely for plotting
# ============================================================
def _prepare_close(meta) -> pd.DataFrame:
    """
    Ensure 'meta' is a DataFrame with a numeric 'close' column.
    Handles DataFrame, Series, ndarray, or list input types safely.
    """
    # Wrap input into DataFrame
    if isinstance(meta, (pd.Series, np.ndarray, list)):
        df = pd.DataFrame(meta).copy()
        if df.shape[1] > 1:
            df = df.iloc[:, [0]]  # keep first column
        df.columns = ["close"]
    elif isinstance(meta, pd.DataFrame):
        df = meta.copy()
    else:
        raise TypeError(f"Unsupported type for meta: {type(meta)}")

    # Find appropriate column
    possible_cols = [c for c in df.columns if str(c).lower() == "close"]
    if possible_cols:
        col = possible_cols[0]
    else:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            # Flatten all values if nothing numeric
            flat = pd.Series(np.array(df).flatten(), name="close")
            return pd.DataFrame(flat)
        col = num_cols[0]

    # Convert safely to numeric
    series_like = df[col]
    if not isinstance(series_like, (pd.Series, np.ndarray, list)):
        series_like = np.ravel(series_like)

    df["close"] = pd.to_numeric(series_like, errors="coerce")
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    return df


# ============================================================
# 1. Price with Moving Averages
# ============================================================
def price_with_ma(meta, symbol: str, ma_fast=10, ma_slow=50) -> str:
    """
    Plot price with moving averages.
    Handles any input type safely (DataFrame, Series, ndarray, list).
    """
    df = _prepare_close(meta)
    close = df["close"]

    fig, ax = plt.subplots(figsize=(10, 5))  # rectangular layout

    # Navy palette
    colors = {
        "close": "#1f3b73",
        "ma_fast": "#2d5aa6",
        "ma_slow": "#6c8ebf",
    }

    # Plot price + moving averages
    ax.plot(df.index, close, label=f"{symbol} Close Price", color=colors["close"], linewidth=1.2)
    ax.plot(df.index, close.rolling(ma_fast).mean(), label=f"{ma_fast}-day MA", color=colors["ma_fast"], linewidth=1.2)
    ax.plot(df.index, close.rolling(ma_slow).mean(), label=f"{ma_slow}-day MA", color=colors["ma_slow"], linewidth=1.2)

    # Style
    ax.set_title(f"{symbol} — Price with Moving Averages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(axis="both", labelsize=8)

    # Save output
    out = REPORTS / f"{symbol}_price_ma.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(out)


# ============================================================
# 2. Out-of-Sample Probability Timeline
# ============================================================
def oos_prob_timeline(meta, oof_proba, y_true=None) -> str:
    """
    Plot model predicted probabilities vs actual outcomes,
    with an accuracy strip and summary stats.
    """
    df = _prepare_close(meta)
    proba = pd.Series(oof_proba, index=df.index)

    fig, (ax_main, ax_acc) = plt.subplots(
        2, 1, figsize=(10, 5),
        gridspec_kw={'height_ratios': [4, 0.3]},
        sharex=True
    )

    # --- Events for visual reference (background) ---
    if y_true is not None:
        events = pd.Series(y_true, index=df.index).fillna(0).astype(int)
        pos_dates = events[events == 1].index
        neg_dates = events[events == 0].index
        for x in pos_dates:
            ax_main.axvline(x=x, color="#90EE90", alpha=0.15, linewidth=0.5, zorder=1)
        for x in neg_dates:
            ax_main.axvline(x=x, color="#FF7F7F", alpha=0.15, linewidth=0.5, zorder=1)

    # --- Main probability line ---
    ax_main.plot(df.index, proba, color="#1f3b73", linewidth=1.0, alpha=0.9, label="Model Probability")

    # --- Rolling mean for clarity ---
    rolling_mean = proba.rolling(20).mean()
    ax_main.plot(df.index, rolling_mean, color="#2d5aa6", linewidth=1.5, label="Monthly Avg. Confidence (20d)")

    # --- Titles & labels ---
    ax_main.set_title("Model Probability vs Market Outcomes", fontsize=14, fontweight="bold")
    ax_main.set_ylabel("Predicted Probability", fontsize=9)

    # --- Formatting ---
    ax_main.set_ylim(0, 1)
    ax_main.grid(True, linestyle="--", alpha=0.4)
    ax_main.tick_params(axis="both", labelsize=8)
    ax_main.legend(fontsize=8, loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # --- Accuracy calculation ---
    acc_strip = pd.Series(index=df.index, dtype=float)
    if y_true is not None:
        preds = (proba > 0.5).astype(int)
        correct = (preds == events).astype(int)
        acc_strip[:] = correct

        overall_acc = correct.mean()
        hit_rate_when_long = correct[(preds == 1)].mean()
        text = f"Overall Accuracy: {overall_acc:.1%}\nHit Rate (when Long): {hit_rate_when_long:.1%}"
        ax_main.text(
            0.02, 0.95, text, transform=ax_main.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3")
        )

    # --- Accuracy strip (subplot) ---
    if y_true is not None:
        colors = acc_strip.map({1: "#3CB371", 0: "#B22222"})  # green=correct, red=wrong
        ax_acc.bar(df.index, [1]*len(df.index), width=1, color=colors, align="edge", linewidth=0)
    else:
        ax_acc.bar(df.index, [1]*len(df.index), width=1, color="#d3d3d3", align="edge", linewidth=0)

    # --- Strip formatting ---
    ax_acc.set_ylim(0, 1)
    ax_acc.set_yticks([])
    ax_acc.set_xlabel("Date", fontsize=9)
    ax_acc.tick_params(axis="x", labelsize=8)
    ax_acc.grid(False)

    # --- Save ---
    out = REPORTS / "oos_proba.png"
    fig.tight_layout(h_pad=0.4)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(out)




# ============================================================
# 3. Standalone Momentum States Plot
# ============================================================
def momentum_states_plot(meta, symbol: str, window: int = 5) -> str:
    """
    Plot momentum states separately from the price chart.
    - Green = positive momentum, Red = negative momentum
    - Works for any input type safely, including nested arrays
    """
    # --- Prepare 'close' values robustly ---
    df = _prepare_close(meta)

    # Flatten and convert all values to numeric
    raw_values = np.array(df["close"]).astype("float").flatten()
    close = pd.Series(raw_values, index=df.index[: len(raw_values)])

    # Compute momentum (% change)
    mom = close.pct_change(window)
    mom = pd.to_numeric(mom, errors="coerce").fillna(0)

    # Colours: green for positive, red for negative
    colors = ["#3CB371" if float(v) > 0 else "#B22222" for v in mom]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(close.index, mom * 100, color=colors, width=1.0, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8)

    # --- Labels ---
    ax.set_title(f"{symbol} — {window}-day Momentum", fontsize=14, fontweight="bold")
    ax.set_ylabel("Momentum (%)", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    # --- Save ---
    out = REPORTS / f"{symbol}_momentum_{window}d.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(out)
