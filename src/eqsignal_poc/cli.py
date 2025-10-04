from __future__ import annotations
import argparse, json
from pathlib import Path
from .data import load_prices
from .features import make_feature_table
from .model import time_series_cv_predictions
from .evaluate import classification_report_oos, simple_signal_pnl, REPORTS
from .plots import price_with_ma, oos_prob_timeline, momentum_states_plot
from .report import build_pdf_report

def main():
    p = argparse.ArgumentParser(prog="eqsignal-poc")
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--start",  default="2015-01-01")
    p.add_argument("--end",    default=None)
    p.add_argument("--splits", type=int, default=5)
    args = p.parse_args()

    # --- Data and features ---
    prices = load_prices(args.symbol, args.start, args.end)
    X, y, meta, feats = make_feature_table(prices)
    res = time_series_cv_predictions(X, y, n_splits=args.splits)

    # --- Metrics ---
    cls = classification_report_oos(y, res["oof_pred"], res["oof_proba"])
    pnl = simple_signal_pnl(meta, res["oof_pred"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(
        json.dumps({"classification": cls, "pnl_sketch": pnl}, indent=2)
    )

    # --- Plots ---
    p1 = price_with_ma(meta, args.symbol)
    p2 = oos_prob_timeline(meta, res["oof_proba"], y)
    p3 = momentum_states_plot(meta, args.symbol)

    # --- PDF Report ---
    pdf = build_pdf_report(
        args.symbol,
        features=feats,
        metrics_path=(REPORTS / "metrics.json"),
        figures=[p1, p2, p3]
    )
    print("PDF Report ->", pdf)

    # --- Summary ---
    print("Features:", feats)
    print("Saved metrics ->", (REPORTS / "metrics.json").as_posix())
    print("Figures ->", p1, " | ", p2, " | ", p3)

if __name__ == "__main__":
    main()
