from __future__ import annotations
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.units import inch
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)


def _table(data, colWidths=None):
    """Helper to create nicely formatted tables."""
    t = Table(data, colWidths=colWidths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
    ]))
    return t


def build_pdf_report(symbol: str, features: list[str], metrics_path: Path, figures: list[Path]) -> str:
    """Generate a professional, explanatory PDF summary with tables and figures."""
    pdf_path = REPORTS / f"{symbol}_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, title=f"{symbol} Report")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionTitle", fontSize=13, fontName="Helvetica-Bold",
                              textColor=colors.HexColor("#1f3b73"), spaceAfter=6))
    styles.add(ParagraphStyle(name="SubHeading", fontSize=11, fontName="Helvetica-Bold",
                              textColor=colors.HexColor("#2d5aa6"), spaceAfter=4))
    styles.add(ParagraphStyle(name="BodyTextSmall", fontSize=9, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="Footer", fontSize=8, textColor=colors.grey, alignment=1))

    story = []

    # === Header Banner ===
    gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    banner = Table(
        [[f"Equity Signal Report — {symbol}", f"Generated on {gen_date}"]],
        colWidths=[3.5 * inch, 2.5 * inch]
    )
    banner.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#e6e6e6")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(banner)
    story.append(Spacer(1, 0.2 * inch))

    # === Feature List ===
    story.append(Paragraph("Features Used", styles["SectionTitle"]))
    story.append(Paragraph(", ".join(features) if features else "N/A", styles["BodyTextSmall"]))
    story.append(Spacer(1, 0.15 * inch))

    # === Market Overview Table ===
    try:
        metrics = json.loads(metrics_path.read_text())
        pnl = metrics.get("pnl_sketch", {})
        cum_ret = pnl.get("strat_cum_return", 0)
        hit_rate = pnl.get("hit_rate_when_long", 0)
        corr = pnl.get("spearman_ic_signal_vs_fwd", 0)

        market_table = _table([
            ["Metric", "Value"],
            ["Cumulative Strategy Return", f"{cum_ret:.2%}"],
            ["Hit Rate (when Long)", f"{hit_rate:.2%}"],
            ["IC (Signal vs Fwd Return)", f"{corr:.2f}"],
        ], colWidths=[3 * inch, 3 * inch])
        story.append(Paragraph("Market Overview", styles["SectionTitle"]))
        story.append(market_table)
    except Exception:
        story.append(Paragraph("Market data summary unavailable.", styles["BodyTextSmall"]))
    story.append(Spacer(1, 0.2 * inch))

    # === Commentary on Price & Momentum ===
    story.append(Paragraph("Price and Momentum Analysis", styles["SectionTitle"]))
    story.append(Paragraph(
        f"The first two figures show the price trend of {symbol} with its short-term and long-term "
        f"moving averages, followed by momentum changes. Positive momentum periods (green) suggest "
        f"strong upward short-term pressure, while red bars indicate cooling phases or potential pullbacks. "
        f"This overview helps spot acceleration or exhaustion patterns before model-driven signals are considered.",
        styles["BodyTextSmall"]
    ))
    story.append(Spacer(1, 0.1 * inch))

    # === Figures: Price & Momentum ===
    for fig in figures[:2]:
        if Path(fig).exists():
            story.append(Image(str(fig), width=5.5 * inch, height=2.5 * inch))
            story.append(Spacer(1, 0.2 * inch))

    # === Model Accuracy Section ===
    story.append(Paragraph("Model Accuracy and Confidence", styles["SectionTitle"]))
    try:
        classification = metrics.get("classification", {})
        acc = classification.get("accuracy", 0)
        prec = classification.get("precision_pos", 0)
        rec = classification.get("recall_pos", 0)
        auc = classification.get("roc_auc", 0)

        model_table = _table([
            ["Metric", "Value"],
            ["Overall Accuracy", f"{acc:.2%}"],
            ["Precision (Positive)", f"{prec:.2%}"],
            ["Recall (Positive)", f"{rec:.2%}"],
            ["ROC AUC", f"{auc:.2f}"]
        ], colWidths=[3 * inch, 3 * inch])
        story.append(model_table)

        # Insight paragraph
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(
            f"The model currently achieves an overall accuracy of about {acc:.0%}, "
            f"with a {prec:.0%} precision rate when predicting upward movements. "
            f"This suggests moderate predictive strength, with room to refine features or thresholds "
            f"to improve discrimination power (ROC AUC = {auc:.2f}).",
            styles["BodyTextSmall"]
        ))
    except Exception:
        story.append(Paragraph("Model performance data unavailable.", styles["BodyTextSmall"]))
    story.append(Spacer(1, 0.2 * inch))

    # === Probability Figure ===
    if len(figures) > 2 and Path(figures[2]).exists():
        story.append(Image(str(figures[2]), width=5.5 * inch, height=2.5 * inch))
        story.append(Spacer(1, 0.2 * inch))

    # === Footer ===
    story.append(Spacer(1, 0.2 * inch))
    footer_line = Table([[f"Generated automatically by eqsignal-poc — {gen_date}"]],
                        colWidths=[6 * inch])
    footer_line.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 0.25, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
    ]))
    story.append(footer_line)

    doc.build(story)
    return str(pdf_path)
