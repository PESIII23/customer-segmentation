"""
Milestone 1 addendum analyses, added in response to instructor feedback.

Three questions, three sections:
    1. Transaction volume and revenue by day of week and hour of day
    2. How negative-quantity (cancellation) records are handled, and why
    3. Product/description analysis for the top stock codes

Figures are written under docs/eda_temporal/, docs/eda_cancellations/ and
docs/eda_products/; tables under docs/tables/.
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS = PROJECT_ROOT / "docs"

# Two-hue categorical pair, CVD-validated (worst adjacent ΔE 20.0 protan).
PRIMARY = "#4C72B0"
ACCENT = "#DD8452"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

_logger = logging.getLogger(__name__)


def _subdir(name: str) -> Path:
    path = DOCS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tables() -> Path:
    return _subdir("tables")


# ---- 1. temporal patterns ---------------------------------------------------
def temporal_profile(purchases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transactions and revenue by day of week and hour of day.

    Transactions are counted as distinct invoices, not line items, so a single
    large basket does not read as many purchases.
    """
    df = purchases.copy()
    df["INVOICE_DATE"] = pd.to_datetime(df["INVOICE_DATE"])
    df["DOW"] = df["INVOICE_DATE"].dt.dayofweek
    df["HOUR"] = df["INVOICE_DATE"].dt.hour

    dow = df.groupby("DOW").agg(TRANSACTIONS=("INVOICE_ID", "nunique"),
                                LINE_ITEMS=("INVOICE_ID", "size"),
                                REVENUE=("TOTAL_PRICE", "sum"))
    # Reindex across all 7 days so an absent trading day shows as a zero bar.
    dow = dow.reindex(range(7), fill_value=0)
    dow.index = DAYS
    dow["REVENUE_PER_TXN"] = (dow["REVENUE"] / dow["TRANSACTIONS"]).fillna(0).round(2)
    dow["REVENUE_PCT"] = (100 * dow["REVENUE"] / dow["REVENUE"].sum()).round(1)

    hour = df.groupby("HOUR").agg(TRANSACTIONS=("INVOICE_ID", "nunique"),
                                  REVENUE=("TOTAL_PRICE", "sum"))
    hour["REVENUE_PER_TXN"] = (hour["REVENUE"] / hour["TRANSACTIONS"]).round(2)
    hour["REVENUE_PCT"] = (100 * hour["REVENUE"] / hour["REVENUE"].sum()).round(1)

    dow.round(2).to_csv(_tables() / "temporal_day_of_week.csv")
    hour.round(2).to_csv(_tables() / "temporal_hour_of_day.csv")

    _plot_dow(dow)
    _plot_hour(hour)
    return dow, hour


def _plot_dow(dow: pd.DataFrame):
    path = _subdir("eda_temporal")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, col, title, ylab in (
            (ax1, "TRANSACTIONS", "Transactions by Day of Week", "distinct invoices"),
            (ax2, "REVENUE", "Revenue by Day of Week", "revenue (GBP)")):
        ax.bar(dow.index, dow[col], color=PRIMARY)
        ax.bar_label(ax.containers[0], fmt="%.0f", padding=3, fontsize=9)
        ax.set(title=title, ylabel=ylab)
        ax.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(path / "day_of_week.png", dpi=110)
    plt.close()


def _plot_hour(hour: pd.DataFrame):
    path = _subdir("eda_temporal")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.bar(hour.index, hour["TRANSACTIONS"], color=PRIMARY)
    ax1.set(title="Transactions by Hour of Day", xlabel="hour", ylabel="distinct invoices")
    ax2.bar(hour.index, hour["REVENUE"], color=PRIMARY)
    ax2.set(title="Revenue by Hour of Day", xlabel="hour", ylabel="revenue (GBP)")
    for ax in (ax1, ax2):
        ax.set_xticks(list(hour.index))
        ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.savefig(path / "hour_of_day.png", dpi=110)
    plt.close()


# ---- 2. cancellation handling -----------------------------------------------
def cancellation_profile(purchases: pd.DataFrame,
                         cancellations: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Quantify the cancellation population and the cost of ignoring it."""
    gross = purchases["TOTAL_PRICE"].sum()
    refund = cancellations["TOTAL_PRICE"].abs().sum()

    cust_gross = purchases.groupby("CUSTOMER_ID")["TOTAL_PRICE"].sum()
    cust_ref = cancellations.groupby("CUSTOMER_ID")["TOTAL_PRICE"].sum().abs()
    ratio = (cust_ref / cust_gross).dropna()

    summary = {
        "cancellation_lines": int(len(cancellations)),
        "purchase_lines": int(len(purchases)),
        "refund_value": round(float(refund), 2),
        "gross_revenue": round(float(gross), 2),
        "net_revenue": round(float(gross - refund), 2),
        "overstatement_pct": round(100 * refund / (gross - refund), 2),
        "customers_with_cancellation": int(cust_ref.index.isin(cust_gross.index).sum()),
        "customers_total": int(len(cust_gross)),
        "median_refund_ratio_pct": round(float(ratio.median() * 100), 1),
        "customers_refund_over_50pct": int((ratio > 0.5).sum()),
        "customers_net_nonpositive": int((ratio >= 1).sum()),
    }
    pd.Series(summary).to_csv(_tables() / "cancellation_summary.csv", header=["value"])

    monthly = (cancellations.assign(M=pd.to_datetime(cancellations["INVOICE_DATE"]).dt.to_period("M"))
               .groupby("M")["TOTAL_PRICE"].sum().abs())
    monthly.index = monthly.index.astype(str)

    _plot_cancellations(summary, ratio, monthly)
    return summary, ratio.to_frame("REFUND_RATIO")


def _plot_cancellations(summary: dict, ratio: pd.Series, monthly: pd.Series):
    path = _subdir("eda_cancellations")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    bars = ax1.bar(["Gross\n(cancellations ignored)", "Net\n(cancellations applied)"],
                   [summary["gross_revenue"], summary["net_revenue"]],
                   color=[ACCENT, PRIMARY])
    ax1.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax1.set(title="Customer Revenue: Gross vs Net of Cancellations", ylabel="revenue (GBP)")
    ax1.margins(y=0.15)

    ax2.hist(ratio[ratio <= 1] * 100, bins=40, color=PRIMARY, edgecolor="black")
    ax2.set(title="Refunds as % of Gross Spend (customers with cancellations)",
            xlabel="refunds as % of that customer's gross spend", ylabel="customers")
    plt.tight_layout()
    plt.savefig(path / "cancellation_impact.png", dpi=110)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(monthly.index, monthly.values, color=PRIMARY)
    ax.set(title="Cancellation Value by Month", ylabel="refunded value (GBP)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    plt.tight_layout()
    plt.savefig(path / "cancellations_by_month.png", dpi=110)
    plt.close()


# ---- 3. product / description analysis --------------------------------------
def product_profile(purchases: pd.DataFrame, top_n: int = 10
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Top stock codes by revenue, and description consistency per stock code."""
    top = (purchases.groupby("STOCK_CODE")
           .agg(DESCRIPTION=("DESCRIPTION", lambda s: s.mode().iloc[0]),
                LINE_ITEMS=("INVOICE_ID", "size"),
                UNITS=("QUANTITY", "sum"),
                REVENUE=("TOTAL_PRICE", "sum"),
                CUSTOMERS=("CUSTOMER_ID", "nunique"),
                AVG_UNIT_PRICE=("UNIT_PRICE", "mean"))
           .sort_values("REVENUE", ascending=False)
           .head(top_n).round(2))
    top["REVENUE_PCT"] = (100 * top["REVENUE"] / purchases["TOTAL_PRICE"].sum()).round(2)

    # Description is free text, so the same code can carry spelling variants.
    per_code = purchases.groupby("STOCK_CODE")["DESCRIPTION"].nunique()
    multi = per_code[per_code > 1].sort_values(ascending=False)
    rows = []
    for code in multi.head(10).index:
        vc = purchases.loc[purchases["STOCK_CODE"] == code, "DESCRIPTION"].value_counts()
        rows.append({"STOCK_CODE": code, "VARIANTS": int(len(vc)),
                     "DESCRIPTIONS": " | ".join(f"{d} (x{c})" for d, c in vc.items())})
    variants = pd.DataFrame(rows)

    stats = {
        "stock_codes": int(len(per_code)),
        "codes_with_multiple_descriptions": int((per_code > 1).sum()),
        "codes_with_multiple_pct": round(100 * float((per_code > 1).mean()), 1),
        "max_descriptions_on_one_code": int(per_code.max()),
        "descriptions_on_multiple_codes": int(
            (purchases.groupby("DESCRIPTION")["STOCK_CODE"].nunique() > 1).sum()),
        "top_n_revenue_pct": round(float(top["REVENUE_PCT"].sum()), 1),
    }

    top.to_csv(_tables() / "top_stock_codes.csv")
    variants.to_csv(_tables() / "description_variants.csv", index=False)
    pd.Series(stats).to_csv(_tables() / "description_consistency.csv", header=["value"])

    _plot_products(top)
    return top, variants, stats


def _plot_products(top: pd.DataFrame):
    path = _subdir("eda_products")
    labels = [f"{c} — {d[:34]}" for c, d in zip(top.index, top["DESCRIPTION"])]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, top["REVENUE"], color=PRIMARY)
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.invert_yaxis()
    ax.set(title=f"Top {len(top)} Stock Codes by Revenue", xlabel="revenue (GBP)")
    ax.margins(x=0.12)
    plt.tight_layout()
    plt.savefig(path / "top_stock_codes.png", dpi=110)
    plt.close()


# ---- orchestration ----------------------------------------------------------
def run_addendum(purchases: pd.DataFrame, cancellations: pd.DataFrame, log=print) -> dict:
    """Run all three addendum analyses; returns a stats dict for eda_stats.json."""
    dow, hour = temporal_profile(purchases)
    cancel_summary, _ = cancellation_profile(purchases, cancellations)
    top, variants, prod_stats = product_profile(purchases)

    missing = [d for d in DAYS if dow.loc[d, "TRANSACTIONS"] == 0]
    log(f"      Temporal: peak day {dow['REVENUE'].idxmax()}, "
        f"peak hour {int(hour['TRANSACTIONS'].idxmax())}:00"
        + (f", no trading on {', '.join(missing)}" if missing else ""))
    log(f"      Cancellations: {cancel_summary['cancellation_lines']:,} lines, "
        f"GBP {cancel_summary['refund_value']:,.0f} netted "
        f"({cancel_summary['overstatement_pct']}% of net revenue)")
    log(f"      Products: top 10 codes = {prod_stats['top_n_revenue_pct']}% of revenue, "
        f"{prod_stats['codes_with_multiple_descriptions']} codes have >1 description")

    return {
        "temporal_day_of_week": dow.round(2).to_dict("index"),
        "temporal_hour_of_day": hour.round(2).to_dict("index"),
        "temporal_days_absent": missing,
        "cancellation_summary": cancel_summary,
        "top_stock_codes": top.reset_index().to_dict("records"),
        "description_consistency": prod_stats,
        "description_variants": variants.to_dict("records"),
    }
