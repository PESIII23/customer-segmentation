"""
Sales Customer Segmentation — EDA, Cleansing & Segmentation pipeline

Orchestrates Milestones 1-2 of the customer-segmentation project:
    1. Ingest raw transaction data
    2. Clean (drop cancellations, service codes, non-positive lines, duplicates)
    3. Transform transactions into a customer-level RFM table
    4. EDA (statistical tables, profiling tables, distributions, correlation)
    5. Export clean master + RFM table to src/data/processed/
    6. Segment customers (four clustering iterations -> Gold/Silver/Bronze)

Usage:
    CLI:    python3 -m src.pipeline
    Python: from src.pipeline import run_pipeline; run_pipeline()
"""
import json
from pathlib import Path

import pandas as pd

from src.modeling import clustering
from src.preprocessing import data_preparation, data_transformation
from src.viz import eda, eda_addendum

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Paths:
    RAW_FILE = PROJECT_ROOT / "src" / "data" / "raw" / "Project_2_Sales_Data.xlsx"
    CLEAN_PARQUET = PROJECT_ROOT / "src" / "data" / "processed" / "transactions_clean.parquet"
    CANCELLATIONS_PARQUET = PROJECT_ROOT / "src" / "data" / "processed" / "cancellations.parquet"
    RFM_PARQUET = PROJECT_ROOT / "src" / "data" / "processed" / "rfm_customer.parquet"
    RFM_CSV = PROJECT_ROOT / "src" / "data" / "processed" / "rfm_customer.csv"
    SEGMENTS_PARQUET = PROJECT_ROOT / "src" / "data" / "processed" / "customer_segments.parquet"
    SEGMENTS_CSV = PROJECT_ROOT / "src" / "data" / "processed" / "customer_segments.csv"
    STATS_JSON = PROJECT_ROOT / "docs" / "tables" / "eda_stats.json"


def run_pipeline(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the full pipeline. Returns (clean_txns, rfm, segments)."""
    log = print if verbose else (lambda *a, **k: None)
    stats = {}

    log("=" * 60)
    log("SALES CUSTOMER SEGMENTATION — CLEANSING, EDA & SEGMENTATION")
    log("=" * 60)

    # STAGE 1: INGEST -------------------------------------------------------
    log("\n[1/6] LOADING DATA FROM SOURCE FILE...")
    raw = pd.read_excel(Paths.RAW_FILE, header=0)
    log(f"      Loaded {len(raw):,} rows, {raw.shape[1]} columns.")
    stats["raw_rows"] = int(len(raw))
    stats["raw_cols"] = list(raw.columns)
    # original numeric summary (before cleaning) for the deliverable
    stats["raw_numeric_describe"] = (
        raw[["Quantity", "UnitPrice"]].describe().round(2).to_dict()
    )

    # STAGE 2: CLEAN --------------------------------------------------------
    log("\n[2/6] CLEANING DATA...")
    clean_df, cancellations, report = data_preparation.clean_data(raw)
    stats["cleaning_report"] = report
    for k, v in report.items():
        log(f"      {k}: {v:,}")

    # STAGE 3: TRANSFORM (RFM) ---------------------------------------------
    log("\n[3/6] BUILDING CUSTOMER RFM TABLE...")
    rfm, snapshot, dropped_customers = data_transformation.transform_data(clean_df, cancellations)
    stats["snapshot_date"] = str(snapshot.date())
    stats["n_customers"] = int(len(rfm))
    stats["dropped_net_nonpositive_customers"] = dropped_customers
    log(f"      Snapshot date: {snapshot.date()} | Customers: {len(rfm):,} "
        f"({dropped_customers} dropped as net non-positive)")

    # STAGE 4: EDA ----------------------------------------------------------
    log("\n[4/6] PERFORMING EDA...")
    e = eda.EDA()

    txn_num = e.numeric_summary(clean_df, ["QUANTITY", "UNIT_PRICE", "TOTAL_PRICE"],
                                "numeric_summary_transactions")
    rfm_num = e.numeric_summary(rfm, ["RECENCY", "FREQUENCY", "MONETARY", "AVG_ORDER_VALUE"],
                                "numeric_summary_rfm")
    cat_prof = e.categorical_profile(
        clean_df, ["INVOICE_ID", "STOCK_CODE", "DESCRIPTION", "CUSTOMER_ID", "COUNTRY"],
        "categorical_profile")
    stats["numeric_summary_transactions"] = txn_num.to_dict("index")
    stats["numeric_summary_rfm"] = rfm_num.to_dict("index")
    stats["categorical_profile"] = cat_prof.to_dict("records")

    # distributions
    e.plot_histogram(clean_df, ["QUANTITY", "UNIT_PRICE", "TOTAL_PRICE"], "eda_numerical")
    e.plot_histogram(rfm, ["RECENCY", "FREQUENCY", "MONETARY"], "eda_numerical")
    e.plot_histogram(rfm, ["MONETARY_log", "FREQUENCY_log"], "eda_numerical")
    e.plot_box(rfm, ["RECENCY", "FREQUENCY", "MONETARY"], "eda_numerical")

    # categorical bars
    e.plot_top_bar(clean_df["COUNTRY"].value_counts().head(10),
                   "Top 10 Countries by Line Items", "top_countries",
                   "eda_categorical", xlabel="line items")
    top_products = clean_df["DESCRIPTION"].value_counts().head(15)
    e.plot_top_bar(top_products, "Top 15 Products by Line Items",
                   "top_products", "eda_categorical", xlabel="line items")
    monthly = e.plot_timeline(clean_df, "eda_categorical")
    stats["monthly_revenue"] = {d.strftime("%Y-%m"): round(v, 2)
                                for d, v in monthly["revenue"].items()}
    stats["top_countries"] = clean_df["COUNTRY"].value_counts().head(10).to_dict()
    stats["uk_share_pct"] = round(
        100 * (clean_df["COUNTRY"] == "United Kingdom").mean(), 1)

    # correlation
    corr = e.plot_correlation_matrix(
        rfm, ["RECENCY", "FREQUENCY", "MONETARY", "AVG_ORDER_VALUE"], "eda_correlation")
    stats["rfm_correlation"] = corr.to_dict("index")

    # Addendum analyses raised in Milestone 1 review: temporal patterns,
    # cancellation handling, and product/description consistency.
    stats.update(eda_addendum.run_addendum(clean_df, cancellations, log=log))
    log("      Tables -> docs/tables/ | Figures -> docs/eda_*/")

    # STAGE 5: EXPORT -------------------------------------------------------
    log("\n[5/6] EXPORTING CLEAN MASTER + RFM TABLE...")
    Paths.CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    # coerce object cols to string so parquet doesn't mis-infer numeric-looking codes
    clean_out = clean_df.astype({c: "string" for c in clean_df.select_dtypes("object").columns})
    clean_out.to_parquet(Paths.CLEAN_PARQUET, index=False)
    rfm.to_parquet(Paths.RFM_PARQUET, index=False)
    rfm.to_csv(Paths.RFM_CSV, index=False)
    # Cancellations are exported alongside so the netting is auditable.
    canc_out = cancellations.astype(
        {c: "string" for c in cancellations.select_dtypes("object").columns})
    canc_out.to_parquet(Paths.CANCELLATIONS_PARQUET, index=False)
    log(f"      Clean transactions: {clean_df.shape} -> {Paths.CLEAN_PARQUET.name}")
    log(f"      Cancellations:      {cancellations.shape} -> {Paths.CANCELLATIONS_PARQUET.name}")
    log(f"      RFM customers:      {rfm.shape} -> {Paths.RFM_PARQUET.name}")

    # STAGE 6: SEGMENTATION -------------------------------------------------
    log("\n[6/6] SEGMENTING CUSTOMERS (this stage takes a few minutes)...")
    segments, model_results, selection = clustering.run_segmentation(rfm, clean_df, log=log)
    profile = clustering.cluster_profile(segments)

    tables = Paths.STATS_JSON.parent
    model_results.to_csv(tables / "model_iterations.csv", index=False)
    profile.to_csv(tables / "segment_profile.csv")

    clustering.plot_elbow(model_results[model_results["algorithm"] == "K-Means"])
    clustering.plot_algorithm_comparison(model_results)
    clustering.plot_segments(segments)

    segments.to_parquet(Paths.SEGMENTS_PARQUET, index=False)
    segments.to_csv(Paths.SEGMENTS_CSV, index=False)

    stats["model_selection"] = selection
    stats["model_iterations"] = model_results.to_dict("records")
    stats["segment_profile"] = profile.to_dict("index")
    log(f"      Segmented customers: {segments.shape} -> {Paths.SEGMENTS_PARQUET.name}")

    Paths.STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(Paths.STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    log("\n" + "=" * 60)
    log("PIPELINE COMPLETE")
    log("=" * 60 + "\n")

    return clean_df, rfm, segments


if __name__ == "__main__":
    run_pipeline(verbose=True)
