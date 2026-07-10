# Sales Customer Segmentation

An end-to-end analytical pipeline that turns raw e-commerce transaction data into
a clean, customer-level **RFM** (Recency, Frequency, Monetary) table for customer
segmentation and targeted campaign recommendations.

> **Project 2 — Milestone 1:** Exploratory Data Analysis & Data Cleansing Approach.
> The graded deliverable is
> `SMITH P2 - Exploratory Data Analysis and Data Cleansing Approach.docx` in the repo root.

---

## Project Structure

```text
├── README.md
├── requirements.txt
├── SMITH P2 - Exploratory Data Analysis and Data Cleansing Approach.docx   # Milestone 1 deliverable
├── docs/
│   ├── tables/                # stat/profile tables (CSV) + eda_stats.json
│   ├── eda_numerical/         # histograms & box plots (Quantity, Price, RFM)
│   ├── eda_categorical/       # country, product, and monthly-revenue bars
│   └── eda_correlation/       # RFM Pearson correlation heatmap
└── src/
    ├── pipeline.py            # entry point: run_pipeline()
    ├── preprocessing/
    │   ├── data_preparation.py    # cleansing (cancellations, duplicates, service codes)
    │   └── data_transformation.py # transaction → customer RFM aggregation
    ├── viz/
    │   └── eda.py             # tables, distributions, bars, correlation matrix
    └── data/
        ├── raw/               # Project_2_Sales_Data.xlsx (immutable source)
        └── processed/         # transactions_clean.parquet, rfm_customer.{parquet,csv}
```

---

## Quick Start

```bash
# 1. Environment
conda create --name sales-seg python=3.13 -y && conda activate sales-seg

# 2. Dependencies
pip install -r requirements.txt

# 3. Run the pipeline (Project_2_Sales_Data.xlsx must be in src/data/raw/)
python3 -m src.pipeline
```

Running the pipeline regenerates every table and figure under `docs/`, writes the
clean master file and RFM table to `src/data/processed/`, and dumps a machine-readable
summary of all statistics to `docs/tables/eda_stats.json`.

---

## Pipeline Stages

| Stage | Description |
| ----- | ----------- |
| **1. Ingest** | Load the raw transaction workbook into a DataFrame |
| **2. Clean** | Drop duplicates, cancelled orders (`C` invoices), non-product service codes, and non-positive quantity/price lines |
| **3. Transform (RFM)** | Derive line revenue and aggregate transactions into a customer-level Recency / Frequency / Monetary table with log transforms |
| **4. EDA** | Statistical summaries, categorical profiling, distributions, and a Pearson correlation matrix |
| **5. Export** | Save the clean master file and RFM table for downstream segmentation |

---

## Data & Approach

- **Source:** `Project_2_Sales_Data.xlsx` — 379,979 line items, 8 columns, full-year 2021.
- **Cleansing:** 14,398 records removed (3.8%); **365,581** clean transaction lines retained.
- **Customers:** **4,214** unique customers profiled on R/F/M.
- **Next milestones:** scale the log-transformed RFM features and apply K-Means /
  hierarchical clustering to define segments, then map each segment to a campaign.

---

## Tools & Libraries

- Python 3.13+
- pandas, numpy — data manipulation and aggregation
- matplotlib, seaborn — visualization
- openpyxl — Excel ingestion; pyarrow — parquet caching

---

## Support

- Maintainer: [Phillip Smith](https://www.linkedin.com/in/pesiii/)
