# Sales Customer Segmentation

An end-to-end analytical pipeline that turns raw e-commerce transaction data into
a clean, customer-level **RFM** (Recency, Frequency, Monetary) table, then segments
every customer into **Gold / Silver / Bronze** tiers with unsupervised clustering.

> **Project 2 — Milestone 2:** Data Wrangling & Model Output.
> The graded deliverable is
> `SMITH P2 - Data Wrangling and Model Output.docx` in the repo root.

---

## Project Structure

```text
├── README.md
├── requirements.txt
├── SMITH P2 - Data Wrangling and Model Output.docx    # Milestone 2 deliverable
├── docs/
│   ├── tables/                # stat/profile tables (CSV) + eda_stats.json
│   ├── eda_numerical/         # histograms & box plots (Quantity, Price, RFM)
│   ├── eda_categorical/       # country, product, and monthly-revenue bars
│   ├── eda_correlation/       # RFM Pearson correlation heatmap
│   ├── modeling/              # elbow, technique comparison, segment figures
│   └── reference_code/        # instructor reference notebooks (local, not tracked)
└── src/
    ├── pipeline.py            # entry point: run_pipeline()
    ├── preprocessing/
    │   ├── data_preparation.py    # cleansing (cancellations, duplicates, service codes)
    │   └── data_transformation.py # transaction → customer RFM aggregation
    ├── modeling/
    │   └── clustering.py      # four clustering iterations → Gold/Silver/Bronze
    ├── viz/
    │   └── eda.py             # tables, distributions, bars, correlation matrix
    └── data/
        ├── raw/               # Project_2_Sales_Data.xlsx (immutable source)
        └── processed/         # transactions_clean.parquet, rfm_customer.{parquet,csv},
                               # customer_segments.{parquet,csv}
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
clean master file, RFM table, and customer segments to `src/data/processed/`, and dumps
a machine-readable summary of all statistics and model results to `docs/tables/eda_stats.json`.
A full run takes roughly 4 minutes; Stage 6 accounts for most of it.

---

## Pipeline Stages

| Stage | Description |
| ----- | ----------- |
| **1. Ingest** | Load the raw transaction workbook into a DataFrame |
| **2. Clean** | Drop duplicates, cancelled orders (`C` invoices), non-product service codes, and non-positive quantity/price lines |
| **3. Transform (RFM)** | Derive line revenue and aggregate transactions into a customer-level Recency / Frequency / Monetary table with log transforms |
| **4. EDA** | Statistical summaries, categorical profiling, distributions, and a Pearson correlation matrix |
| **5. Export** | Save the clean master file and RFM table for downstream segmentation |
| **6. Segment** | Four clustering iterations (K-Means, Mean Shift, DBSCAN, Agglomerative), metric-based model selection, and Gold/Silver/Bronze banding |

---

## Data & Approach

- **Source:** `Project_2_Sales_Data.xlsx` — 379,979 line items, 8 columns, full-year 2021.
- **Cleansing:** 14,398 records removed (3.8%); **365,581** clean transaction lines retained.
- **Customers:** **4,214** unique customers profiled on R/F/M.
- **Model input:** R/F/M percentile ratings (0–5), so all three behaviors share one scale.
- **Selected model:** **K-Means, k=4** — best on all three validity metrics
  (silhouette 0.381, Calinski-Harabasz 4,296.5, Davies-Bouldin 0.970).

### Segments

| Segment | Customers | % of base | % of revenue | Mean recency | Mean frequency |
| ------- | --------: | --------: | -----------: | -----------: | -------------: |
| Gold    | 1,270 | 30.1% | **76.1%** | 17 days  | 9.10 |
| Silver  | 1,783 | 42.3% | 19.5%     | 67 days  | 2.35 |
| Bronze  | 1,161 | 27.6% | 4.5%      | 186 days | 1.09 |

- **Next milestone:** map each segment to a concrete campaign recommendation and build
  the business-facing presentation.

---

## Tools & Libraries

- Python 3.13+
- pandas, numpy — data manipulation and aggregation
- matplotlib, seaborn — visualization
- openpyxl — Excel ingestion; pyarrow — parquet caching

---

## Support

- Maintainer: [Phillip Smith](https://www.linkedin.com/in/pesiii/)
