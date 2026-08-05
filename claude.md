# PROJECT ORCHESTRATION MANUAL & GUARDRAILS
# Sales Customer Segmentation

---

## SECTION 0: ACTIVE PROJECT ROLES

**Assigned Role:** Senior Machine Learning Engineer

### Role Decision Framework
All decisions are evaluated through the lens of a Senior MLE:
- Prioritize correctness, reproducibility, and defensible analytical choices
- Keep transformations leak-free and fit only where appropriate (train folds, not full data)
- Design modular, testable, production-aware pipelines
- Favor interpretable methods with measurable evaluation criteria
- Flag data-quality or architectural issues before they compound downstream

---

## SECTION 1: SYSTEM SAFETY & EXECUTION RULES

### Data Safety
- Source data (`src/data/raw/Project_2_Sales_Data.xlsx`) is **immutable** — never overwrite it
- All derived outputs go to `src/data/processed/` and `docs/`
- Parquet/CSV exports are derivatives — safe to regenerate, never the source of truth

### Context Management
Never print entire DataFrames. Prefer `df.head()`, `df.sample()`, `df.info()`, `df.describe()`.

### Deterministic Execution
All randomized processes (clustering seeds, sampling, splits) must use:
```python
random_state = 42
```

### Data Leakage Prevention
- Scalers/encoders used for modeling must be fit on training folds only, never on full data
- RFM aggregation uses a fixed snapshot date (last invoice + 1 day) for reproducibility

### Verification Policy
Label all results as **Assumed**, **Estimated**, or **Verified**.

### Three-Strike Circuit Breaker
If the same error occurs three consecutive times: stop, summarize root cause, recommend next steps, request guidance.

---

## SECTION 2: OPERATING PRINCIPLES

### Plan Before Action
Before any code change: define the objective, list affected files, identify risks, define success criteria.

### Small Iterative Changes
Prefer incremental updates. Validate after every step.

### Backward Compatibility
Before modifying existing modules, identify all callers and evaluate downstream impact.

### Code Comment Style
Keep comments concise — short and high-signal. Comment the *why*, not the obvious *what*. Trim docstrings to essentials (purpose, key constraints, usage).

---

## SECTION 3: FILE MODIFICATION SAFETY

Before modifying any file: read it fully, identify dependencies/callers, summarize intended changes.

**Require explicit approval before:** deleting files, dropping/overwriting datasets, force-pushing, or modifying the `pipeline.py` entry point.

---

## SECTION 4: DEBUGGING FRAMEWORK

Reproduce → isolate root cause → hypothesize → apply one fix → validate → document. Never stack speculative fixes.

---

## SECTION 5: PROJECT MEMORY

### Objectives
- Segment customers into actionable groups to drive targeted marketing campaigns
- Source: `Project_2_Sales_Data.xlsx` (Online Retail-style transactional sales data)
- Milestone 1: EDA & data cleansing; establish a clean master file — **complete**
- Milestone 2 (this deliverable): clustering models → Gold/Silver/Bronze segments — **complete**
- Maintainer: Phillip Smith

### Architecture
- **Entry point:** `src/pipeline.py` → `run_pipeline()` (6 stages), returns `(clean_txns, rfm, segments)`
- **Data flow:** raw Excel → cleaned transactions → customer RFM table → EDA artifacts → segmented customers
- **Paths:** portable via `pathlib` (`PROJECT_ROOT = Path(__file__).resolve().parents[N]`)
- **Modules:** `preprocessing/data_preparation.py` (clean), `preprocessing/data_transformation.py` (RFM), `viz/eda.py` (tables + plots), `modeling/clustering.py` (segmentation)
- **Runtime:** a full run is ~4 min; Stage 6's Mean Shift and DBSCAN sweeps dominate

### Data Facts (verified 2026-08-01)
- Raw: 379,979 line items, 8 columns, dates 2021-01-04 → 2021-12-09
- No missing values in any column; `CustomerID` fully populated → **no imputation required**
- Cleansing drops 6,558 rows: duplicates 4,729; non-product service codes 1,799; non-positive price 30
- Cancellations are **retained, not deleted**: 7,840 lines held back and netted against customer value
- Transaction master: 365,581 purchase lines; RFM table: 4,199 customers; snapshot 2021-12-10
- Negative quantity ⟺ "C" invoice prefix is an exact 1:1 equivalence (8,215 raw rows, no exceptions)
- Netting removes GBP 454,445 (5.89%) of overstated revenue; 15 net-non-positive customers dropped
- No **Friday** trading anywhere in the source; trading window 06:00–20:00, 77% of revenue 10:00–15:59
- 191 of 3,590 stock codes (5.3%) carry >1 description → StockCode is the reliable product key
- UK = 89% of line items; strong Q4 (Sep–Nov) revenue seasonality

### Model Facts (verified 2026-08-01, post-netting)
- Model input: R/F/M percentile ratings scaled 0–5 (ranking removes the need for a separate scaler)
- 55 models fit across 4 techniques; best silhouette per technique: K-Means 0.469 (k=2),
  Agglomerative 0.431 (k=2), Mean Shift 0.469 (bw=1.6, 2 clusters), DBSCAN 0.208
- **Selected: K-Means k=4** at ≥3 clusters — silhouette 0.3789 and Davies-Bouldin 0.974 are both
  best, but the metrics do **not** agree unanimously: Mean Shift (bw=1.5, 3 clusters) takes
  Calinski-Harabasz 4,395.4 vs 4,269.3. Selection rests on 2 of 3, justified because silhouette and
  DB measure per-customer cluster fit (which drives banding correctness) and the CH gap is <3%.
- Holdout check (fit 80% / score unseen 20%): silhouette 0.3682 vs full-population 0.3789 → stable
- Segments: Gold 1,244 (29.6% of base, 76.1% of revenue) | Silver 1,776 (42.3%, 19.1%) |
  Bronze 1,180 (28.1%, 4.8%); total net revenue GBP 7,721,552
- Each band is 90–91% UK → the model split on behavior, not geography
- `Q4_SPEND_SHARE` must divide by `GROSS_MONETARY`, not net — a net denominator near zero for
  heavily-refunded customers blows the ratio up (was producing values in the trillions)

### Key Decisions
- Frame as **unsupervised segmentation** (no labeled target); organize around RFM
- Derived `TOTAL_PRICE = QUANTITY × UNIT_PRICE`
- RFM features: `RECENCY`, `FREQUENCY`, `MONETARY`, plus `AVG_ORDER_VALUE`
- Added `log1p` transforms of R/F/M to tame right-skew ahead of distance-based clustering
- Non-product `StockCode`s treated as service lines and removed: POST, DOT, C2, M, BANK CHARGES, D, CRUK, PADS
- Milestone 2 clusters on **percentile ratings**, not the log columns — ranking puts R/F/M on one
  0–5 scale, so no separate scaler is needed and outliers compress automatically. The `_log`
  columns remain in the RFM table for reporting.
- Model selection restricted to **≥3 clusters**: silhouette peaks at k=2, but two clusters cannot
  express three business tiers and would merge lapsed with occasional buyers
- With k=4 → 3 bands, the two middle clusters merge into Silver (they differ mainly on recency,
  96d vs 25d, at similar spend) — same approach as the instructor's reference notebook
- Bands assigned by ranking clusters on mean `RFM_SCORE`, never on raw cluster number
- `IS_UK` and `Q4_SPEND_SHARE` are built but deliberately **excluded from the feature matrix** —
  they profile segments rather than define them

### Assumptions
- `InvoiceNo` prefix "C" = cancellation — **verified** as an exact 1:1 match with negative quantity
- One customer row per `CustomerID`; `Country` is customer-level

### Open Questions
- ~~Which clustering algorithm and k?~~ **Resolved:** K-Means, k=4 (best silhouette + Davies-Bouldin;
  Mean Shift edges it on Calinski-Harabasz, see Model Facts)
- ~~Segment on log-scaled R/F/M or include `AVG_ORDER_VALUE`?~~ **Resolved:** percentile ratings
  of R/F/M only; AOV is derived from M÷F and would double-count monetary value
- ~~Treat non-UK customers separately or filter?~~ **Resolved:** kept all 4,199 customers;
  bands came out 90–91% UK anyway, so a filter would have changed little and lost the
  international opportunity
- Milestone 3: which campaign maps to each band, and how is lift measured?

### Known Issues & Technical Debt
- `requirements.txt` versions are unpinned.
- No `tests/` directory yet.
- `pipeline.py:117` emits a `Pandas4Warning` — `select_dtypes("object")` will stop matching
  string dtypes in pandas 3. Pre-existing (Milestone 1); harmless today, one-word fix.
- Agglomerative clustering is scored on the full rating matrix only; it has no `predict()`,
  so it cannot take part in the holdout stability check.
- `is_missing_vals()` and `detect_iqr_outlier()` remain unreachable from the pipeline.

### Future Improvements
- Milestone 3: campaign recommendations per segment; presentation deck
- Add unit tests for cleansing (`clean_data`), RFM (`build_rfm`), and band assignment
  (`band_customers` — verify Gold always has the highest mean `RFM_SCORE`)
- Pin dependency versions
- Cache Stage 6 results so re-running the pipeline for an EDA tweak skips the 3-minute sweep

---

## SECTION 6: CHANGE LOG

| Date       | Change                                                                       | Type   |
|------------|------------------------------------------------------------------------------|--------|
| 2026-07-08 | Built Milestone 1 pipeline (cleansing, RFM, EDA modules), portable paths, and the EDA & Data Cleansing deliverable | feat   |
| 2026-08-01 | Milestone 2: clustering models (4 techniques, 55 fits), Gold/Silver/Bronze banding | feat |
| 2026-08-01 | M1 review response: cancellations netted instead of deleted; added temporal, cancellation and product/description EDA (`viz/eda_addendum.py`) | feat |
| 2026-07-26 | Added `modeling/clustering.py` (4 clustering iterations, metric-based selection, Gold/Silver/Bronze banding), wired in as pipeline Stage 6, and generated the Data Wrangling & Model Output deliverable | feat   |

---

## SECTION 7: TESTING REQUIREMENTS

**Current state: No tests exist.**

Before marking work complete:
- Pipeline runs end-to-end without error (`python3 -m src.pipeline`)
- Generated tables/figures reflect the current data
- Recommended coverage: `clean_data()` (row-drop counts), `build_rfm()` (per-customer R/F/M
  correctness), EDA table shapes, `band_customers()` (Gold must hold the highest mean `RFM_SCORE`)

---

## SECTION 8: GIT WORKFLOW

Before any commit provide: summary of changes, risk assessment, recommended commit message.

**Prefixes:** `feat` / `fix` / `refactor` / `docs` / `test` / `chore`.
Never commit or force-push without approval.

---

## SECTION 9: PERFORMANCE REVIEW

- Excel ingest of ~380K rows is the heaviest Milestone 1 step; parquet caching of cleaned output avoids re-parsing downstream
- EDA figures are written to disk synchronously (headless `Agg` backend) — fine for offline use
- Stage 6 is now the slowest stage (~3 of ~4 min). K-Means and Agglomerative are trivial on
  4,199 customers; the cost is Mean Shift's 12 bandwidth fits and DBSCAN's 20-point grid, both
  of which scale super-linearly. Kept as-is because the assignment requires comparing techniques.

---

## SECTION 10: DEFINITION OF DONE

Work is **not complete** until:
- [ ] Requirements met and verified against expected outputs
- [ ] Pipeline runs end-to-end without error
- [ ] No leakage introduced; `random_state = 42` where randomized
- [ ] Documentation updated (this file and README)
- [ ] Project memory (Section 5) updated with decisions and outcomes
- [ ] Deliverable reviewed

---

## PROJECT PROGRESS TRACKER

| Stage | Description                  | Status      |
|-------|------------------------------|-------------|
| 1     | Data Ingestion               | Complete    |
| 2     | Data Cleansing               | Complete    |
| 3     | Transformation (RFM)         | Complete    |
| 4     | EDA                          | Complete    |
| 5     | Export Clean Master + RFM    | Complete    |
| —     | Milestone 1 Deliverable      | Complete    |
| 6     | Segmentation (clustering)    | Complete    |
| —     | Milestone 2 Deliverable      | Complete    |
| 7     | Campaign Recommendations     | Not Started (Milestone 3) |
