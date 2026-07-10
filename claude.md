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
- Milestone 1 (this deliverable): EDA & data cleansing; establish a clean master file
- Maintainer: Phillip Smith

### Architecture
- **Entry point:** `src/pipeline.py` → `run_pipeline()` (5 stages)
- **Data flow:** raw Excel → cleaned transactions → customer RFM table → EDA artifacts
- **Paths:** portable via `pathlib` (`PROJECT_ROOT = Path(__file__).resolve().parents[N]`)
- **Modules:** `preprocessing/data_preparation.py` (clean), `preprocessing/data_transformation.py` (RFM), `viz/eda.py` (tables + plots)

### Data Facts (verified 2026-07-08)
- Raw: 379,979 line items, 8 columns, dates 2021-01-04 → 2021-12-09
- No missing values in any column; `CustomerID` fully populated → **no imputation required**
- Cleansing removed 14,398 rows (3.8%): duplicates 4,729; cancellations 8,191; non-product service codes 1,448; non-positive qty/price 30
- Clean master: 365,581 transactions; RFM table: 4,214 customers; snapshot 2021-12-10
- UK = 89% of line items; strong Q4 (Sep–Nov) revenue seasonality

### Key Decisions
- Frame as **unsupervised segmentation** (no labeled target); organize around RFM
- Derived `TOTAL_PRICE = QUANTITY × UNIT_PRICE`
- RFM features: `RECENCY`, `FREQUENCY`, `MONETARY`, plus `AVG_ORDER_VALUE`
- Added `log1p` transforms of R/F/M to tame right-skew ahead of distance-based clustering
- Non-product `StockCode`s treated as service lines and removed: POST, DOT, C2, M, BANK CHARGES, D, CRUK, PADS

### Assumptions
- `InvoiceNo` prefix "C" = cancellation (paired with negative quantity)
- One customer row per `CustomerID`; `Country` is customer-level

### Open Questions
- Which clustering algorithm and k for Milestone 2 (K-Means vs. hierarchical)?
- Segment on log-scaled R/F/M only, or include `AVG_ORDER_VALUE`?
- Treat non-UK customers as a separate segmentation or a filter?

### Known Issues & Technical Debt
- `requirements.txt` versions are unpinned.
- No `tests/` directory yet.

### Future Improvements
- Milestone 2: scale features, run K-Means (elbow + silhouette), profile segments
- Milestone 3: campaign recommendations per segment; presentation deck
- Add unit tests for cleansing (`clean_data`) and RFM (`build_rfm`)
- Pin dependency versions

---

## SECTION 6: CHANGE LOG

| Date       | Change                                                                       | Type   |
|------------|------------------------------------------------------------------------------|--------|
| 2026-07-08 | Built Milestone 1 pipeline (cleansing, RFM, EDA modules), portable paths, and the EDA & Data Cleansing deliverable | feat   |

---

## SECTION 7: TESTING REQUIREMENTS

**Current state: No tests exist.**

Before marking work complete:
- Pipeline runs end-to-end without error (`python3 -m src.pipeline`)
- Generated tables/figures reflect the current data
- Recommended coverage: `clean_data()` (row-drop counts), `build_rfm()` (per-customer R/F/M correctness), EDA table shapes

---

## SECTION 8: GIT WORKFLOW

Before any commit provide: summary of changes, risk assessment, recommended commit message.

**Prefixes:** `feat` / `fix` / `refactor` / `docs` / `test` / `chore`.
Never commit or force-push without approval.

---

## SECTION 9: PERFORMANCE REVIEW

- Excel ingest of ~380K rows is the heaviest step; parquet caching of cleaned output avoids re-parsing downstream
- EDA figures are written to disk synchronously (headless `Agg` backend) — fine for offline use
- Clustering in later milestones on 4,214 customers is trivial in cost

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
| 6     | Segmentation (clustering)    | Not Started (Milestone 2) |
| 7     | Campaign Recommendations     | Not Started (Milestone 3) |
