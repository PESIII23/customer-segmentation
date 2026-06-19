# PROJECT ORCHESTRATION MANUAL & GUARDRAILS
# Quote Decision Predictor

---

## SECTION 0: ACTIVE PROJECT ROLES

**Assigned Role:** Senior Machine Learning Engineer

### Role Decision Framework
All decisions are evaluated through the lens of a Senior MLE:
- Prioritize model correctness, reproducibility, and generalization
- Enforce proper train/validation/test separation and leakage prevention
- Design modular, testable, production-aware ML pipelines
- Favor interpretable model choices with measurable evaluation criteria
- Flag architectural or data issues before they compound downstream

---

## SECTION 1: SYSTEM SAFETY & EXECUTION RULES

### Data Safety
- Source data (`src/data/raw/`) is **immutable** — never overwrite original files
- All transformed outputs go to `src/data/processed/`
- Raw parquet cache (`dataset.parquet`) is a derivative — safe to regenerate, never treat as source of truth

### Context Management
Never print entire DataFrames. Prefer:
- `df.head()`, `df.sample()`, `df.info()`, `df.describe()`

### Deterministic Execution
All randomized processes must use:
```python
random_state = 42
```
> **Current violation:** `FeatureSelector` uses `random_state=0` and `Regression` uses `random_state=3`. Standardize to 42.

### Data Leakage Prevention
- `StandardScaler` in `FeatureSelector` is applied inside the CV pipeline via `make_pipeline` — **correct**
- Future scalers, encoders, and imputers must **only be fit on training folds**, never on full datasets

### Verification Policy
All results must be labeled as:
- **Assumed** — not yet validated
- **Estimated** — from cross-validation or approximation
- **Verified** — confirmed with held-out test data

### Three-Strike Circuit Breaker
If the same error occurs three consecutive times:
1. Stop execution
2. Summarize root cause
3. Recommend next steps
4. Request user guidance before continuing

---

## SECTION 2: OPERATING PRINCIPLES

### Plan Before Action
Before any code change:
- Define the objective
- List affected files
- Identify risks
- Define success criteria

### Small Iterative Changes
Prefer incremental updates. Validate after every step.

### Backward Compatibility
Before modifying existing modules, identify all callers and evaluate downstream impact.

### Code Comment Style
Keep comments concise — short and high-signal. Prefer a one-line rationale over multi-line explanation. Do not narrate what the code obviously does; comment only the *why* or non-obvious tradeoffs. Trim docstrings to essentials (purpose, key constraints, usage) — no verbose multi-paragraph blocks or long inline option lists.

---

## SECTION 3: FILE MODIFICATION SAFETY

Before modifying any file:
1. Read the entire file
2. Identify all dependencies and callers
3. Summarize intended changes

**Require explicit approval before:**
- Deleting files or models
- Dropping or overwriting datasets
- Force-pushing to any branch
- Modifying pipeline entry points in `pipeline.py`

---

## SECTION 4: DEBUGGING FRAMEWORK

When debugging:
1. Reproduce the issue
2. Isolate root cause
3. Form a hypothesis
4. Apply one fix at a time
5. Validate the fix
6. Document findings

Never stack speculative fixes.

---

## SECTION 5: PROJECT MEMORY

### Objectives
- Build a classification model to predict insurance quote approval decisions (`STAT_Q`: 0=Denied, 1=Approved)
- Target: Geico auto insurance quote data (`P1_Geico_Quote_Data.xlsx`)
- Maintainer: Phillip Smith

### Architecture
- **Entry point:** `src/pipeline.py` → `run_pipeline()`
- **Data path:** `src/data/raw/P1_Geico_Quote_Data.xlsx` → parquet cache → `src/data/processed/modeling_df.parquet`
- **Stages:** Ingest → Clean → Transform → EDA → Feature Selection → (Export) → (Modeling)
- **Absolute paths** are hardcoded in `pipeline.py` and `eda.py` — portability risk

### Key Decisions
- Target variable is `STAT_Q` (binary: 0=Denied, 1=Approved)
- Label encoding used for ordinal features (DRIVE_XP, EDU_PH, DRIVE_RISK, AGE_CAR, AVG_MILE_DAILY, ANNUAL_MILE, NUM_CAR)
- One-hot encoding applied to nominal features (MARR_STAT, TYPE_CAR)
- Binary encoding applied to: TYPE_Q, STAT_Q, SEX, HOME_STAT, OWN_CAR
- KNN classifier used for initial feature evaluation via cross-validated AUC

### Assumptions
- `STAT_Q` encodes quote outcome: 0=Denied, 1=Approved (confirmed via `encode_reverse_binary`)
- `AGE_PH` is policyholder age (numeric, binned 17–105+)
- `ZIP` is excluded from modeling (positional column, not yet encoded)

### Known Issues & Technical Debt
1. **Stages 6–9 are commented out** in `pipeline.py` — modeling export, regression, and evaluation are incomplete
2. **`regression_models.py` has undefined variables** — `apply_regression()` references `X` and `y` before assignment (lines 144, 155, 163, 170)
3. **`select_features()` in `feature_selection.py`** accepts `n_neighbors` as a param but `FeatureSelector.__init__` does not — would raise `TypeError` at runtime
4. **Hardcoded absolute paths** in `pipeline.py` (line 22) and `eda.py` (line 16) — breaks on any other machine
5. **`random_state` inconsistency** — `FeatureSelector` uses 0, `Regression` uses 3; should be standardized to 42
6. **No tests exist** — no `tests/` directory with any test files
7. **`ZIP` column is loaded but never used** — no encoding or drop decision documented
8. **`MARR_STAT` encoding** — one-hot produces unknown number of dummies; no documentation of expected categories

### Open Questions
- What is the final modeling approach? (KNN used for feature eval, but regression models defined — classification vs. regression needs resolution)
- What features should be passed to the final model? (Feature selection outputs evaluated but not consumed)
- Should `ZIP` be encoded, binned by region, or dropped?
- What is the business success threshold for model performance?

### Risks
- **Data leakage** if scaler or encoder is ever fit outside the pipeline (currently safe)
- **Model type mismatch** — `regression_models.py` implements regression on a binary classification target (`STAT_Q`); logistic regression or a tree-based classifier is more appropriate
- **No reproducibility guarantee** — no `environment.yml` or pinned `requirements.txt` versions

### Future Improvements
- Replace regression approach with classification models (LogisticRegression, RandomForestClassifier, XGBoost)
- Replace hardcoded paths with `pathlib` relative paths from project root
- Add `tests/` with unit tests for preprocessing and feature selection
- Pin dependency versions in `requirements.txt`
- Add `environment.yml` for conda environment reproducibility
- Implement `select_features()` to consume KNN evaluation output and return a feature subset

---

## SECTION 6: CHANGE LOG

| Date       | Change                                                                 | Type   |
|------------|------------------------------------------------------------------------|--------|
| 2026-06-11 | Created `claude.md` — initial repository audit and project memory      | docs   |
| 2024       | Simplified feature selector class (`514161a`)                          | refactor |
| 2024       | Wrapped up EDA; feature selection added (`bba7a1a`)                   | feat   |
| 2024       | Data cleaning, transformation, and EDA complete (`f5af3cb`)           | feat   |
| 2024       | Initial data cleaning (`2bba19a`)                                      | feat   |

---

## SECTION 7: TESTING REQUIREMENTS

**Current state: No tests exist.**

Before marking any work complete:
- Validate that existing pipeline stages run end-to-end without error
- New preprocessing logic requires unit tests in `tests/preprocessing/`
- Model evaluation requires validation against a held-out test set (not just CV)
- No regressions allowed in existing pipeline stages

Recommended test coverage gaps:
- `CleanData.rename_cols()` — assert expected column names post-rename
- `TransformData.encode_binary()` — assert 0/1 values and no nulls
- `FeatureSelector.evaluate_neighbors()` — assert results structure and best_n selection
- End-to-end pipeline smoke test with synthetic data

---

## SECTION 8: GIT WORKFLOW

Before any commit, provide:
- Summary of changes
- Risk assessment
- Recommended commit message

**Commit prefix conventions:**
```
feat:      new functionality
fix:       bug fix
refactor:  restructure without behavior change
docs:      documentation only
test:      test additions or changes
chore:     dependency, config, or tooling changes
```

Never commit automatically without approval.
Never force-push without explicit instruction.

---

## SECTION 9: PERFORMANCE REVIEW

- **KNN cross-validation** runs 11 neighbor values × 5 folds × 2 passes (predict + predict_proba) = 110 CV fits — acceptable for current data size, may become slow at scale
- **EDA plots** write to disk synchronously — acceptable for offline use
- **Parquet caching** of raw transformed data is a good pattern — avoids re-parsing Excel on every run
- No GPU or distributed compute considerations currently needed

---

## SECTION 10: DEFINITION OF DONE

Work is **not complete** until:
- [ ] Requirements are met and verified against expected outputs
- [ ] Pipeline runs end-to-end without error
- [ ] No data leakage introduced
- [ ] `random_state = 42` used consistently
- [ ] Documentation updated (this file and README if applicable)
- [ ] Project memory (Section 5) updated with decisions and outcomes
- [ ] Risks documented
- [ ] Commit message reviewed and approved

---

## REPOSITORY AUDIT TRACKER

| Item                          | Status      | Notes |
|-------------------------------|-------------|-------|
| Repository Inventory          | Complete    | Single project: `quote_decision_predictor` |
| Architecture Review           | Complete    | 9-stage pipeline, stages 6–9 incomplete |
| Dependency Review             | Complete    | `requirements.txt` unpinned; no conda env file |
| Technical Debt Review         | Complete    | 8 issues identified; see Section 5 |
| Knowledge Gaps Identified     | Complete    | See Open Questions in Section 5 |
| Project Memory Initialized    | Complete    | Section 5 populated |
| Change Plan Approved          | Pending     | Awaiting user direction |
| Changes Implemented           | Not Started | — |
| Validation Completed          | Not Started | — |
| Documentation Updated         | In Progress | `claude.md` created |

---

## PROJECT PROGRESS TRACKER

| Stage | Description                  | Status      |
|-------|------------------------------|-------------|
| 1     | Data Ingestion               | Complete    |
| 2     | Data Cleaning                | Complete    |
| 3     | Data Transformation          | Complete    |
| 4     | EDA                          | Complete    |
| 5     | Feature Selection (eval)     | Complete    |
| 6     | Export Modeling Data         | Not Started |
| 7     | Modeling                     | Not Started |
| 8     | Evaluation & Validation      | Not Started |
| 9     | Iteration & Interpretation   | Not Started |
