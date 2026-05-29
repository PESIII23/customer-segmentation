# Quote Decision Predictor

Predictive machine learning system for insurance quote approval decisions using boosted classification models and sales analytics.

---

## Project Structure

```text
src/
├── preprocessing/
│   ├── data_prep.py             # Cleaning, wrangling, train/test split
│   └── feature_engineering.py   # Feature creation and transformation
├── models/
│   └── model.py                 # Model definitions, training, evaluation
├── notebooks/
│   └── quote_decision_model.ipynb  # EDA and modeling notebook
├── data/
│   ├── raw/                     # Source quote and customer data
│   └── processed/               # Cleaned data outputs
└── viz/
    └── plotting.py              # Reusable visualization utilities
```

---

## Quick Start

**1. Create a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run the notebook:**

```text
Place source data files in src/data/raw/, then open
src/notebooks/quote_decision_model.ipynb.
```

---

## Pipeline Stages

| Stage                                  | Description                                                                 |
| -------------------------------------- | --------------------------------------------------------------------------- |
| **1. Data Loading**                    | Read raw quote and customer data into a DataFrame                           |
| **2. Data Cleaning**                   | Standardize column names, handle missing values, and apply initial cleaning |
| **3. Data Transformation**             | Prepare and transform data for analysis and modeling                        |
| **4. Exploratory Data Analysis (EDA)** | Generate statistical summaries, visualizations, and identify trends         |
| **5. Feature Engineering**             | Create and prepare features for modeling                                    |
| **6. Export Modeling Data**            | Save processed datasets for downstream use                                  |
| **7. Modeling**                        | Train and evaluate predictive models                                        |
| **8. Evaluation & Validation**         | Assess model performance and compare results                                |
| **9. Iteration & Interpretation**      | Analyze outcomes, refine features, and summarize findings                   |

---

## Key Features

* **End-to-End Modular Pipeline** – Clean separation of data cleaning, transformation, feature engineering, modeling, and evaluation for reproducibility and maintainability
* **Robust Data Preparation** – Standardized workflows for handling and transforming raw data
* **Comprehensive EDA** – Statistical summaries and visualizations for understanding customer and quote behavior
* **Flexible Modeling Workflow** – Modular architecture that supports experimentation and iteration
* **Evaluation & Validation** – Consistent approach to assessing model performance
* **Automated Export** – Processed datasets are saved for downstream use and reproducibility
* **Clear Visualizations** – Reusable plotting utilities for analysis and reporting
* **Scalable & Extensible** – Designed for adaptation to new datasets, features, and business requirements

---

## Tools & Libraries

* Python 3.13+
* pandas, numpy – Data manipulation
* scikit-learn – Modeling and preprocessing
* matplotlib, seaborn – Visualization

---

## Support

* Maintainer: Data Science Team
* Documentation: docs/
* Repository: [GitHub Repository URL]

This mirrors the structure and tone of your original Real Estate README while remaining implementation-agnostic. 
