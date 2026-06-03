# Quote Decision Predictor

Predictive machine learning system for insurance quote approval decisions using boosted classification models and sales analytics.

---

## Project Structure

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── docs/
│   ├── .gitkeep
│   ├── eda_correlations/
│   ├── eda_countplots/
│   └── eda_histograms/
└── src/
    ├── __init__.py
    ├── pipeline.py
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── data_preparation.py
    │   ├── data_transformation.py
    │   └── feature_engineering.py
    ├── models/
    │   ├── __init__.py
    │   └── regression_models.py
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── viz/
    │   ├── __init__.py
    │   ├── eda.py
    │   └── model_plots.py
    └── notebooks/
        └── quote_decision_predictor.ipynb
```

---

## Quick Start

**1. Create a virtual environment:**

```bash
conda create --name myenv
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

* Maintainer: Phillip Smith
* Documentation: [Milestone 1](https://docs.google.com/document/d/1d_OWABJuWxOc7XirqNIVVhhuZt3xJWAoKOeZA1D3iBs/edit?usp=sharing) | 
* Repository: [GitHub](https://github.com/PESIII23/quote_decision_predictor)
