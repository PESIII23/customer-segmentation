"""
Geico Quote Decision Predictor

Orchestrates the end-to-end quote decision analysis workflow:
    1. Data ingestion from imported data file
    2. Exploratory Data Analysis
    3. Data transformations (data value mods, log transforms)
    4. Feature engineering
    5. Export modeling-ready DataFrame

Usage:
    CLI:      python3 -m src.pipeline
    ENV:      conda deactivate --> conda activate quotes
    Python:   from src.pipeline import run_pipeline; full_df, modeling_df = run_pipeline()
"""
import pandas as pd
from pathlib import Path
from src.preprocessing import data_preparation, data_transformation, feature_selection
from src.viz import eda, model_plots
from src.models import classifiers

PROJECT_ROOT = Path('/Users/phillipsmith/Desktop/Python/quote_decision_predictor')

class Paths:
    RAW_FILE = PROJECT_ROOT / 'src' / 'data' / 'raw' / 'P1_Geico_Quote_Data.xlsx'
    RAW_PARQUET = PROJECT_ROOT / 'src' / 'data' / 'raw' / 'dataset.parquet'
    MODELING_DATA = PROJECT_ROOT / 'src' / 'data' / 'processed' / 'modeling_df.parquet'

    pd.set_option('display.max_columns', None)

def run_pipeline(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full pipeline. Returns (full_df, modeling_df).
        """
        log = print if verbose else lambda *args, **kwargs: None
        
        """
        HEADER
        """
        print()
        log("=" * 60)
        log("GEICO QUOTE DECISION PREDICTION PIPELINE")
        log("=" * 60)
        
        """
        STAGE 1: INGEST DATA
        """
        log("\n[1/9] LOADING DATA FROM SOURCE FILE...\n")
        df = pd.read_excel(Paths.RAW_FILE, header=0)
        print(df.columns)
        log(f"      Loaded {len(df):,} records.")
        log(f"      DataFrame shape: {df.shape}\n")
        
        """
        STAGE 2: CLEAN DATA
        """
        log("\n[2/9] CLEANING DATA...\n")
        df_cleaned = data_preparation.clean_data(df)
        # missing = data_preparation.CleanData(df_cleaned).is_missing_vals()
        # if missing:
        #     log("      WARNING: Missing values detected in cleaned data.\n")
        #     # apply imputation or other missing data approaches
        log(f"      Data is cleaned.\n")

        """
        STAGE 3: TRANSFORM DATA
        """
        log("\n[3/9] TRANSFORMING DATA...\n")
        df_transformed = data_transformation.transform_data(df_cleaned)
        df_transformed.to_parquet(Paths.RAW_PARQUET, engine='fastparquet')
        df_transformed = pd.read_parquet(Paths.RAW_PARQUET, engine='fastparquet')
        log("      Data is transformed.\n")

        """
        STAGE 4: EXPLORATORY DATA ANALYSIS
        """
        log("\n[4/9] PERFORMING EDA...\n")
        df_eda = eda.explore_data(df_transformed)
        log("      Generated EDA plots. Please see docs/eda_*.\n")
        print(df_eda.columns)

        """
        STAGE 5: FEATURE SELECTION
        """
        log("\n[5/9] SELECTING FEATURES...\n")
        selector = feature_selection.FeatureSelection(df_eda)
        df_modeling = selector.run(verbose=verbose)
        log(f"\n      Modeling DataFrame shape: {df_modeling.shape}\n")

        """
        STAGE 6: MODELING
        """
        log("\n[6/9] TRAINING AND EVALUATING MODELS...\n")
        results = classifiers.run_models(df_modeling, verbose=verbose)
        log("\n      Model comparison (sorted by ROC-AUC):")
        log(results.to_string(index=False))
        log("\n      ROC curves saved to docs/model_plots/roc_curves.png\n")

        """
        FOOTER
        """
        log("\n" + "=" * 60)
        log("PIPELINE COMPLETE")
        log("=" * 60 + '\n')
        # log(f"Modeling DataFrame: {df_modeled.shape}\n")
        
        return


if __name__ == "__main__":
    run_pipeline(verbose=True)