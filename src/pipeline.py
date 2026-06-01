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
    ENV:      conda deactivate quotes --> conda activate quotes
    Python:   from src.pipeline import run_pipeline; full_df, modeling_df = run_pipeline()
"""
import pandas as pd
from pathlib import Path
from src.preprocessing import data_preparation, data_transformation, feature_engineering
from src.viz import eda, model_plots
from src.models import regression_models

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
        # df_cleaned.to_parquet(Paths.RAW_PARQUET, engine='fastparquet')
        # df_cleaned = pd.read_parquet(Paths.RAW_PARQUET, engine='fastparquet')
        # missing = data_preparation.CleanData(df_cleaned).is_missing_vals()
        # if missing:
        #     log("      WARNING: Missing values detected in cleaned data.\n")
        #     # apply imputation or other missing data approaches
        log(f"      Data is cleaned.\n")

        # """
        # STAGE 3: TRANSFORM DATA
        # """
        # log("\n[3/9] TRANSFORMING DATA...\n")
        # df_transformed = data_transformation.transform_data(df_cleaned)
        # log("      Data is transformed.\n")

        # """
        # STAGE 4: EXPLORATORY DATA ANALYSIS
        # """
        # log("\n[4/9] PERFORMING EDA...\n")
        # df_correlated = eda.explore_data(df_transformed)
        # log("      Generated EDA plots. Please see docs/eda_*.\n")

        # """
        # STAGE 5: FEATURE ENGINEERING
        # """
        # log("\n[5/9] ENGINEERING FEATURES...\n")
        # engineer = feature_engineering.FeatureEngineer(df_correlated, n_splits=10, n_neighbors=10)
        # # get baseline error
        # X = df_correlated.drop(columns='price')
        # y = df_correlated['price']
        # rmse, r2 = engineer.evaluate_error(X, y)
        # print(f"      BASELINE ERROR: RMSE={rmse} --> R^2={r2}")
        # # get SFS error and engineered dataframe
        # df_modeled = feature_engineering.engineer_features(df_correlated)
        # log(f"      Created {len(df_modeled.columns) - 1} input features.")
        # log(df_modeled.columns)
        
        # """
        # STAGE 6: EXPORT MODELING DATAFRAME
        # """
        # log("\n[6/9] EXPORTING MODELING DATA...\n")
        # Paths.MODELING_DATA.parent.mkdir(parents=True, exist_ok=True)
        # df_modeled.to_parquet(Paths.MODELING_DATA, engine='fastparquet', index=False)
        # log(f"      Exported modeling dataframe with {len(df_modeled.columns)} total features.")

        # """
        # STAGE 7: APPLY REGRESSION MODELS TO DATAFRAME
        # """
        # log("\n[7/9] APPLYING REGRESSION AND EVALUATING MODELS...\n")
        # df = pd.read_parquet(Paths.MODELING_DATA, engine='fastparquet')
        # regression_models.apply_regression(df)
        
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