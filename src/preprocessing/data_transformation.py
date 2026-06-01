"""
Transform data elements for EDA readiness...
"""
import pandas as pd
import numpy as np
import logging

class TransformData:
    _logger = logging.getLogger(__name__ + "." + __qualname__) 

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def convert_percent_to_decimal(self, percentage_cols):
        self.df[percentage_cols] = self.df[percentage_cols] / 100
        return self.df
    
    def convert_tax_rate_to_decimal(self, tax_rate_cols):
        self.df[tax_rate_cols] = self.df[tax_rate_cols] / 10000
        return self.df

    def scale_price(self, price_col):
        self.df[price_col] = self.df[price_col] * 10000
        return self.df
    
    def reverse_negative_skew(self, flip_cols):
        for col in flip_cols:
            self.df[f"{col}_flip"] = ((self.df[col].max() + 1) - self.df[col])
        return self.df
    
    def log_transform_plus_one(self, log_cols):
        for col in log_cols:
            self.df[f"{col}_log"] = np.log1p(self.df[col])
        return self.df
    
    def detect_iqr_outlier(self, outlier_cols):
        for col in outlier_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.df[f"{col}_outlier"] = ((self.df[col] < Q1 - 1.5*IQR) | 
                         (self.df[col] > Q3 + 1.5*IQR)).astype(int)
        return self.df

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all transformation methods to the cleaned data source"""
    transform = TransformData(df)
    return transform.get_dataframe()