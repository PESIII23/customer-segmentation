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
    
    def encode_one_hot(self, col):
        """Refactor to encode list of variables"""
        dummies = pd.get_dummies(self.df[col], prefix='rad').astype(int)
        self.df = self.df.drop(columns=col).merge(dummies, left_index=True, right_index=True)
        return self.df

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
    transform.encode_one_hot(col='radial_highway_access_idx_rad')
    transform.convert_percent_to_decimal(
        percentage_cols=[
            'large_lot_zoning_ratio_zn', 
            'non_retail_acre_ratio_indus', 
            'pre_1940_housing_ratio_age', 
            'low_ses_population_pct_lstat'
        ]
    )
    transform.convert_tax_rate_to_decimal('property_tax_rate_tax')
    transform.scale_price('price')
    transform.reverse_negative_skew(
        flip_cols=[
            'pre_1940_housing_ratio_age',
            'pupil_teacher_ratio_ptratio',
            'population_distribution_popul'
        ]
    )
    transform.log_transform_plus_one(
        log_cols=[
            'crime_rate_per_capita_crim', 
            'large_lot_zoning_ratio_zn', 
            'non_retail_acre_ratio_indus',
            'nox_concentration_nox',
            'pre_1940_housing_ratio_age_flip',
            'employment_center_distance_dis',
            'property_tax_rate_tax',
            'pupil_teacher_ratio_ptratio_flip',
            'population_distribution_popul_flip',
            'low_ses_population_pct_lstat',
            'price'
        ]
    )

    # find sum of outliers in jupyter to generate outlier_cols
    transform.detect_iqr_outlier(
        outlier_cols=[
            'crime_rate_per_capita_crim_log',
            'large_lot_zoning_ratio_zn_log',
            'low_ses_population_pct_lstat_log',
            'price'
        ]
    )
    return transform.get_dataframe()