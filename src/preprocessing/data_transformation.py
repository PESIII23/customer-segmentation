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

    def encode_binary(self, col):
        values = self.df[col].unique()
        mapping = {
            values[0]: 0,
            values[1]: 1
        }
        self.df[col] = self.df[col].map(mapping)
        return self.df, mapping
    
    def encode_reverse_binary(self, col):
        values = self.df[col].unique()
        mapping = {
            values[0]: 1,
            values[1]: 0
        }
        self.df[col] = self.df[col].map(mapping)
        return self.df, mapping
    
    def encode_label(self, col, mapping):
        self.df[col] = self.df[col].astype(str).map(mapping)
        return self.df


    def encode_one_hot(self, col, prefix):
        dummies = pd.get_dummies(self.df[col], prefix=prefix).astype(int)
        self.df = self.df.drop(columns=col).merge(dummies, left_index=True, right_index=True)
        self._logger.info("        One hot encoding applied to selected columns.")
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

    transform.encode_binary('TYPE_Q')
    transform.encode_reverse_binary('STAT_Q')
    transform.encode_reverse_binary('SEX')
    transform.encode_binary('HOME_STAT')
    transform.encode_reverse_binary('OWN_CAR')

    drive_xp_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Very High": 3
    }

    edu_ph_map = {
        "Below High School": 0,
        "High School": 1,
        "Bachelors": 2,
        "Masters": 3,
        "Above Masters": 4
    }

    drive_risk_map = {
        "Very Low": 0,
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Very High": 4
    }

    age_car_map = {
        "<1": 0,
        "1 to 2": 1,
        "2 to 5": 2,
        "5 to 10": 3,
        "10+": 4
    }

    avg_mile_daily_map = {
        "<5": 0,
        "5 to 10": 1,
        "10 to 25": 2,
        "25 to 50": 3,
        "50+": 4
    }

    annual_mile_map = {
        "<5K": 0,
        "5K-12K": 1,
        "12K-30K": 2,
        "30K-50K": 3,
        "50K+": 4
    }

    num_car_map = {
        "1": 0,
        "2": 1,
        "3": 2,
        "3+": 3
    }

    transform.encode_label('DRIVE_XP', drive_xp_map)
    transform.encode_label('EDU_PH', edu_ph_map)
    transform.encode_label('DRIVE_RISK', drive_risk_map)
    transform.encode_label('AGE_CAR', age_car_map)
    transform.encode_label('AVG_MILE_DAILY', avg_mile_daily_map)
    transform.encode_label('ANNUAL_MILE', annual_mile_map)
    transform.encode_label('NUM_CAR', num_car_map)

    transform.encode_one_hot('MARR_STAT', 'MARR_STAT')
    transform.encode_one_hot('TYPE_CAR', 'TYPE_CAR')

    return transform.get_dataframe()