"""
Cleaning, wrangling, NaN handling, outlier removal, and train/test splitting...
"""
import pandas as pd
import logging

class CleanData:
    _logger = logging.getLogger(__name__ + "." + __qualname__) 

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def rename_cols(self):
        headers = self.df.columns.values.tolist()
        self.df = self.df.rename(columns={
            headers[0]: 'ID',
            headers[1]: 'ZIP',
            headers[2]: 'TYPE_Q',
            headers[3]: 'STAT_Q',
            headers[4]: 'AGE_PH',
            headers[5]: 'SEX',
            headers[6]: 'MARR_STAT',
            headers[7]: 'HOME_STAT',
            headers[8]: 'DRIVE_XP',
            headers[9]: 'EDU_PH',
            headers[10]: 'DRIVE_RISK',
            headers[11]: 'AGE_CAR',
            headers[12]: 'TYPE_CAR',
            headers[13]: 'OWN_CAR',
            headers[14]: 'AVG_MILE_DAILY',
            headers[15]: 'ANNUAL_MILE',
            headers[16]: 'NUM_CAR'
        })
        self._logger.info("        Renamed column headers.")
        return self.df
    
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

    def is_missing_vals(self):
        missing_vals_cols = []
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                missing_vals_cols.append(col)
        self._logger.info(f"        Columns missing data: {missing_vals_cols}.")
        return len(missing_vals_cols) > 0
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning methods to the data source"""
    clean = CleanData(df)
    clean.rename_cols()

    clean.encode_binary('TYPE_Q')
    clean.encode_reverse_binary('STAT_Q')
    clean.encode_reverse_binary('SEX')
    clean.encode_binary('HOME_STAT')
    clean.encode_reverse_binary('OWN_CAR')

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

    clean.encode_label('DRIVE_XP', drive_xp_map)
    clean.encode_label('EDU_PH', edu_ph_map)
    clean.encode_label('DRIVE_RISK', drive_risk_map)
    clean.encode_label('AGE_CAR', age_car_map)
    clean.encode_label('AVG_MILE_DAILY', avg_mile_daily_map)
    clean.encode_label('ANNUAL_MILE', annual_mile_map)
    clean.encode_label('NUM_CAR', num_car_map)

    clean.encode_one_hot('MARR_STAT', 'MARR_STAT')
    clean.encode_one_hot('TYPE_CAR', 'TYPE_CAR')

    return clean.get_dataframe()