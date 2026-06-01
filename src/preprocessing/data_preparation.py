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
    return clean.get_dataframe()