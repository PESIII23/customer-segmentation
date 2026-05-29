"""
Cleaning, wrangling, NaN handling, outlier removal, and train/test splitting...
"""
import pandas as pd
import logging

class CleanData:
    _logger = logging.getLogger(__name__ + "." + __qualname__) 

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def manage_cols(self):

        self.df = self.df.drop([
            'Unnamed: 0'
        ], axis=1)

        self.df = self.df.rename(columns={
            'crim': 'crime_rate_per_capita_crim',
            'zn': 'large_lot_zoning_ratio_zn',
            'indus': 'non_retail_acre_ratio_indus',
            'chas': 'river_boundary_flag_chas',
            'nox': 'nox_concentration_nox',
            'rm': 'avg_rooms_per_dwelling_rm',
            'age': 'pre_1940_housing_ratio_age',
            'dis': 'employment_center_distance_dis',
            'rad': 'radial_highway_access_idx_rad',
            'tax': 'property_tax_rate_tax',
            'ptratio': 'pupil_teacher_ratio_ptratio',
            'lstat': 'low_ses_population_pct_lstat',
            'popul': 'population_distribution_popul'
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
    clean.manage_cols()
    return clean.get_dataframe()