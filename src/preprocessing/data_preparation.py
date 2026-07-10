"""
Cleaning and wrangling for the Online Retail sales data.

Removes transactional artifacts (cancellations, non-product service codes,
zero-price lines, exact duplicates) so a clean master file can support
customer segmentation. Records the row count dropped at each step for reporting.
"""
import pandas as pd
import logging

# StockCodes that are service/adjustment lines, not sellable products.
NON_PRODUCT_CODES = {"POST", "DOT", "C2", "M", "BANK CHARGES", "D", "CRUK", "PADS"}


class CleanData:
    _logger = logging.getLogger(__name__ + "." + __qualname__)

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = {"rows_start": len(self.df)}  # audit trail of rows dropped

    def rename_cols(self):
        headers = self.df.columns.values.tolist()
        self.df = self.df.rename(columns={
            headers[0]: 'INVOICE_ID',
            headers[1]: 'STOCK_CODE',
            headers[2]: 'DESCRIPTION',
            headers[3]: 'QUANTITY',
            headers[4]: 'INVOICE_DATE',
            headers[5]: 'UNIT_PRICE',
            headers[6]: 'CUSTOMER_ID',
            headers[7]: 'COUNTRY',
        })
        self._logger.info("        Renamed column headers.")
        return self.df

    def is_missing_vals(self):
        missing = [c for c in self.df.columns if self.df[c].isna().sum() > 0]
        self._logger.info(f"        Columns missing data: {missing}.")
        return len(missing) > 0

    def drop_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        self.report["dropped_duplicates"] = before - len(self.df)
        return self.df

    def remove_cancellations(self):
        """Cancelled orders carry a 'C' invoice prefix and negative quantity."""
        before = len(self.df)
        is_cancel = self.df['INVOICE_ID'].astype(str).str.startswith('C')
        self.df = self.df[~is_cancel].reset_index(drop=True)
        self.report["dropped_cancellations"] = before - len(self.df)
        return self.df

    def remove_nonproduct_codes(self):
        """Drop postage, manual, discount, and bank-charge service lines."""
        before = len(self.df)
        self.df = self.df[~self.df['STOCK_CODE'].isin(NON_PRODUCT_CODES)].reset_index(drop=True)
        self.report["dropped_nonproduct"] = before - len(self.df)
        return self.df

    def remove_nonpositive(self):
        """Remaining rows must have positive quantity and price to be a real sale."""
        before = len(self.df)
        self.df = self.df[(self.df['QUANTITY'] > 0) & (self.df['UNIT_PRICE'] > 0)].reset_index(drop=True)
        self.report["dropped_nonpositive"] = before - len(self.df)
        return self.df

    def add_total_price(self):
        self.df['TOTAL_PRICE'] = self.df['QUANTITY'] * self.df['UNIT_PRICE']
        return self.df

    def get_report(self) -> dict:
        self.report["rows_end"] = len(self.df)
        return self.report

    def get_dataframe(self) -> pd.DataFrame:
        return self.df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply all cleaning steps; returns (clean_df, cleaning_report)."""
    clean = CleanData(df)
    clean.rename_cols()
    clean.drop_duplicates()
    clean.remove_cancellations()
    clean.remove_nonproduct_codes()
    clean.remove_nonpositive()
    clean.add_total_price()
    return clean.get_dataframe(), clean.get_report()
