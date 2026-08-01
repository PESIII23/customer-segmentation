"""
Cleaning and wrangling for the Online Retail sales data.

Removes non-product service codes, zero-price lines and exact duplicates, then
splits the remainder into purchase lines and cancellation lines. Cancellations
are retained rather than discarded so customer value can be reported net of
refunds (see data_transformation.build_rfm). Records row counts at each step.
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

    def remove_nonproduct_codes(self):
        """Drop postage, manual, discount, and bank-charge service lines."""
        before = len(self.df)
        self.df = self.df[~self.df['STOCK_CODE'].isin(NON_PRODUCT_CODES)].reset_index(drop=True)
        self.report["dropped_nonproduct"] = before - len(self.df)
        return self.df

    def remove_nonpositive_price(self):
        """Price must be positive on any real line; quantity sign carries meaning."""
        before = len(self.df)
        self.df = self.df[self.df['UNIT_PRICE'] > 0].reset_index(drop=True)
        self.report["dropped_nonpositive_price"] = before - len(self.df)
        return self.df

    def add_total_price(self):
        self.df['TOTAL_PRICE'] = self.df['QUANTITY'] * self.df['UNIT_PRICE']
        return self.df

    def split_cancellations(self):
        """Separate cancellation lines from purchase lines.

        A 'C' invoice prefix and a negative quantity are equivalent in this
        source (verified: 8,215 rows each, no exceptions), so either test
        identifies the same rows. Cancellations are kept for netting, not
        discarded, because deleting them while keeping the original order
        overstates customer value.
        """
        is_cancel = self.df['INVOICE_ID'].astype(str).str.startswith('C')
        self.cancellations = self.df[is_cancel].reset_index(drop=True)
        self.df = self.df[~is_cancel].reset_index(drop=True)
        self.report["cancellations_retained"] = len(self.cancellations)
        self.report["purchases_retained"] = len(self.df)
        return self.df

    def get_report(self) -> dict:
        self.report["rows_end"] = len(self.df)
        return self.report

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    def get_cancellations(self) -> pd.DataFrame:
        return self.cancellations


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply all cleaning steps.

    Returns (purchases, cancellations, cleaning_report). Purchases are the
    transaction master; cancellations are retained separately so monetary value
    can be reported net of refunds.
    """
    clean = CleanData(df)
    clean.rename_cols()
    clean.drop_duplicates()
    clean.remove_nonproduct_codes()
    clean.remove_nonpositive_price()
    clean.add_total_price()
    clean.split_cancellations()
    return clean.get_dataframe(), clean.get_cancellations(), clean.get_report()
