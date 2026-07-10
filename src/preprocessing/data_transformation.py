"""
Transform cleaned transaction lines into an analysis-ready customer table.

Aggregates line-item sales into RFM features (Recency, Frequency, Monetary)
per customer — the standard basis for the segmentation and campaign targeting
work in later milestones. Log transforms are added to tame right-skew ahead of
distance-based clustering.
"""
import pandas as pd
import numpy as np
import logging


class TransformData:
    _logger = logging.getLogger(__name__ + "." + __qualname__)

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def build_rfm(self, snapshot_date=None) -> pd.DataFrame:
        """Collapse transactions to one row per customer with R/F/M measures."""
        if snapshot_date is None:
            # day after the last invoice, so the most recent buyer has recency >= 1
            snapshot_date = self.df['INVOICE_DATE'].max() + pd.Timedelta(days=1)

        rfm = self.df.groupby('CUSTOMER_ID').agg(
            RECENCY=('INVOICE_DATE', lambda s: (snapshot_date - s.max()).days),
            FREQUENCY=('INVOICE_ID', 'nunique'),
            MONETARY=('TOTAL_PRICE', 'sum'),
        ).reset_index()

        rfm['AVG_ORDER_VALUE'] = rfm['MONETARY'] / rfm['FREQUENCY']
        self.snapshot_date = snapshot_date
        self.rfm = rfm
        self._logger.info(f"        Built RFM table for {len(rfm):,} customers.")
        return rfm

    def log_transform_plus_one(self, log_cols):
        for col in log_cols:
            self.rfm[f"{col}_log"] = np.log1p(self.rfm[col])
        return self.rfm

    def detect_iqr_outlier(self, outlier_cols):
        for col in outlier_cols:
            Q1 = self.rfm[col].quantile(0.25)
            Q3 = self.rfm[col].quantile(0.75)
            IQR = Q3 - Q1
            self.rfm[f"{col}_outlier"] = (
                (self.rfm[col] < Q1 - 1.5 * IQR) | (self.rfm[col] > Q3 + 1.5 * IQR)
            ).astype(int)
        return self.rfm

    def get_rfm(self) -> pd.DataFrame:
        return self.rfm


def transform_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build the customer RFM table from cleaned transactions."""
    transform = TransformData(df)
    transform.build_rfm()
    transform.log_transform_plus_one(['RECENCY', 'FREQUENCY', 'MONETARY'])
    return transform.get_rfm(), transform.snapshot_date
