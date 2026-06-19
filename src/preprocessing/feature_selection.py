"""
Chi-square feature selection.

Scores each (non-negative) feature against the target with the chi-square test
and keeps those significantly associated with it. Returns a modeling-ready
DataFrame of the selected features + target.
"""
import pandas as pd
import logging

from sklearn.feature_selection import chi2

class FeatureSelection:
    _logger = logging.getLogger(__name__ + "." + __qualname__)

    def __init__(self, df: pd.DataFrame, target: str = "STAT_Q", alpha: float = 0.05):
        self.df = df.copy()
        self.target = target
        self.alpha = alpha  # keep features with p < alpha

    def chi_square(self) -> pd.DataFrame:
        """Per-feature chi2 statistic and p-value, sorted by statistic."""
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # chi2 requires non-negative inputs; encoded features satisfy this.
        chi_stats, p_values = chi2(X, y)

        return pd.DataFrame({
            "feature": X.columns,
            "chi2": chi_stats,
            "p_value": p_values,
        }).sort_values("chi2", ascending=False).reset_index(drop=True)

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """Select features with p < alpha; return selected features + target."""
        scores = self.chi_square()
        selected = scores.loc[scores["p_value"] < self.alpha, "feature"].tolist()

        if verbose:
            print("      Chi-square scores (sorted by statistic):")
            print(scores.to_string(index=False))
            print(f"\n      Selected {len(selected)} feature(s) at alpha={self.alpha}:")
            print(f"      {selected}")

        return self.df[selected + [self.target]]
