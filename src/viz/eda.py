"""
Reusable EDA utilities: statistical/profiling tables, distribution plots,
categorical bar charts, and a correlation heatmap.

Writes figures under docs/ and summary tables under docs/tables/. Paths are
resolved relative to the project root so the pipeline is portable.
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS = PROJECT_ROOT / "docs"

sns.set_theme(style="whitegrid")


class EDA:
    _logger = logging.getLogger(__name__ + "." + __qualname__)

    def __init__(self, docs_dir: Path = DOCS):
        self.docs = Path(docs_dir)

    def _subdir(self, name: str) -> Path:
        path = self.docs / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---- tables -----------------------------------------------------------
    def numeric_summary(self, df: pd.DataFrame, cols, name: str) -> pd.DataFrame:
        """Mean/median/std/quartiles etc. for numerical attributes."""
        summary = df[cols].describe().T
        summary["median"] = df[cols].median()
        summary = summary[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]
        path = self._subdir("tables")
        summary.round(2).to_csv(path / f"{name}.csv")
        return summary.round(2)

    def categorical_profile(self, df: pd.DataFrame, cols, name: str) -> pd.DataFrame:
        """Distinct counts + most-frequent value for categorical attributes."""
        rows = []
        for c in cols:
            vc = df[c].value_counts()
            rows.append({
                "attribute": c,
                "distinct_values": df[c].nunique(),
                "most_frequent": vc.index[0],
                "most_frequent_count": int(vc.iloc[0]),
                "most_frequent_pct": round(100 * vc.iloc[0] / len(df), 1),
                "missing": int(df[c].isna().sum()),
            })
        profile = pd.DataFrame(rows)
        path = self._subdir("tables")
        profile.to_csv(path / f"{name}.csv", index=False)
        return profile

    # ---- distributions ----------------------------------------------------
    def plot_histogram(self, df: pd.DataFrame, cols, subdir, bins=40, log_x=False):
        path = self._subdir(subdir)
        for var in cols:
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.histplot(df[var], bins=bins, kde=True, edgecolor="black", ax=ax)
            if log_x:
                ax.set_xscale("log")
            ax.set_title(var)
            plt.tight_layout()
            plt.savefig(path / f"{var}_hist.png", dpi=110)
            plt.close()

    def plot_box(self, df: pd.DataFrame, cols, subdir):
        path = self._subdir(subdir)
        for var in cols:
            fig, ax = plt.subplots(figsize=(10, 3.2))
            sns.boxplot(x=df[var], ax=ax, color="#4C72B0")
            ax.set_title(f"{var} — spread & outliers")
            plt.tight_layout()
            plt.savefig(path / f"{var}_box.png", dpi=110)
            plt.close()

    # ---- categorical bars -------------------------------------------------
    def plot_top_bar(self, series_counts: pd.Series, title, fname, subdir, xlabel="count"):
        path = self._subdir(subdir)
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(x=series_counts.values, y=series_counts.index.astype(str),
                    color="#4C72B0", ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(path / f"{fname}.png", dpi=110)
        plt.close()

    def plot_timeline(self, df: pd.DataFrame, subdir):
        """Monthly revenue and invoice volume to show the sales cycle."""
        path = self._subdir(subdir)
        monthly = df.set_index("INVOICE_DATE").resample("ME").agg(
            revenue=("TOTAL_PRICE", "sum"),
            invoices=("INVOICE_ID", "nunique"),
        )
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(range(len(monthly)), monthly["revenue"], color="#4C72B0")
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels([d.strftime("%b") for d in monthly.index], rotation=0)
        ax.set_title("Monthly Revenue (2021)")
        ax.set_ylabel("revenue")
        plt.tight_layout()
        plt.savefig(path / "monthly_revenue.png", dpi=110)
        plt.close()
        return monthly

    # ---- correlation ------------------------------------------------------
    def plot_correlation_matrix(self, df: pd.DataFrame, cols, subdir, fname="corr_matrix"):
        path = self._subdir(subdir)
        corr = df[cols].corr(method="pearson")
        fig, ax = plt.subplots(figsize=(9, 7.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu", center=0,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title("Pearson Correlation — Customer RFM Features")
        plt.tight_layout()
        plt.savefig(path / f"{fname}.png", dpi=120)
        plt.close()
        corr.round(2).to_csv(self._subdir("tables") / f"{fname}.csv")
        return corr.round(2)
