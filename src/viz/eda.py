"""
Reusable chart functions for bar charts, scatter plots, histplots, correlation heatmaps, etc.
"""
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class EDA:
    _logger = logging.getLogger(__name__ + "." + __qualname__)

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.parent_path = "/Users/phillipsmith/Desktop/Python/quote_decision_predictor/docs"

    def plot_histogram_fd(self, vars_freedman_diaconis, subdir="eda_histograms"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)

        for var in vars_freedman_diaconis:
            fig, ax = plt.subplots(figsize=(14, 8))

            sns.histplot(
                self.df[var],
                bins='fd',
                kde=True,
                edgecolor='black',
                ax=ax
            )

            # Add count labels above each bin
            for patch in ax.patches:
                height = patch.get_height()
                if height > 0:
                    ax.annotate(
                        f'{int(height)}',
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )

            ax.set_title(var)

            plt.tight_layout()
            plt.savefig(f"{path}/{var}_hist.png")
            plt.close()

    def plot_binary_counts(self, vars_binary, subdir="eda_countplots"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)

        for var in vars_binary:
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.countplot(
                x=var,
                data=self.df,
                ax=ax
            )

            # Add count labels above bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', padding=3)

            ax.set_title(var)

            plt.tight_layout()
            plt.savefig(f"{path}/{var}_cnt.png")
            plt.close()

    def plot_scatter_plot(self, vars, subdir="eda_scatter"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10,4))
        for ax in vars:
            sns.scatterplot(
            data=self.df,
            x=ax[0], 
            y=ax[1],
            hue=ax[1]
            )

            plt.savefig(f"{path}/{ax[0]}_scatter.png")
            plt.close()

    def plot_correlation_matrix(self, remove_cols, subdir="eda_correlations"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)
        df_correlated = self.df.drop(columns=remove_cols)
        correlation_matrix = df_correlated.corr()
        plt.figure(figsize=(25, 25))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu')
        plt.savefig(f"{path}/corr_matrix.png")
        plt.close()

        self.df = df_correlated
        return self.df

    def plot_binned_dist(self, col, bins, labels, subdir="eda_binned"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)
    
        col_raw=self.df[[col]].copy()
        col_bins=bins
        col_labels=labels
        col_raw[f"{col}_bin"] = pd.cut(col_raw[col], bins=col_bins, labels=col_labels, include_lowest=True)
        col_raw[f"{col}_bin"] = (col_raw[f"{col}_bin"].cat.add_categories('NaN').fillna('NaN'))

        x_var = f"{col}_bin"
        plt.figure(figsize=(12, 8))
        plt.title(f"{col} Binned")
        ax = sns.histplot(data=col_raw, x=x_var)
        ax.bar_label(ax.containers[0], padding=3, fmt='%.0f')
        ax.margins(y=0.1)
        
        plt.savefig(f"{path}/{col}_binned.png")
        plt.close()

    def plot_binned_binary(self, col, target_col, bins, labels, subdir="eda_binned"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)
    
        col_raw=self.df[[col, target_col]].copy()
        col_bins=bins
        col_labels=labels

        col_raw[f"{col}_bin"] = pd.cut(
            col_raw[col], 
            bins=col_bins, 
            labels=col_labels, 
            include_lowest=True
        )

        col_raw[f"{col}_bin"] = (
            col_raw[f"{col}_bin"]
            .cat.add_categories('NaN')
            .fillna('NaN')
        )

        col_raw[target_col] = col_raw[target_col].map({
            0: 'Denied',
            1: 'Approved'
        })

        plt.figure(figsize=(12, 8))

        ax = sns.countplot(
            data=col_raw, 
            x=f"{col}_bin",
            hue=target_col
        )
        ax.bar_label(ax.containers[0], padding=3, fmt='%.0f')
        ax.bar_label(ax.containers[1], padding=3, fmt='%.0f')
        ax.margins(y=0.1)
        

        plt.title(f"{target_col} Decision Binned by {col}")
        plt.xticks(rotation=45)
        
        plt.savefig(f"{path}/{target_col}_binned_by_{col}.png")
        plt.close()

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

def explore_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate EDA from the transformed data source"""
    explore = EDA(df)

    bins = [17, 25, 35, 45, 55, 65, 75, 85, 95, 105]
    labels = ['17-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-95', '96+']
    explore.plot_binned_dist('AGE_PH', bins, labels)
    explore.plot_binned_binary('AGE_PH', 'STAT_Q', bins, labels)
    
    explore.plot_histogram_fd(
        vars_freedman_diaconis=[
            'EDU_PH',
            'DRIVE_RISK',
            'AGE_CAR',
            'AVG_MILE_DAILY',
            'ANNUAL_MILE',
            'NUM_CAR',
            'DRIVE_XP'
        ]
    )

    explore.plot_binary_counts(
        vars_binary=[
            'TYPE_Q',
            'STAT_Q',
            'SEX',
            'HOME_STAT',
            'OWN_CAR'
            ]
    )

    explore.plot_correlation_matrix(
        remove_cols=[
            'ID',
            'AGE_PH',
            'ANNUAL_MILE'
        ]
    )

    return explore.get_dataframe()