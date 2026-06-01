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
            fig, ax = plt.subplots(figsize=(10, 6))

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

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

def explore_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate EDA from the transformed data source"""
    explore = EDA(df)
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

    # explore.plot_scatter_plot(
    #     vars=[
    #         ("x_col", "y_col"),
    #     ]
    # )
    explore.plot_correlation_matrix(
        remove_cols=[
            'STAT_Q',
            'ID'
        ]
    )

    return explore.get_dataframe()