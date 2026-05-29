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
        self.parent_path = "/Users/phillipsmith/Desktop/pythonProjects/real-estate-price-prediction/docs"

    def plot_histogram_fd(self, vars_freedman_diaconis, subdir="eda_histograms"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)

        for var in vars_freedman_diaconis:
            fig, ax = plt.subplots(figsize=(10,4))
            sns.histplot(
                self.df[var],
                bins='fd',
                kde=True,
                edgecolor='black',
                ax=ax
            )

            ax.set_title(var)
            plt.savefig(f"{path}/{var}_hist.png")
            plt.close()

    def plot_binary_counts(self, vars_binary, subdir="eda_countplots"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)

        for var in vars_binary:
            fig, ax = plt.subplots(figsize=(10,4))
            sns.countplot(
                x=var, 
                data=self.df
            )

            ax.set_title(var)
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
        plt.figure(figsize=(25, 30))
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
            'crime_rate_per_capita_crim_log',
            'large_lot_zoning_ratio_zn_log',
            'non_retail_acre_ratio_indus_log',
            'nox_concentration_nox_log',
            'avg_rooms_per_dwelling_rm',
            'pre_1940_housing_ratio_age_flip_log',
            'employment_center_distance_dis_log',
            'property_tax_rate_tax_log',
            'pupil_teacher_ratio_ptratio_flip_log',
            'population_distribution_popul_flip_log',
            'low_ses_population_pct_lstat_log',
            'price'
        ]
    )
    explore.plot_binary_counts(
        vars_binary=['river_boundary_flag_chas']
    )
    explore.plot_scatter_plot(
        vars=[
            ("crime_rate_per_capita_crim_log", "price"),
            ("large_lot_zoning_ratio_zn_log", "price"),
            ("non_retail_acre_ratio_indus_log", "price"),
            ("nox_concentration_nox_log", "price"),
            ("avg_rooms_per_dwelling_rm", "price"),
            ("pre_1940_housing_ratio_age_flip_log", "price"),
            ("employment_center_distance_dis_log", "price"),
            ("property_tax_rate_tax_log", "price"),
            ("pupil_teacher_ratio_ptratio_flip_log", "price"),
            ("population_distribution_popul_flip_log", "price"),
            ("low_ses_population_pct_lstat_log", "price")
        ]
    )
    explore.plot_correlation_matrix(
        remove_cols=[
            'crime_rate_per_capita_crim', 
            'large_lot_zoning_ratio_zn', 
            'non_retail_acre_ratio_indus',
            'nox_concentration_nox',
            'pre_1940_housing_ratio_age',
            'pre_1940_housing_ratio_age_flip',
            'employment_center_distance_dis',
            'property_tax_rate_tax',
            'pupil_teacher_ratio_ptratio',
            'pupil_teacher_ratio_ptratio_flip',
            'population_distribution_popul',
            'population_distribution_popul_flip',
            'low_ses_population_pct_lstat',
            'price_outlier',
            'price_log'
        ]
    )

    return explore.get_dataframe()