"""
Reusable chart functions for bar charts, scatter plots, histplots, correlation heatmaps, etc.
"""
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelPlots:
    _logger = logging.getLogger(__name__ + "." + __qualname__)

    def __init__(self, y_pred, y_test, X_test, X):
        self.parent_path = "/Users/phillipsmith/Desktop/pythonProjects/real-estate-price-prediction/docs"
        self.y_pred = y_pred
        self.y_test = y_test
        self.X_test = X_test
        self.X = X

    def plot_linearity_and_residuals(self, model: str, subdir="model_plots"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)

        f = plt.figure(figsize=(14,5))
        ax = f.add_subplot(121)
        sns.scatterplot(x=self.y_test,y=self.y_pred)
        ax.set_title('Check for Linearity:\n Actual Vs Predicted value')

        ax = f.add_subplot(122)
        sns.histplot((self.y_test - self.y_pred ))
        ax.axvline((self.y_test - self.y_pred).mean())
        ax.set_title('Check for Residual normality & mean: \n Residual eror')

        plt.savefig(f"{path}/{model}_linearity_residuals.png")
        plt.close()


    def plot_distribution(self, model: str, subdir="model_plots"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)

        f = plt.figure(figsize=(20,5))
        ax = f.add_subplot(121)
        sns.histplot(self.y_test)
        ax.set_title('Distribution of target variable (test data)')

        ax = f.add_subplot(122)
        sns.histplot(self.y_pred)
        ax.set_title('Distribution of predictions of target variable')

        plt.savefig(f"{path}/{model}_distribution.png")
        plt.close()

    def plot_pairplot(self, model: str, subdir="model_plots"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)

        self.X_test["y_pred"]=pd.Series(self.y_pred)
        self.X_test["y_test"]=pd.Series(self.y_test)
        sns.pairplot(self.X_test, x_vars=["y_pred","y_test"])

        plt.savefig(f"{path}/{model}_pairplot.png")
        plt.close()

    def plot_input_with_target(self, subdir="model_plots"):
        path = os.path.join(self.parent_path, subdir)
        os.makedirs(path, exist_ok=True)

        for feature in self.X.columns:
            plt.figure(figsize=(8, 5))
            plt.scatter(self.X_test[feature], self.y_test, color='black', label='Actual Data')
            sorted_indices = self.X_test[feature].argsort()
            plt.plot(self.X_test[feature].iloc[sorted_indices], self.y_pred[sorted_indices], color='red', label='Predictions')
            plt.xlabel(feature)
            plt.ylabel('Target')
            plt.title(f'{feature} vs Target with Random Forest Regression')
            plt.legend()

            plt.savefig(f"{path}/{feature}_random_forest.png")
            plt.close()