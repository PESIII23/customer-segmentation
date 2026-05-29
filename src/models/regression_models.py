"""
Model definitions (Linear Regression, Random Forest, Lasso, Ridge), training, evaluation, and comparison.
"""
import pandas as pd
import numpy as np
import logging
import warnings

from src.viz import model_plots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Regression:
    """
    Class configs
    """
    _logger = logging.getLogger(__name__ + "." + __qualname__)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.ensemble._forest")

    """
    Constructor
    """
    def __init__(self, df: pd.DataFrame, test_size, random_state):
        self.df = df.copy()
        self.test_size = test_size
        self.random_state = random_state

    """
    Linear regression models the relationship between target and 
    independent variable by fitting a straight line to the data.
    """
    def linear_regression(self, X, y):
        model = LinearRegression(fit_intercept=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=self.test_size,
            random_state=self.random_state)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"      LINEAR REGRESSION RESULTS:\n")
        print(f"      INTERCEPT: {model.intercept_}")
        print(f"      COEFFICIENTS: {len(model.coef_)}")
        return y_pred, y_test, X_test, model, X
    
    """
    Random forest is a non-parametric, ensemble tree-based model.
    Partitions data to predict the mean value of target variables in leaf nodes.
    Ideal for fitting datasets with non-linear relationships with target variable.
    """
    def random_forest(self, X, y):
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
            oob_score=True
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=self.test_size,
            random_state=self.random_state)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"      RANDOM FOREST RESULTS:\n")
        print("      OUT-OF-BAG SCORE:", model.oob_score_)
        return y_pred, y_test, X_test, model, X
    
    """
    Lasso regression is a regularized linear regression technique used to improve
    generalization and handl high-dimensional data efficiently.
    Balances prediction accuracy and model simplicity by penalizing large coefficient values during training.
    """
    def lasso_regression(self, X, y):
        model = Lasso(
            alpha=0.01,
            fit_intercept=True,
            random_state=self.random_state
        )

        X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=self.test_size,
        random_state=self.random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"      LASSO REGRESSION RESULTS:\n")
        print(f"      PREDICTED VALUES: {np.round(y_pred[:3], 2)}")
        print(f"      REAL VALUES: {y_test[:3]}")
        return y_pred, y_test, X_test, model, X
    
    """
    Ridge regression is a type of linear regression that includes L2 regularization. 
    It adds a penalty term to the loss function equal to the square of the magnitude 
    of the coefficients discouraging large coefficient values, helping to prevent 
    overfitting and improve generalization.
    """
    def ridge_regression(self, X, y):
        model = Ridge(
            alpha=0.01,
            fit_intercept=True,
            random_state=self.random_state
        )

        X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=self.test_size,
        random_state=self.random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"      RIDGE REGRESSION RESULTS:\n")
        print(f"      PREDICTED VALUES: {np.round(y_pred[:3], 2)}")
        print(f"      REAL VALUES: {y_test[:3]}")
        return y_pred, y_test, X_test, model, X

    def print_results(self, y_pred, y_test, X_test, model):
        print(f"      MEAN ABSOLUTE ERROR: {mean_absolute_error(y_test, y_pred)}")
        print(f"      MEAN SQUARED ERROR: {mean_squared_error(y_test,y_pred)}")
        print(f"      ROOT MEAN SQUARED ERROR: {np.sqrt(mean_squared_error(y_test,y_pred))}")
        print(f"      R-SQUARED VALUE: {model.score(X_test,y_test)}\n")

def apply_regression(df):
    """
    Apply regression methods to the modeling data source
    """
    # LINEAR REGRESSION
    linear_regression = Regression(df, test_size=.2, random_state=3)
    X = df.drop(columns=['price', 'rad_3', 'population_distribution_popul_flip_log', 'crime_rate_per_capita_crim_log'])
    y = df['price']
    y_pred, y_test, X_test, model, X = linear_regression.linear_regression(X, y)
    linear_regression.print_results(y_pred, y_test, X_test, model)
    linear_reg_plots = model_plots.ModelPlots(y_pred=y_pred, y_test=y_test, X_test=X_test, X=X)
    linear_reg_plots.plot_linearity_and_residuals(model="linear_reg")
    linear_reg_plots.plot_distribution(model="linear_reg")
    linear_reg_plots.plot_pairplot(model="linear_reg")

    # RANDOM FOREST
    random_forest = Regression(df, test_size=.2, random_state=3)
    X = df.drop(columns=['price', 'population_distribution_popul_flip_log'])
    y = df['price']
    y_pred, y_test, X_test, model, X = random_forest.random_forest(X, y)
    random_forest.print_results(y_pred, y_test, X_test, model)
    random_forest_plots = model_plots.ModelPlots(y_pred=y_pred, y_test=y_test, X_test=X_test, X=X)
    random_forest_plots.plot_input_with_target()

    # LASSO REGRESSION
    lasso_regression = Regression(df, test_size=.2, random_state=3)
    X = df.drop(columns=['price'])
    y = df['price']
    y_pred, y_test, X_test, model, X = lasso_regression.lasso_regression(X, y)
    lasso_regression.print_results(y_pred, y_test, X_test, model)

    # RIDGE REGRESSION
    ridge_regression = Regression(df, test_size=.2, random_state=3)
    X = df.drop(columns=['price', 'rad_3', 'population_distribution_popul_flip_log', 'crime_rate_per_capita_crim_log'])
    y = df['price']
    y_pred, y_test, X_test, model, X = ridge_regression.ridge_regression(X, y)
    ridge_regression.print_results(y_pred, y_test, X_test, model)

    return