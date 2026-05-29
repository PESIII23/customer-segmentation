"""
Engineer features for applying ML models...
"""
import pandas as pd
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from math import sqrt

class FeatureEngineer:
    _logger = logging.getLogger(__name__ + "." + __qualname__)

    def __init__(self, df: pd.DataFrame, n_splits, n_neighbors):
        self.df = df.copy()
        self.cv = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        self.classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))

    def evaluate_error(self, X, y):
        y_pred = cross_val_predict(self.classifier_pipeline, X, y, cv=self.cv)
        rmse = round(sqrt(mean_squared_error(y, y_pred)),2)
        r2 = round(r2_score(y, y_pred),2)
        return rmse, r2
    
    def get_sequential_forward_selection(self, k_features, col_y):
        sfs1 = SFS(
            self.classifier_pipeline,
            k_features=k_features,
            forward=True,
            scoring='neg_mean_squared_error',
            cv=self.cv
        )
        best_k = 9
        X_1 = self.df.drop(columns=col_y)
        y_1 = self.df[col_y]
        sfs1.fit(X_1, y_1)
        return self.df[list(sfs1.subsets_[best_k]['feature_names'])]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering methods to the transformed data source"""
    engineer = FeatureEngineer(df, n_splits=10, n_neighbors=10)
    X = engineer.get_sequential_forward_selection(k_features=9, col_y='price')
    y = df['price']
    rmse, r2 = engineer.evaluate_error(X, y)
    print(f"      SEQUENTIAL FORWARD SELECTION ERROR: RMSE={rmse} --> R^2={r2}")
    return pd.concat([X, y], axis=1)