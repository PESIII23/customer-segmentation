"""
Determine optimal feature selector for ML model inputs...
"""
import pandas as pd
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import f1_score, mean_squared_error, precision_score, r2_score, recall_score, accuracy_score, roc_auc_score
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from math import sqrt

class FeatureSelector:
    _logger = logging.getLogger(__name__ + "." + __qualname__)

    def __init__(self, df: pd.DataFrame, n_splits=5):
        self.df = df.copy()
        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=0
        )

    def evaluate_neighbors(self) :

        X = self.df.drop(columns=['STAT_Q'])
        y = self.df['STAT_Q']

        results = []
        neighbors = [3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100]

        for n in neighbors:
            pipeline = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=n)
            )

            y_pred = cross_val_predict(
                pipeline,
                X,
                y,
                cv=self.cv
            )

            y_prob = cross_val_predict(
                pipeline,
                X,
                y,
                cv=self.cv,
                method='predict_proba'
            )[:, 1]

            results.append(
                (
                    n,
                    roc_auc_score(y, y_prob),
                    precision_score(y, y_pred),
                    recall_score(y, y_pred),
                    f1_score(y, y_pred),
                    accuracy_score(y, y_pred)
                )
            )
            
            best_result = max(results, key=lambda x: x[1])

        print("      Best n:", best_result[0])
        print("      Best AUC:", best_result[1])
        print("      Best Precision:", best_result[2])
        print("      Best Recall:", best_result[3])
        print("      Best F1:", best_result[4])
        print("      Best Accuracy:", best_result[5])

def select_features(df: pd.DataFrame) -> pd.DataFrame:

    selector = FeatureSelector(df, n_splits=5, n_neighbors=40)

    X = df.drop(columns=['STAT_Q'])
    y = df['STAT_Q']

    return 

    # print(f"      SEQUENTIAL FORWARD SELECTION ERROR: RMSE={rmse} --> R^2={r2}")
    # return pd.concat([X, y], axis=1)