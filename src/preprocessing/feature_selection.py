"""
Determine optimal feature selector for ML model inputs...
"""
import pandas as pd
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import f1_score, mean_squared_error, precision_score, r2_score, recall_score, accuracy_score, roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from math import sqrt

class FeatureSelector:
    _logger = logging.getLogger(__name__ + "." + __qualname__)

    def __init__(self, df: pd.DataFrame, n_splits, n_neighbors=None):
        self.df = df.copy()
        self.cv = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
        self.classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))

    def evaluate_regression_error(self, X, y, n_neighbors):
        self.classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))
        y_pred = cross_val_predict(self.classifier_pipeline, X, y, cv=self.cv)
        rmse = round(sqrt(mean_squared_error(y, y_pred)),2)
        r2 = round(r2_score(y, y_pred),2)
        return rmse, r2
    
    def evaluate_classification_error(self, X, y, n_neighbors):
        self.classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))
        y_pred = cross_val_predict(self.classifier_pipeline, X, y, cv=self.cv)
        y_prob = cross_val_predict(self.classifier_pipeline, X, y, cv=self.cv, method='predict_proba')[:, 1]
        auc = roc_auc_score(y, y_prob)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        return auc, precision, recall, f1, accuracy
    
    def get_sequential_forward_selection(self, k_features, col_y):
        sfs1 = SFS(
            self.classifier_pipeline,
            k_features=k_features,
            forward=True,
            scoring='roc_auc',
            cv=self.cv
        )
        best_k = 9
        X_1 = self.df.drop(columns=col_y)
        y_1 = self.df[col_y]
        sfs1.fit(X_1, y_1)
        return self.df[list(sfs1.subsets_[best_k]['feature_names'])]

def evaluate_neighbors(df: pd.DataFrame) -> pd.DataFrame:

    selector = FeatureSelector(df, n_splits=5, n_neighbors=None)

    results = []
    neighbors = [3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100]
    X = df.drop(columns=['ID', 'STAT_Q'])
    y = df['STAT_Q']
    for n in neighbors:
        auc, precision, recall, f1, accuracy = selector.evaluate_classification_error(X, y, n_neighbors=n)
        results.append((n, auc, precision, recall, f1, accuracy))

    best_result = max(results, key=lambda x: x[1])

    print("      Best n:", best_result[0])
    print("      Best AUC:", best_result[1])
    print("      Best Precision:", best_result[2])
    print("      Best Recall:", best_result[3])
    print("      Best F1:", best_result[4])
    print("      Best Accuracy:", best_result[5])

def select_features(df: pd.DataFrame) -> pd.DataFrame:

    selector = FeatureSelector(df, n_splits=5, n_neighbors=50)

    X = df.drop(columns=['ID', 'STAT_Q'])
    y = df['STAT_Q']

    return 

    # print(f"      SEQUENTIAL FORWARD SELECTION ERROR: RMSE={rmse} --> R^2={r2}")
    # return pd.concat([X, y], axis=1)