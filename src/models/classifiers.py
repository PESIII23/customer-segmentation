"""
Classification models for quote-decision prediction.

A small collection of binary classifiers (logistic regression, random forest,
gradient boosting) sharing a common base that handles the train/test split,
fitting, evaluation, and ROC plotting. `run_models` runs them all in one stage
and returns a sorted comparison table.
"""
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

DOCS_PATH = Path(__file__).resolve().parents[2] / "docs" / "model_plots"


class ClassificationModel:
    """Base class: split, fit, evaluate, and plot ROC for a binary classifier."""
    _logger = logging.getLogger(__name__ + "." + __qualname__)
    name = "model"

    def __init__(self, test_size: float = 0.2, random_state: int = 42,
                 class_weight=None):
        self.test_size = test_size
        self.random_state = random_state
        self.class_weight = class_weight  # e.g. "balanced" to offset imbalance
        self.model = self._build()

    def _build(self):
        raise NotImplementedError

    def fit_predict(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)
        self.X_test, self.y_test = X_test, y_test
        self.y_pred = self.model.predict(X_test)
        self.y_prob = self.model.predict_proba(X_test)[:, 1]
        return self

    def evaluate(self, verbose: bool = True) -> dict:
        metrics = {
            "model": self.name,
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred),
            "recall": recall_score(self.y_test, self.y_pred),
            "f1": f1_score(self.y_test, self.y_pred),
            "roc_auc": roc_auc_score(self.y_test, self.y_prob),
        }
        if verbose:
            print(f"\n      {self.name.upper()}")
            for k, v in metrics.items():
                if k != "model":
                    print(f"        {k}: {v:.4f}")
            print(f"        confusion_matrix:\n{confusion_matrix(self.y_test, self.y_pred)}")
        return metrics

    def plot_roc(self, ax):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        auc = roc_auc_score(self.y_test, self.y_prob)
        ax.plot(fpr, tpr, label=f"{self.name} (AUC={auc:.3f})")

    def save_roc(self):
        """Save a standalone ROC chart for this model to docs/model_plots/."""
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_roc(ax)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)  # chance line
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=f"ROC Curve — {self.name}")
        ax.legend()
        DOCS_PATH.mkdir(parents=True, exist_ok=True)
        path = DOCS_PATH / f"{self.name}_roc.png"
        fig.savefig(path)
        plt.close(fig)
        return path


class LogisticRegressionModel(ClassificationModel):
    name = "logistic_regression"

    def _build(self):
        # Scale inside a pipeline so coefficients converge cleanly.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=self.random_state,
                               class_weight=self.class_weight),
        )


class RandomForestModel(ClassificationModel):
    name = "random_forest"

    def _build(self):
        return RandomForestClassifier(
            n_estimators=100, random_state=self.random_state,
            class_weight=self.class_weight,
        )


class GradientBoostingModel(ClassificationModel):
    name = "gradient_boosting"

    def _build(self):
        # sklearn's boosting (same family as XGBoost, no extra dependency).
        return GradientBoostingClassifier(random_state=self.random_state)


def run_models(df: pd.DataFrame, target: str = "STAT_Q",
               test_size: float = 0.2, random_state: int = 42,
               verbose: bool = True) -> pd.DataFrame:
    """Fit every model on one split; print metrics, save a combined ROC, return a comparison table."""
    X = df.drop(columns=[target])
    y = df[target]

    fig, ax = plt.subplots(figsize=(8, 6))
    results = []
    for cls in (LogisticRegressionModel, RandomForestModel, GradientBoostingModel):
        model = cls(test_size, random_state).fit_predict(X, y)
        results.append(model.evaluate(verbose))
        model.plot_roc(ax)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)  # chance line
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend()
    DOCS_PATH.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_PATH / "roc_curves.png")
    plt.close(fig)

    return pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
