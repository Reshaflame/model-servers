# src/models/isolation_forest.py
from sklearn.ensemble import IsolationForest
from skopt import gp_minimize
from skopt.space import Real, Integer
import numpy as np
from utils.metrics import Metrics

class IsolationForestModel:
    def __init__(self, contamination=0.05, n_estimators=100, max_samples='auto', random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state
        )
        self.metrics = Metrics()

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        preds = self.model.predict(X)
        y_pred = np.where(preds == -1, 1, 0)
        return self.metrics.compute_standard_metrics(y_true, y_pred)
