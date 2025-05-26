# src/models/isolation_forest.py

from sklearn.ensemble import IsolationForest
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
        """Train the isolation forest on the feature matrix."""
        self.model.fit(X)

    def predict(self, X):
        """Raw prediction: returns +1 (normal) or -1 (anomaly)."""
        return self.model.predict(X)

    def predict_labels(self, X):
        """Return binary labels: 1 = anomaly, 0 = normal — same format as GRU/LSTM."""
        return np.where(self.predict(X) == -1, 1, 0)

    def evaluate(self, X, y_true):
        """Evaluate predictions against ground truth using shared Metrics class."""
        y_pred = self.predict_labels(X)
        return self.metrics.compute_standard_metrics(y_true, y_pred)
