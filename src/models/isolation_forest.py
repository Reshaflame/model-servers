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

    def optimize_hyperparameters(self, X):
        def objective(params):
            contamination, n_estimators = params
            model = IsolationForest(
                contamination=contamination,
                n_estimators=int(n_estimators),
                random_state=42
            )
            preds = model.fit_predict(X)
            y_dummy = np.zeros_like(preds)
            y_pred = np.where(preds == -1, 1, 0)
            metrics = self.metrics.compute_standard_metrics(y_dummy, y_pred)
            return -(metrics['F1'])  # maximize F1

        space = [
            Real(0.01, 0.2, name='contamination'),
            Integer(50, 300, name='n_estimators')
        ]

        result = gp_minimize(objective, space, n_calls=20, random_state=42)
        print(f"Optimized contamination: {result.x[0]}, n_estimators: {int(result.x[1])}")

        # Update model with best parameters
        self.model.set_params(contamination=result.x[0], n_estimators=int(result.x[1]))
