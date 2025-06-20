# src/models/iso_backbone.py
from sklearn.ensemble import IsolationForest
import joblib

class IsoBackbone:
    """
    Thin wrapper around a scikit-learn Isolation Forest.
    • fit → train once, then freeze.
    • score→ return the *signed* anomaly score (+ = normal, − = anomaly).
    """
    def __init__(self,
                 contamination=0.05,
                 n_estimators=100,
                 max_samples='auto',
                 random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state
        )

    # ---------- training ----------
    def fit(self, X):
        self.model.fit(X)

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = joblib.load(path)
        return obj

    # ---------- inference ----------
    def raw_predict(self, X):
        """scikit output: +1 (normal)  /  −1 (anomaly)"""
        return self.model.predict(X)

    def score(self, X):
        """
        IsolationForest.decision_function gives *higher* scores for normal points.
        We flip it so that *higher → more anomalous* (like the deep models).
        """
        raw = self.model.decision_function(X)      # higher = normal
        return -raw                                # higher = anomaly
