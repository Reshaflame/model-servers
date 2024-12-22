# Main script to load data, run models, and make decisions

import sys
import os
import pandas as pd

# Add the `src` directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.isolation_forest import IsolationForestModel

# Placeholder for other models
# from src.models.lstm_rnn import LSTMModel
# from src.models.gru import GRUModel

class MultiModelProcessor:
    def __init__(self):
        self.models = []  # Array to hold all models
        self.decision_method = self.majority_voting  # Decision method
    
    def add_model(self, model):
        self.models.append(model)
    
    def majority_voting(self, predictions):
        # Simple majority voting to decide if there's an anomaly
        return sum(predictions) / len(predictions) < 0.5  # True if majority vote is an anomaly (-1)
    
    def process(self, X):
        results = []
        for model in self.models:
            results.append(model.is_anomaly(X))
        return self.decision_method(results)

if __name__ == "__main__":
    # Load unlabeled data (sampled)
    data = pd.read_csv('data/sampled_data/auth_sample.csv')
    features = data[['feature1', 'feature2', 'feature3']]  # Replace with actual feature columns
    
    # Initialize processor
    processor = MultiModelProcessor()
    
    # Add Isolation Forest model
    iso_forest_model = IsolationForestModel(contamination=0.01)
    iso_forest_model.fit(features)
    processor.add_model(iso_forest_model)
    
    # (Optional) Add other models later
    # processor.add_model(LSTMModel(...))
    # processor.add_model(GRUModel(...))
    
    # Process data
    is_anomalous = processor.process(features)
    print("Anomaly Detected?" , is_anomalous)
