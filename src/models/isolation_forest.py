from sklearn.ensemble import IsolationForest
from src.utils.metrics import Metrics
import pandas as pd
import numpy as np

metrics = Metrics()

# Load preprocessed dataset
auth_data = pd.read_csv('data/sampled_data/auth_sample.csv', low_memory=False)
print("Loaded auth data shape:", auth_data.shape)

# Dynamically identify numerical and one-hot encoded columns
categorical_columns = [
    col for col in auth_data.columns if col.startswith(('auth_type_', 'logon_type_', 'auth_orientation_', 'success_'))
]

# Select only the columns available in the dataset
selected_features = categorical_columns
print("Selected feature columns:", selected_features)

# Handle missing values
# Option 1: Drop columns with excessive NaNs
# threshold = 0.5 * len(auth_data)
# selected_features = [col for col in selected_features if auth_data[col].isna().sum() < threshold]
# print("Selected features after dropping high-NaN columns:", selected_features)

# Option 2: Fill missing values (uncomment to use)
auth_data[selected_features] = auth_data[selected_features].fillna(0)

# Drop rows with remaining missing values
features = auth_data[selected_features].dropna()
print("Features shape after handling missing values:", features.shape)

# Initialize and train Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(features)

# Predict anomalies
predictions = iso_forest.predict(features)

# Convert predictions from (-1, 1) to (1 = anomaly, 0 = normal)
y_pred = np.where(predictions == -1, 1, 0)

# Generate dummy y_true (you currently have unlabeled data)
y_true = np.zeros_like(y_pred)  # Assume normal data as baseline (optional if you have labels)

# Call the metrics
results = metrics.compute_standard_metrics(y_true, y_pred)
print("Isolation Forest Metrics:", results)

# Add predictions to the dataset
auth_data = auth_data.loc[features.index]  # Align with features after handling missing values
auth_data['anomaly'] = predictions  # -1 = anomaly, 1 = normal
print("Anomalies added to the dataset.")

# Save results
output_path = 'data/sampled_data/auth_with_anomalies.csv'
auth_data.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# Debugging: Count anomalies
anomaly_count = (auth_data['anomaly'] == -1).sum()
print(f"Number of anomalies detected: {anomaly_count}")
