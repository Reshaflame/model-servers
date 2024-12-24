from sklearn.ensemble import IsolationForest
import pandas as pd

# Load preprocessed datasets
auth_data = pd.read_csv('data/sampled_data/auth_sample.csv')
flows_data = pd.read_csv('data/sampled_data/flows_sample.csv')

# Downsample to reduce size
auth_data = auth_data.sample(frac=0.01, random_state=42)
flows_data = flows_data.sample(frac=0.01, random_state=42)
print("Auth data shape after downsampling:", auth_data.shape)
print("Flows data shape after downsampling:", flows_data.shape)

# Ensure 'time' is datetime and drop invalid rows
auth_data['time'] = pd.to_datetime(auth_data['time'], errors='coerce')
flows_data['time'] = pd.to_datetime(flows_data['time'], errors='coerce')
auth_data = auth_data.dropna(subset=['time'])
flows_data = flows_data.dropna(subset=['time'])

# Sort by 'time' for merge_asof
auth_data = auth_data.sort_values('time')
flows_data = flows_data.sort_values('time')

# Merge datasets with a 1-second tolerance
merged_data = pd.merge_asof(
    auth_data,
    flows_data,
    on='time',
    direction='nearest',
    tolerance=pd.Timedelta('1s')
)
print("Merged data shape:", merged_data.shape)

# Downsample merged data
merged_data = merged_data.sample(frac=0.01, random_state=42)
print("Downsampled merged data shape:", merged_data.shape)

# Select features for Isolation Forest
features = merged_data[['auth_type', 'logon_type', 'auth_orientation', 'success', 'duration', 'packets', 'bytes']]
print("Selected features columns:", features.columns)
print("Features shape:", features.shape)

# Initialize and train Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(features)

# Predict anomalies
predictions = iso_forest.predict(features)

# Add predictions to the merged dataset
merged_data['anomaly'] = predictions  # Add anomaly column

# Save results
merged_data.to_csv('data/sampled_data/merged_with_anomalies.csv', index=False)
print("Results saved to data/sampled_data/merged_with_anomalies.csv")
