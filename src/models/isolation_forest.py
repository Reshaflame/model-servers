from sklearn.ensemble import IsolationForest
import pandas as pd

# Load datasets
auth_data = pd.read_csv('data/sampled_data/auth_sample.csv')
flows_data = pd.read_csv('data/sampled_data/flows_sample.csv')

# Inspect raw time columns (for debugging)
print("Raw auth time sample:", auth_data['time'].head())
print("Raw flows time sample:", flows_data['time'].head())

# Clean and parse 'time' column with specified format
auth_data['time'] = pd.to_datetime(auth_data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
flows_data['time'] = pd.to_datetime(flows_data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Drop rows with invalid time
auth_data = auth_data.dropna(subset=['time'])
flows_data = flows_data.dropna(subset=['time'])

# Downsample datasets
auth_data = auth_data.sample(frac=0.001, random_state=42)  # Adjust fraction as needed
flows_data = flows_data.sample(frac=0.001, random_state=42)

# Aggregate time by minute
auth_data['time'] = auth_data['time'].dt.floor('min')
flows_data['time'] = flows_data['time'].dt.floor('min')

# Check for overlapping times
common_times = set(auth_data['time']).intersection(set(flows_data['time']))
print("Number of overlapping times:", len(common_times))

# Merge datasets
merged_data = auth_data.merge(flows_data, on='time', how='inner')
print("Merged data shape:", merged_data.shape)

# Select numerical features for Isolation Forest
if not merged_data.empty:
    features = merged_data[['auth_type', 'logon_type', 'auth_orientation', 'success', 'duration', 'packets', 'bytes']]
    print("Features shape:", features.shape)

    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(features)

    # Predict anomalies
    predictions = iso_forest.predict(features)
    merged_data['anomaly'] = predictions  # Add anomaly column

    # Save results
    merged_data.to_csv('data/sampled_data/auth_sample_with_anomalies.csv', index=False)
    print("Results saved to data/sampled_data/auth_sample_with_anomalies.csv")
else:
    print("Merged data is empty. Check time column alignment.")
