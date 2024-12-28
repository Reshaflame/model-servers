import pandas as pd
from utils.visualizations import ModelVisualizer
from models.isolation_forest import IsolationForest

if __name__ == "__main__":
    chunk_size = 50000  # Adjust chunk size
    reader = pd.read_csv('data/sampled_data/auth_sample.csv', chunksize=chunk_size, low_memory=False)

    processed_chunks = []
    visualizer = ModelVisualizer(model_name='IsolationForest')

    for chunk in reader:
        print(f"Processing chunk with shape: {chunk.shape}")

        # Dynamically identify feature columns
        feature_columns = [
            col for col in chunk.columns if col.startswith(('auth_type_', 'logon_type_', 'auth_orientation_', 'success_'))
        ]
        if not feature_columns:
            print("No valid feature columns in this chunk. Skipping...")
            continue

        # Drop rows with missing values in feature columns
        features = chunk[feature_columns].dropna()
        print(f"Features shape after dropping NaNs: {features.shape}")

        if features.empty:
            print("This chunk has no valid data after dropping NaNs. Skipping...")
            continue

        # Train Isolation Forest on the chunk
        iso_forest = IsolationForest(contamination=0.01, random_state=42)
        iso_forest.fit(features)

        # Predict anomalies
        chunk['anomaly'] = iso_forest.predict(features)
        processed_chunks.append(chunk)

    # Combine all processed chunks
    combined_data = pd.concat(processed_chunks, ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}")

    # Save the combined dataset
    combined_output_path = 'data/sampled_data/auth_with_anomalies_combined.csv'
    combined_data.to_csv(combined_output_path, index=False)
    print(f"Combined dataset with anomalies saved to {combined_output_path}")

    # Visualize combined results
    visualizer.visualize(
        data=combined_data,
        plot_type='categorical',
        x_col='src_comp',
        y_col='dst_comp',
        label_col='anomaly',
        title='Anomalies by Source and Destination Computers'
    )
    print("Visualization completed.")
