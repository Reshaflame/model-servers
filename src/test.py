import pandas as pd

def sample_preprocessed_data(file_path, sample_size=5):
    """
    Load and sample the preprocessed dataset to inspect column names and data structure.

    Args:
        file_path (str): Path to the preprocessed dataset.
        sample_size (int): Number of rows to sample.

    Returns:
        pd.DataFrame: Sampled data.
    """
    print(f"Loading dataset from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {data.shape}")

    print("Column names:")
    print(data.columns.tolist())

    sampled_data = data.sample(n=sample_size, random_state=42)
    print(f"Sampled {sample_size} rows:")
    print(sampled_data)

    # Check for NaN or Inf values
    print("\nChecking for invalid values in the dataset...")
    if data.isna().sum().sum() > 0:
        print("Dataset contains NaN values.")
    else:
        print("No NaN values found.")

    if (data == float('inf')).sum().sum() > 0 or (data == float('-inf')).sum().sum() > 0:
        print("Dataset contains Inf or -Inf values.")
    else:
        print("No Inf or -Inf values found.")

    print("\nBasic statistics of the dataset:")
    print(data.describe())

    return sampled_data

if __name__ == "__main__":
    # File path to the preprocessed labeled dataset
    labeled_data_path = "data/labeled_data/labeled_auth_sample.csv"

    # Sample the dataset
    sampled_data = sample_preprocessed_data(labeled_data_path)

    # Optionally, save the sampled data for further inspection
    sampled_output_path = "data/labeled_data/sample_output.csv"
    sampled_data.to_csv(sampled_output_path, index=False)
    print(f"Sampled data saved to {sampled_output_path}")
