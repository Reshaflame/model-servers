import pandas as pd

# Load a small sample from the preprocessed dataset
file_path = "data/sampled_data/auth_sample.csv"
chunk_size = 1000  # Number of rows to sample for quick analysis
data = pd.read_csv(file_path, nrows=chunk_size)

# Identify non-Boolean columns in the sample
non_boolean_columns = [
    col for col in data.columns
    if not data[col].dropna().isin([0, 1]).all()
]

# Display non-Boolean columns and their unique values
print(f"--- Inspecting a Sample of {chunk_size} Rows ---")
print("Non-Boolean Columns:")
for col in non_boolean_columns:
    print(f"{col}: {data[col].nunique()} unique values")

# Suggest potential columns for visualization
print("\nSuggested Columns for Visualization:")
for col in non_boolean_columns:
    if pd.api.types.is_numeric_dtype(data[col]):
        print(f"Numeric Column: {col}")
    else:
        print(f"Categorical Column (needs encoding): {col}")
