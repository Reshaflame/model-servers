# save as inspect_chunk.py
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd

# Point to a labeled chunk instead of shared
chunk_path = "data/labeled_data/chunks/chunk_0_labeled.csv"

print(f"\n📂 Inspecting: {chunk_path}\n")
try:
    df = pd.read_csv(chunk_path)
except Exception as e:
    print(f"❌ Failed to read CSV: {e}")
    exit()

print("🔹 First 5 rows:")
print(df.head())

print("\n🔹 Column Names:")
print(list(df.columns))

print("\n🔹 Data Types:")
print(df.dtypes)

print("\n🔹 Null Count:")
print(df.isnull().sum())
