import os
import pandas as pd
import json
from glob import glob

CHUNK_DIR = "data/labeled_data/chunks"  # Or wherever your labeled chunks are
OUTPUT_JSON = "data/expected_features.json"

def generate_expected_features():
    chunk_paths = sorted(glob(os.path.join(CHUNK_DIR, "*.csv")))
    if not chunk_paths:
        print("[❌] No labeled chunks found!")
        return

    df = pd.read_csv(chunk_paths[0])
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(numeric_columns, f)

    print(f"[✅] Exported {len(numeric_columns)} expected features to {OUTPUT_JSON}")

if __name__ == "__main__":
    generate_expected_features()
