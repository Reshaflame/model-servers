import pandas as pd
import json
import os
from tqdm import tqdm

INPUT_FILE = "data/auth.txt.gz"
OUTPUT_JSON = "data/auth_categories.json"
COLUMNS = [
    "time",
    "source_user@domain",
    "destination_user@domain",
    "source_computer",
    "destination_computer",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "label"
]

def extract_categories():
    print(f"🔍 Scanning: {INPUT_FILE}")
    chunk_size = 100_000
    auth_types = set()
    logon_types = set()
    orientations = set()

    with pd.read_csv(INPUT_FILE, names=COLUMNS, compression="gzip", chunksize=chunk_size) as reader:
        for chunk in tqdm(reader, desc="Processing chunks"):
            auth_types.update(chunk["auth_type"].dropna().unique())
            logon_types.update(chunk["logon_type"].dropna().unique())
            orientations.update(chunk["auth_orientation"].dropna().unique())

    category_dict = {
        "auth_type": sorted(auth_types),
        "logon_type": sorted(logon_types),
        "auth_orientation": sorted(orientations)
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(category_dict, f, indent=2)

    print(f"✅ Saved auth categories to: {OUTPUT_JSON}")

if __name__ == "__main__":
    extract_categories()
