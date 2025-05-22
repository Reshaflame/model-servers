import os
import pandas as pd

chunk_dir = "data/labeled_data/chunks"
bad_chunks = []

print(f"\n🔎 Scanning labeled chunks in: {chunk_dir}")

for file in sorted(os.listdir(chunk_dir)):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(chunk_dir, file)
    try:
        df = pd.read_csv(path, nrows=5)  # Only need the first few rows
        if "label" not in df.columns:
            print(f"⛔ MISSING 'label' in {file}")
            bad_chunks.append(file)
        elif df["label"].isnull().any():
            print(f"⚠️ NULLS in 'label' column in {file}")
            bad_chunks.append(file)
    except Exception as e:
        print(f"❌ Failed to load {file}: {e}")
        bad_chunks.append(file)

if bad_chunks:
    print(f"\n❌ Found {len(bad_chunks)} bad chunk(s):")
    for b in bad_chunks:
        print(f" - {b}")
else:
    print("\n✅ All labeled chunks look good!")
