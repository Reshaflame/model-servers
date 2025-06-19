# scan_anomaly_chunks.py  (put it in data/chunks_labeled_phase2 or point CHUNKS_DIR to it)
import os, pandas as pd

CHUNKS_DIR   = "."                     # current directory = chunks folder
OUTPUT_FILE  = "anomaly_chunks.txt"    # results file

def scan_chunks():
    results = []
    for fname in os.listdir(CHUNKS_DIR):
        if not fname.endswith(".csv"): 
            continue
        path = os.path.join(CHUNKS_DIR, fname)
        try:
            # read only the label column – much faster
            df = pd.read_csv(path, usecols=["label"])
            if "label" not in df.columns:
                continue

            total      = len(df)
            anomalies  = (df["label"] == 1).sum()     # ✅ changed
            if anomalies:                             # keep only chunks with at least one anomaly
                results.append((fname, anomalies, anomalies / total))
        except Exception as e:
            print(f"❌ {fname}: {e}")

    with open(OUTPUT_FILE, "w") as fh:
        for fname, cnt, ratio in sorted(results, key=lambda x: (-x[1], x[0])):
            fh.write(f"{fname}, anomalies: {cnt}, ratio: {ratio:.4%}\n")

    print(f"✅ Finished. {len(results)} anomaly-bearing chunks → {OUTPUT_FILE}")

if __name__ == "__main__":
    scan_chunks()
