import pandas as pd
import gzip
import os


def generate_anomaly_only_file(auth_path='data/auth.txt.gz', 
                                redteam_path='data/redteam.txt.gz',
                                output_path='data/anomaly_auth.txt.gz'):
    print("[Generator] 🚀 Loading redteam ground truth...")
    redteam_columns = ["time", "user", "src_comp", "dst_comp"]
    with gzip.open(redteam_path, 'rt') as f:
        red_df = pd.read_csv(f, names=redteam_columns)

    # 🧼 Normalize redteam data
    for col in red_df.columns:
        red_df[col] = red_df[col].astype(str).str.strip().str.upper()

    redteam_events = set(zip(red_df["time"], red_df["user"], red_df["src_comp"], red_df["dst_comp"]))
    print(f"[Generator] ✅ Loaded {len(redteam_events)} redteam events.")

    print("[Generator] 📥 Reading auth data in chunks...")
    chunksize = 100_000
    filtered_rows = []

    auth_columns = ["time", "src_user", "dst_user", "src_comp", "dst_comp", "auth_type", "logon_type", "auth_orientation", "success"]

    with gzip.open(auth_path, 'rt') as auth_file:
        for chunk in pd.read_csv(auth_file, names=auth_columns, chunksize=chunksize):
            original_len = len(chunk)

            # 🧼 Normalize auth chunk
            for col in ["time", "src_user", "src_comp", "dst_comp"]:
                chunk[col] = chunk[col].astype(str).str.strip().str.upper()

            # 🔍 Check for matching redteam events
            matches = chunk.apply(
                lambda row: (
                    row["time"],
                    row["src_user"],
                    row["src_comp"],
                    row["dst_comp"]
                ) in redteam_events,
                axis=1
            )
            filtered = chunk[matches]
            filtered_rows.append(filtered)
            print(f"   └─ Chunk processed: {original_len} rows → {len(filtered)} anomalies")

    anomaly_df = pd.concat(filtered_rows, ignore_index=True)
    print(f"[Generator] ✅ Total anomaly rows collected: {len(anomaly_df)}")

    print(f"[Generator] 💾 Saving to {output_path}...")
    with gzip.open(output_path, 'wt') as out_file:
        anomaly_df.to_csv(out_file, index=False)

    print("[Generator] 🏁 Done!")


if __name__ == "__main__":
    generate_anomaly_only_file()
