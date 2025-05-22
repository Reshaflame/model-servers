import pandas as pd
import gzip

class AnomalyMerger:
    def __init__(self, chunk_path, anomaly_path, output_path, total_rows=100):
        self.chunk_path = chunk_path
        self.anomaly_path = anomaly_path
        self.output_path = output_path
        self.total_rows = total_rows

    def merge(self):
        # Load both files
        chunk_df = pd.read_csv(self.chunk_path)
        anomaly_df = pd.read_csv(self.anomaly_path)

        if len(anomaly_df) > self.total_rows:
            raise ValueError("Anomaly data has more rows than the total requested.")

        # Select only enough rows from chunk to reach the total size
        chunk_sample = chunk_df.sample(n=self.total_rows - len(anomaly_df), random_state=42)

        # Combine and shuffle
        merged_df = pd.concat([chunk_sample, anomaly_df], ignore_index=True).sample(frac=1, random_state=42)

        # Write to gzip-compressed text file
        with gzip.open(self.output_path, 'wt', encoding='utf-8') as f_out:
            merged_df.to_csv(f_out, index=False)

        print(f"✅ Merged file saved to: {self.output_path}")


# Example usage:
if __name__ == "__main__":
    merger = AnomalyMerger(
        chunk_path="src/utils/chunk_000.csv",
        anomaly_path="src/utils/Mock_Anomaly_Sample.csv",
        output_path="src/utils/merged_100.txt.gz"
    )
    merger.merge()
