import os
import sys

# ─── Thread & Ray limits ──────────────────────────────────────────
os.environ["OMP_NUM_THREADS"]           = "1"
os.environ["OPENBLAS_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]           = "1"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
os.environ["RAY_use_popen_spawn_worker"]     = "1"
os.environ["RAY_object_spilling_config"]     = (
    '{"type":"filesystem","params":{"directory_path":"/tmp"}}'
)

# ─── Pipelines ────────────────────────────────────────────────────
from pipeline.iso_pipeline   import run_iso_pipeline
from pipeline.gru_pipeline   import run_gru_pipeline
from pipeline.lstm_pipeline  import run_lstm_pipeline
from pipeline.ensemble_pipeline import run_ensemble_training
# (LightGBM pipeline can be added later: from pipeline.lgbm_pipeline import …)

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
def chunks_exist(path: str) -> bool:
    """Return True if the directory contains at least one *.csv chunk."""
    return os.path.isdir(path) and any(p.endswith(".csv") for p in os.listdir(path))

# -----------------------------------------------------------------
# Pre-processing wrappers (updated paths)
# -----------------------------------------------------------------
LABELED_DIR   = "data/chunks_labeled"
UNLABELED_DIR = "data/chunks_unlabeled"

def preprocess_labeled():
    if chunks_exist(LABELED_DIR):
        print("[Preprocess] ✅ Labeled data already preprocessed. Skipping.")
    else:
        print("[Preprocess] 🔄 Running labeled data preprocessing...")
        os.system(f"cd src && {sys.executable} -m preprocess.labeledPreprocess")

# def preprocess_unlabeled():
#     if chunks_exist(UNLABELED_DIR):
#         print("[Preprocess] ✅ Unlabeled data already preprocessed. Skipping.")
#     else:
#         print("[Preprocess] 🔄 Running unlabeled data preprocessing...")
#         os.system("python src/preprocess/unlabeledPreprocess.py") # default out_dir is data/chunks_unlabeled

# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
MENU = """
📊 TrueDetect CLI – Select an option:
1. Preprocess Labeled Data
2. Preprocess Unlabeled Data (deprecated)
3. Train GRU
4. Train LSTM+RNN
5. Evaluate Isolation Forest
6. Train Ensemble Voting
0. Exit
"""

def main():
    while True:
        print(MENU)
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            preprocess_labeled()
        elif choice == "3":
            run_gru_pipeline(preprocess=False)
        elif choice == "4":
            run_lstm_pipeline(preprocess=False)
        elif choice == "5":
            run_iso_pipeline(preprocess=False)
        elif choice == "6":
            run_ensemble_training()
        elif choice == "0":
            print("👋 Exiting. Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()
