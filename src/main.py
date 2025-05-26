import os
import sys

# Limit threads and processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
os.environ["RAY_use_popen_spawn_worker"] = "1"
os.environ["RAY_object_spilling_config"] = '{"type":"filesystem","params":{"directory_path":"/tmp"}}'

from pipeline.iso_pipeline import run_iso_pipeline
from pipeline.gru_pipeline import run_gru_pipeline
from pipeline.lstm_pipeline import run_lstm_pipeline
from pipeline.ensemble_pipeline import run_ensemble_training

def chunks_exist(path: str) -> bool:
    return os.path.exists(path) and any(fname.endswith(".csv") for fname in os.listdir(path))

def preprocess_labeled():
    if chunks_exist("data/labeled_data/chunks"):
        print("[Preprocess] ✅ Labeled data already preprocessed. Skipping.")
    else:
        print("[Preprocess] 🔄 Running labeled data preprocessing...")
        os.system("python src/preprocess/labeledPreprocess.py")

def preprocess_unlabeled():
    if chunks_exist("data/preprocessed_unlabeled/chunks"):
        print("[Preprocess] ✅ Unlabeled data already preprocessed. Skipping.")
    else:
        print("[Preprocess] 🔄 Running unlabeled data preprocessing...")
        os.system("python src/preprocess/unlabeledPreprocess.py")

def show_menu():
    print("\n📊 TrueDetect CLI - Select an option:")
    print("1. Preprocess Labeled Data")
    print("2. Preprocess Unlabeled Data")
    print("3. Train GRU")
    print("4. Train LSTM+RNN")
    print("6. Evaluate Isolation Forest")
    print("7. Train Ensemble Voting")
    print("0. Exit")

def main():
    while True:
        show_menu()
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            preprocess_labeled()
        elif choice == '2':
            preprocess_unlabeled()
        elif choice == '3':
            run_gru_pipeline(preprocess=False)
        elif choice == '4':
            run_lstm_pipeline(preprocess=False)
        elif choice == '6':
            run_iso_pipeline(preprocess=False)
        elif choice == '7':
            run_ensemble_training()
        elif choice == '0':
            print("👋 Exiting. Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()
