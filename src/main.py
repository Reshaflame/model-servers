from pipeline.iso_pipeline import run_iso_pipeline
from pipeline.gru_pipeline import run_gru_pipeline
from pipeline.lstm_pipeline import run_lstm_pipeline
from pipeline.tst_pipeline import run_tst_pipeline


if __name__ == "__main__":
    print("Select a model to run:")
    print("1. Isolation Forest")
    print("2. GRU")
    print("3. LSTM+RNN Hybrid")
    print("4. Transformer")
    choice = input("Enter the number of the model you want to run: ")

    if choice == '1':
        run_iso_pipeline(preprocess=False)
    elif choice == '2':
        run_gru_pipeline(preprocess=False)
    elif choice == '3':
        run_lstm_pipeline(preprocess=False)
    elif choice == '4':
        run_tst_pipeline(preprocess=False)
    else:
        print("Invalid choice. Please select a valid model number.")
