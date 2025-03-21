import pandas as pd
import torch
import torch.nn as nn
from pipeline.iso_pipeline import run_iso_pipeline
from models.gru import GRUAnomalyDetector, prepare_dataset as prepare_gru_dataset, train_model as train_gru, evaluate_model as eval_gru
from models.lstm_rnn import LSTM_RNN_Hybrid, prepare_dataset as prepare_lstm_dataset, train_model as train_lstm, evaluate_model as eval_lstm
from models.transformer import TimeSeriesTransformer, prepare_dataset as prepare_transformer_dataset, train_transformer, evaluate_transformer
from utils.gpu_utils import GPUUtils


def run_gru():
    print("Running GRU...")
    device = GPUUtils.get_device()
    train_ds, test_ds = prepare_gru_dataset('data/labeled_data/labeled_auth_sample.csv')
    model = GRUAnomalyDetector(input_size=len(train_ds[0][0])).to(device)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

    train_gru(model, train_loader, nn.BCELoss(), torch.optim.Adam(model.parameters()), device)
    eval_gru(model, test_loader, device)


def run_lstm_rnn():
    print("Running LSTM+RNN Hybrid...")
    device = GPUUtils.get_device()
    train_ds, test_ds = prepare_lstm_dataset('data/labeled_data/labeled_auth_sample.csv')
    model = LSTM_RNN_Hybrid(input_size=train_ds[0][0].shape[1]).to(device)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

    train_lstm(model, train_loader, nn.BCELoss(), torch.optim.Adam(model.parameters()), device)
    eval_lstm(model, test_loader, device)


def run_transformer():
    print("Running Transformer...")
    device = GPUUtils.get_device()
    train_ds, test_ds = prepare_transformer_dataset('data/labeled_data/labeled_auth_sample.csv')
    model = TimeSeriesTransformer(input_size=train_ds[0][0].shape[1]).to(device)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

    train_transformer(model, train_loader, nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device)), torch.optim.Adam(model.parameters()), device)
    evaluate_transformer(model, test_loader, device)


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
        run_gru()
    elif choice == '3':
        run_lstm_rnn()
    elif choice == '4':
        run_transformer()
    else:
        print("Invalid choice. Please select a valid model number.")
