from utils.tuning import RayTuner
from models.lstm_rnn import LSTM_RNN_Hybrid, train_model
from preprocess.labeledPreprocess import preprocess_labeled_data_with_matching_parallel
from utils.model_exporter import export_model
from utils.evaluator import evaluate_and_export
from utils.SequenceChunkedDataset import SequenceChunkedDataset
from ray import tune
import torch


def run_lstm_pipeline(preprocess=False):
    if preprocess:
        print("[Pipeline] Running preprocessing...")
        preprocess_labeled_data_with_matching_parallel(
            auth_file='data/auth.txt.gz',
            redteam_file='data/redteam.txt.gz'
        )
        print("[Pipeline] Preprocessing completed.")
    else:
        print("[Pipeline] Skipping preprocessing. Using existing labeled dataset.")

    # ✅ Load chunked dataset with sequence support
    chunk_dataset = SequenceChunkedDataset(
        chunk_dir='data/labeled_data/chunks',
        label_column='label',
        batch_size=256,
        shuffle_files=True,
        binary_labels=True,
        sequence_length=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    input_size = chunk_dataset.input_size

    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2])
    }

    def train_func(config):
        return train_model(
            config=config,
            train_loader=chunk_dataset.train_loader(),
            val_loader=chunk_dataset.val_loader(),
            input_size=input_size
        )

    tuner = RayTuner(train_func, param_space, num_samples=4, max_epochs=5)
    best_config = tuner.optimize()
    print(f"[Ray Tune] Best hyperparameters: {best_config}")

    model = LSTM_RNN_Hybrid(
        input_size=input_size,
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"]
    ).to(chunk_dataset.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])
    criterion = torch.nn.BCELoss()

    for epoch in range(5):
        model.train()
        for batch_features, batch_labels in chunk_dataset.train_loader():
            batch_features, batch_labels = batch_features.to(chunk_dataset.device), batch_labels.to(chunk_dataset.device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    export_model(model, "/app/models/lstm_rnn_trained_model.pth")

    evaluate_and_export(
        model,
        chunk_dataset.full_loader(),
        model_name="lstm",
        device=chunk_dataset.device,
        export_ground_truth=True
    )
