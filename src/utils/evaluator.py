# checked
import torch
import numpy as np
import os
from utils.metrics import Metrics


# --- utils/evaluator.py ---
def quick_f1(model, val_loader_fn, device, default_th=0.5):
    """
    Runs one pass over val_loader_fn() and returns a dict with
    Precision, Recall, F1, and th no matter what.
    """
    m = Metrics()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_loader_fn():
            x, y = x.to(device), y.to(device)
            if x.dim() == 2:  # GRU case
                x = x.unsqueeze(1)
            probs = torch.sigmoid(model(x)).flatten()
            preds = (probs > default_th).float()

            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    if len(set(y_true)) < 2:       # no positives or no negatives
        # Avoid sklearn warnings / errors
        precision = recall = f1 = 0.0
    else:
        metrics = m.compute_standard_metrics(y_true, y_pred)
        precision, recall, f1 = (
            metrics["Precision"],
            metrics["Recall"],
            metrics["F1"],
        )

    return {"Precision": precision, "Recall": recall, "F1": f1, "th": default_th}



def evaluate_and_export(model, dataset, model_name, device="cpu", export_ground_truth=False):
    print(f"[DEBUG] 🧠 Running evaluate_and_export() from src/utils/evaluator.py for model: {model_name}")
    model.eval()

    output_dir = f"/app/models/preds/{model_name}_batches"
    os.makedirs(output_dir, exist_ok=True)

    batch_id = 0
    print(f"[Evaluator] 🚀 Evaluating and streaming predictions for model: {model_name}")

    with torch.no_grad():
        for batch_features, batch_labels in dataset:
            batch_id += 1

            if not torch.is_tensor(batch_features):
                batch_features = torch.tensor(batch_features)
            if not torch.is_tensor(batch_labels):
                batch_labels = torch.tensor(batch_labels)

            batch_features = batch_features.to(dtype=torch.float32, device=device)
            batch_labels = batch_labels.to(dtype=torch.float32, device=device)

            if len(batch_features.shape) == 2:
                batch_features = batch_features.unsqueeze(1)

            outputs = model(batch_features)
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            labels = batch_labels.cpu().numpy().flatten()

            # Save predictions per batch
            np.save(os.path.join(output_dir, f"batch_{batch_id:05d}_preds.npy"), preds)
            if export_ground_truth:
                np.save(os.path.join(output_dir, f"batch_{batch_id:05d}_labels.npy"), labels)

            if batch_id % 1000 == 0:
                print(f"[Evaluator] 💾 Saved batch {batch_id}")

    print(f"[Evaluator] ✅ Done. Saved {batch_id} batches to: {output_dir}")
    if export_ground_truth:
        print(f"[Evaluator] 📌 Ground truth also saved per batch.")
