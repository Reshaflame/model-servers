# checked
import torch
import numpy as np
import os
from utils.metrics import Metrics


def quick_f1(model, val_loader, device, thresh_list=(0.5, 0.3, 0.1)):
    m = Metrics()
    y_true, logits = [], []
    with torch.no_grad():
        for x, y in val_loader():
            x, y = x.to(device), y.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            logits.append(torch.sigmoid(model(x)).cpu())
            y_true.append(y.cpu())
    y_true  = torch.cat(y_true).numpy().flatten()
    logits  = torch.cat(logits).numpy().flatten()

    best = {"F1":0}
    for t in thresh_list:
        preds = (logits > t).astype(int)
        cur   = m.compute_standard_metrics(y_true, preds)
        if cur["F1"] > best["F1"]:
            best = cur | {"th":t}
    return best


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
