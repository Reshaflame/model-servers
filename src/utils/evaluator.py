import torch
import numpy as np
import os
from torch.utils.data import DataLoader

# /src/utils/evaluator.py
def evaluate_and_export(model, dataset, model_name, device="cpu", batch_size=256, export_ground_truth=False):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size)
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            if len(batch_features.shape) == 2:
                batch_features = batch_features.unsqueeze(1)
            outputs = model(batch_features)
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            y_pred.extend(preds)
            y_true.extend(batch_labels.numpy().flatten())

    output_dir = "/app/models/preds"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{model_name}_preds.npy"), np.array(y_pred))
    print(f"[Evaluator] ✅ Saved {model_name} predictions to {output_dir}/{model_name}_preds.npy")

    # Export y_true ONCE
    if export_ground_truth:
        np.save(os.path.join(output_dir, f"y_true.npy"), np.array(y_true))
        print(f"[Evaluator] ✅ Saved y_true to {output_dir}/y_true.npy")
