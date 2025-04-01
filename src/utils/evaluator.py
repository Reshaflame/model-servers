import torch
import numpy as np
import os

# /src/utils/evaluator.py
def evaluate_and_export(model, dataset, model_name, device="cpu", export_ground_truth=False):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_features, batch_labels in dataset:  # ← dataset is already iterable
            if not torch.is_tensor(batch_features):
                batch_features = torch.tensor(batch_features)
            if not torch.is_tensor(batch_labels):
                batch_labels = torch.tensor(batch_labels)
            
            # ✅ Always enforce float32 dtype
            batch_features = batch_features.to(dtype=torch.float32, device=device)
            batch_labels = batch_labels.to(dtype=torch.float32, device=device)

            if len(batch_features.shape) == 2:
                batch_features = batch_features.unsqueeze(1)

            outputs = model(batch_features)
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            y_pred.extend(preds)
            y_true.extend(batch_labels.cpu().numpy().flatten())

    output_dir = "/app/models/preds"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{model_name}_preds.npy"), np.array(y_pred))
    print(f"[Evaluator] ✅ Saved {model_name} predictions to {output_dir}/{model_name}_preds.npy")

    if export_ground_truth:
        np.save(os.path.join(output_dir, f"y_true.npy"), np.array(y_true))
        print(f"[Evaluator] ✅ Saved y_true to {output_dir}/y_true.npy")
