import torch
import numpy as np
import os

# /src/utils/evaluator.py
def evaluate_and_export(model, dataset, model_name, device="cpu", export_ground_truth=False):
    model.eval()

    preds_path = f"/app/models/preds/{model_name}_preds.npy"
    y_true_path = "/app/models/preds/y_true.npy"

    os.makedirs("/app/models/preds", exist_ok=True)

    first_batch = True

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
            labels = batch_labels.cpu().numpy().flatten()
            
            # 🟡 Save or append predictions
            if first_batch:
                np.save(preds_path, preds)
                if export_ground_truth:
                    np.save(y_true_path, labels)
                first_batch = False
            else:
                with open(preds_path, 'ab') as f:
                    np.save(f, preds)
                if export_ground_truth:
                    with open(y_true_path, 'ab') as f:
                        np.save(f, labels)

        
    print(f"[Evaluator] ✅ Predictions saved to {preds_path}")
    
    if export_ground_truth:
        print(f"[Evaluator] ✅ Ground truth saved to {y_true_path}")
