import torch
import numpy as np
import os

def evaluate_and_export(model, dataset, model_name, device="cpu", export_ground_truth=False):
    model.eval()

    output_dir = "/app/models/preds"
    os.makedirs(output_dir, exist_ok=True)

    preds_path = os.path.join(output_dir, f"{model_name}_preds.npy")
    y_true_path = os.path.join(output_dir, "y_true.npy")

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch_idx, (batch_features, batch_labels) in enumerate(dataset):
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

            preds_list.append(preds)
            labels_list.append(labels)

            # Optional: Periodically print to avoid feeling "stuck"
            if batch_idx % 500 == 0:
                print(f"Processed {batch_idx} batches...")

    # ✅ After all batches — concat and save ONCE
    preds_all = np.concatenate(preds_list)
    np.save(preds_path, preds_all)

    if export_ground_truth:
        labels_all = np.concatenate(labels_list)
        np.save(y_true_path, labels_all)

    print(f"[Evaluator] ✅ Saved all predictions to {preds_path}")
    if export_ground_truth:
        print(f"[Evaluator] ✅ Saved ground truth to {y_true_path}")
