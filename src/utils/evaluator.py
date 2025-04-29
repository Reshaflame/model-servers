import torch
import numpy as np
import os

def evaluate_and_export(model, dataset, model_name, device="cpu", export_ground_truth=False):
    model.eval()

    output_dir = "/app/models/preds"
    os.makedirs(output_dir, exist_ok=True)

    preds_path = os.path.join(output_dir, f"{model_name}_preds.npy")
    y_true_path = os.path.join(output_dir, "y_true.npy")

    batch_buffer_preds = []
    batch_buffer_labels = []

    buffer_size = 1000  # 👈 flush every 1000 batches
    batch_count = 0

    first_write = True

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

            batch_buffer_preds.append(preds)
            batch_buffer_labels.append(labels)

            batch_count += 1

            # 🔥 Flush if buffer is big
            if batch_count % buffer_size == 0:
                preds_block = np.concatenate(batch_buffer_preds)
                labels_block = np.concatenate(batch_buffer_labels)

                if first_write:
                    np.save(preds_path, preds_block)
                    if export_ground_truth:
                        np.save(y_true_path, labels_block)
                    first_write = False
                else:
                    with open(preds_path, 'ab') as f:
                        np.save(f, preds_block)
                    if export_ground_truth:
                        with open(y_true_path, 'ab') as f:
                            np.save(f, labels_block)

                batch_buffer_preds.clear()
                batch_buffer_labels.clear()

                print(f"📝 Flushed {batch_count} batches... Memory safe!")

        # 🚀 Flush whatever small leftovers at end
        if batch_buffer_preds:
            preds_block = np.concatenate(batch_buffer_preds)
            labels_block = np.concatenate(batch_buffer_labels)

            if first_write:
                np.save(preds_path, preds_block)
                if export_ground_truth:
                    np.save(y_true_path, labels_block)
            else:
                with open(preds_path, 'ab') as f:
                    np.save(f, preds_block)
                if export_ground_truth:
                    with open(y_true_path, 'ab') as f:
                        np.save(f, labels_block)

        print(f"[Evaluator] ✅ Saved all predictions to {preds_path}")
        if export_ground_truth:
            print(f"[Evaluator] ✅ Saved ground truth to {y_true_path}")
