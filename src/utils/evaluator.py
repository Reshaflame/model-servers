import torch
import numpy as np
import os
from torch.utils.data import DataLoader

def evaluate_and_export(model, dataset, model_name, device="cpu", batch_size=256):
    """
    Runs inference on the entire dataset and saves predictions to /app/models/preds/.
    """
    model.eval()
    y_pred = []
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size)
        for batch_features, _ in loader:
            batch_features = batch_features.to(device)
            # Handle models expecting sequence dim (like GRU/LSTM)
            if len(batch_features.shape) == 2:
                batch_features = batch_features.unsqueeze(1)
            outputs = model(batch_features)
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            y_pred.extend(preds)

    # Save predictions
    output_dir = "/app/models/preds"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{model_name}_preds.npy"), np.array(y_pred))
    print(f"[Evaluator] ✅ Saved {model_name} predictions to {output_dir}/{model_name}_preds.npy")
