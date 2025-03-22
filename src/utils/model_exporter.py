import torch
import os

def export_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Model Exporter] Exported model to {path}")
