import torch
import os

def export_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Model Exporter] ✅ Model exported to {path}")
    
    # Auto-notify location for easy Runpod access
    print(f"[Model Exporter] 📢 You can now download it via Runpod UI from inside '/app/models/'.")
