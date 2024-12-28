import torch
import pandas as pd


class GPUUtils:
    @staticmethod
    def get_device():
        """
        Get the appropriate device (GPU if available, else CPU).
        """
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            print("Using CPU.")
            return torch.device("cpu")

    @staticmethod
    def to_tensor(data, device):
        """
        Convert data to a PyTorch tensor and move to the specified device.
        Args:
            data: Input data (e.g., NumPy array or list).
            device: Target device (CPU or GPU).
        """
        # Convert data to numeric, forcing errors to NaN
        data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN values with 0 (if this is acceptable for your use case)
        data = data.fillna(0)
        
        # Convert to a PyTorch tensor
        return torch.tensor(data.values, device=device)

