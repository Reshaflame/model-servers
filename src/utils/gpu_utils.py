import torch
import pandas as pd


class GPUUtils:
    @staticmethod
    def get_device(verbose=True):
        """
        Get the appropriate device (GPU if available, else CPU), with debug info.
        """
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if verbose:
                print(f"🚀 GPU available: {device_name}")
                print(f"🧠 Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                print(f"🧠 Memory Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
            return torch.device("cuda")
        else:
            if verbose:
                print("⚠️ GPU not available. Using CPU instead.")
            return torch.device("cpu")

    @staticmethod
    def to_tensor(data, device, dtype=torch.float32):
        """
        Convert data (e.g., NumPy or Pandas) to a PyTorch tensor and move to device.
        Automatically handles NaNs and type casting.
        """
        df = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').fillna(0)
        tensor = torch.tensor(df.values, dtype=dtype)
        return tensor.to(device)

    @staticmethod
    def show_gpu_stats():
        """
        Print current GPU memory usage if available.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🖥️  Device: {torch.cuda.get_device_name(device)}")
            print(f"📊 Allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
            print(f"📊 Reserved:  {torch.cuda.memory_reserved(device) / 1e6:.2f} MB")
        else:
            print("❌ No GPU available.")
