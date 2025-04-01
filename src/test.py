from utils.gpu_utils import GPUUtils
import torch

if __name__ == "__main__":
    print("\n🔍 Checking GPU availability...\n")
    device = GPUUtils.get_device()

    print("\n📊 Showing current GPU memory stats...\n")
    GPUUtils.show_gpu_stats()

    print("\n✅ Creating dummy tensor on device to verify computation...\n")
    x = torch.randn(1000, 1000).to(device)
    y = x @ x.T
    print(f"Tensor shape: {y.shape} | Device: {y.device}")

    print("\n📊 GPU memory after tensor operation:\n")
    GPUUtils.show_gpu_stats()

    print("\n🚀 GPU check complete.")
