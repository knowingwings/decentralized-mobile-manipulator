# check_gpu.py
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"Has CUDA: {torch.cuda.is_available()}")
print(f"Has HIP/ROCm: {hasattr(torch, 'hip') and hasattr(torch.hip, 'is_available') and torch.hip.is_available()}")

if hasattr(torch, 'hip') and hasattr(torch.hip, 'is_available') and torch.hip.is_available():
    print(f"AMD GPU detected: {torch.hip.get_device_name(0)}")
    print(f"Number of GPUs: {torch.hip.device_count()}")
elif torch.cuda.is_available():
    print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("No GPU detected")

# Check environment variables
import os
for var in ['ROCM_PATH', 'HIP_PATH', 'PYTORCH_ROCM_ARCH']:
    print(f"{var}: {os.environ.get(var, 'Not set')}")