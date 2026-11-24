"""Quick check that PyTorch can use the NVIDIA GPU."""
import torch

print("PyTorch:", torch.__version__, "| built for CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    print("\nNO GPU FOUND. Things to try:")
    print(" 1. Update the NVIDIA driver (GeForce Game Ready/Studio driver, version 570 or newer for the RTX 50 series).")
    print(" 2. Re-run setup_windows.bat so the CUDA build of PyTorch is installed (not the CPU one).")
    raise SystemExit(1)

p = torch.cuda.get_device_properties(0)
print("GPU:", p.name, "| memory:", round(p.total_memory / 1e9, 1), "GB | compute capability:", f"{p.major}.{p.minor}")
print("bf16 supported:", torch.cuda.is_bf16_supported())
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); y = (x @ x).float().abs().mean().item(); torch.cuda.synchronize()
print("test calculation on the GPU worked. The GPU is ready.")
