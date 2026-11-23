"""
Run this before starting the inference server to confirm torch can see your
local GPU and to get a suggested RF_MODEL_CHUNK_SIZE for your VRAM.

    python scripts/check_gpu.py
"""
import torch

def main():
    print(f"torch version:     {torch.__version__}")
    print(f"CUDA available:    {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo CUDA GPU detected. The server will fall back to CPU,")
        print("which works but is roughly 10-30x slower per generation.")
        print("Check: `nvidia-smi` runs, GPU drivers installed, and torch")
        print("was installed with a CUDA build (see requirements.txt).")
        return

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total_vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)

    print(f"GPU:                {name}")
    print(f"Total VRAM:         {total_vram_gb:.1f} GB")

    if total_vram_gb < 5:
        suggestion = 2048
        note = "Tight — also consider RF_DEVICE=cpu fallback for 'high' quality jobs."
    elif total_vram_gb < 7:
        suggestion = 4096
        note = "Should comfortably run 'low'/'medium' quality; 'high' may be slow."
    elif total_vram_gb < 10:
        suggestion = 8192
        note = "Default chunk size should work fine at all quality levels."
    else:
        suggestion = 8192
        note = "Plenty of headroom — default settings are fine."

    print(f"\nSuggested RF_MODEL_CHUNK_SIZE = {suggestion}")
    print(note)
    print("\nSet this in your .env, then run:")
    print("    uvicorn app.main:app --host 0.0.0.0 --port 8001")

if __name__ == "__main__":
    main()
