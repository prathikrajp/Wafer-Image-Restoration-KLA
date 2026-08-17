"""
Inference script: Runs the trained SRUNet on the blind Test_NoisyLR dataset (.npy files)
and saves the restored 256x256 outputs.
"""

import os
import glob
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Import the exact architecture from your train.py
from train import SRUNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running inference on: {device}")

    # 1. Load Model
    model = SRUNet(base_ch=32).to(device)
    checkpoint_path = "./checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: {checkpoint_path} not found. Please place your best_model.pth in checkpoints/")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Output Folders
    test_dir = Path(r"C:\Users\prath\Documents\CIT\Projects\Wafer-Image-Restoration-KLA\semicon\Test_NoisyLR\NoisyLR")
    out_npy_dir = Path("./Restored_Outputs/npy")
    out_png_dir = Path("./Restored_Outputs/png")
    os.makedirs(out_npy_dir, exist_ok=True)
    os.makedirs(out_png_dir, exist_ok=True)

    test_files = sorted(test_dir.rglob("*.npy"))
    print(f"[*] Found {len(test_files)} test files to restore.")

    # 3. Process Each Image
    with torch.no_grad():
        for fpath in test_files:
            # Load and prepare input
            lr = np.load(fpath).astype(np.float32)
            lr = np.clip(lr, 0.0, 1.0)
            lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 128, 128)

            # Predict restored 256x256 image
            pred = model(lr_t).squeeze().cpu().numpy()  # (256, 256)
            pred = np.clip(pred, 0.0, 1.0)

            # Save as .npy (raw output)
            np.save(out_npy_dir / fpath.name, pred)

            # Save as .png (for presentation/viewing)
            img_uint8 = (pred * 255.0).astype(np.uint8)
            Image.fromarray(img_uint8).save(out_png_dir / fpath.with_suffix(".png").name)

    print(f"[*] Done! Restored images saved to {out_npy_dir} and {out_png_dir}")

if __name__ == "__main__":
    main()