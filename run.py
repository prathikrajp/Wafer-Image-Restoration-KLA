"""
Official run.py for automated grading pipeline.
Reads degraded .npy files from an input directory and saves restored .npy files to an output directory.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

# Import the architecture from your train.py
from train import SRUNet

def main():
    # 1. Accept command line arguments for input and output directories
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
        
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # Create the output directory if it does not already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running inference on: {device}")

    # 2. Load Model from the 'models/' directory
    model = SRUNet(base_ch=32).to(device)
    checkpoint_path = "./models/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: {checkpoint_path} not found.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 3. Read all .npy files from the input directory
    test_files = sorted(input_dir.glob("*.npy"))
    print(f"[*] Found {len(test_files)} test files to restore.")

    # 4. Process Each Image
    with torch.no_grad():
        for fpath in test_files:
            # Load and prepare input
            lr = np.load(fpath).astype(np.float32)
            lr = np.clip(lr, 0.0, 1.0)
            lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device) 

            # Predict restored 256x256 image
            pred = model(lr_t).squeeze().cpu().numpy()
            
            # Ensure output values are strictly within [0,1] and contain no NaN or Inf values
            pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
            pred = np.clip(pred, 0.0, 1.0)

            # 5. Generate one restored .npy file for every input file with the exact same filename
            np.save(output_dir / fpath.name, pred)

    print(f"[*] Done! Restored .npy images saved to {output_dir}")

if __name__ == "__main__":
    main()