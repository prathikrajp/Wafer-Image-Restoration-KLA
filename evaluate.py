"""
Task 1.2 + 1.3 — Evaluate the trained model on the validation split.

Computes:
    - Average PSNR, SSIM, LPIPS on the validation set
    - Per-image inference latency (ms)
    - Model checkpoint file size (MB)

Requires (install once):
    pip install scikit-image lpips

Run:
    python evaluate.py --data_dir . --checkpoint checkpoints/best_model.pth
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn

from train import SRUNet, SRDataset


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def compute_ssim(pred_np, target_np):
    # pred_np, target_np: (H, W) numpy arrays in [0,1]
    return ssim_fn(target_np, pred_np, data_range=1.0)


def try_load_lpips():
    try:
        import lpips
    except ImportError:
        print("WARNING: lpips not installed (pip install lpips) — skipping LPIPS metric.")
        return None
    try:
        loss_fn = lpips.LPIPS(net="alex")
        return loss_fn
    except Exception as e:
        print(f"WARNING: lpips model failed to load ({e}) — skipping LPIPS metric. "
              "This usually means the pretrained AlexNet weights couldn't be downloaded; "
              "check your internet connection and try again.")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--train_split_dir", type=str, default="train")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1,
                         help="Use 1 for accurate per-image latency measurement")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load model ----
    model = SRUNet(base_ch=args.base_ch).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    ckpt_size_mb = Path(args.checkpoint).stat().st_size / (1024 * 1024)
    n_params = sum(p.numel() for p in model.parameters())

    # ---- Rebuild the same val split used during training (same seed) ----
    full_ds = SRDataset(args.data_dir, args.train_split_dir)
    n_val = int(len(full_ds) * args.val_frac)
    n_train = len(full_ds) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    _, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Validation pairs: {len(val_ds)}")

    lpips_fn = try_load_lpips()
    if lpips_fn is not None:
        lpips_fn = lpips_fn.to(device)

    psnr_scores, ssim_scores, lpips_scores = [], [], []
    latencies_ms = []

    with torch.no_grad():
        for lr_img, gt_img in val_loader:
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred = model(lr_img)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000 / lr_img.size(0))  # per-image ms

            # PSNR (batch-safe via loop over batch dim)
            for i in range(pred.size(0)):
                p = pred[i:i+1]
                g = gt_img[i:i+1]
                psnr_scores.append(psnr(p, g))

                p_np = p.squeeze().cpu().numpy()
                g_np = g.squeeze().cpu().numpy()
                ssim_scores.append(compute_ssim(p_np, g_np))

                if lpips_fn is not None:
                    # lpips expects 3-channel, range [-1,1]
                    p_3ch = p.repeat(1, 3, 1, 1) * 2 - 1
                    g_3ch = g.repeat(1, 3, 1, 1) * 2 - 1
                    lp = lpips_fn(p_3ch, g_3ch).item()
                    lpips_scores.append(lp)

    print("\n" + "=" * 50)
    print("TASK 1.2 — QUANTITATIVE METRICS (validation set)")
    print("=" * 50)
    print(f"Avg PSNR : {np.mean(psnr_scores):.2f} dB")
    print(f"Avg SSIM : {np.mean(ssim_scores):.4f}")
    if lpips_scores:
        print(f"Avg LPIPS: {np.mean(lpips_scores):.4f}  (lower is better)")
    else:
        print("Avg LPIPS: skipped (lpips not installed)")

    print("\n" + "=" * 50)
    print("TASK 1.3 — LATENCY & FOOTPRINT")
    print("=" * 50)
    # Skip first few for warmup effects (GPU kernels compiling on first calls)
    warm_latencies = latencies_ms[3:] if len(latencies_ms) > 3 else latencies_ms
    print(f"Avg inference latency: {np.mean(warm_latencies):.2f} ms/image (after warmup)")
    print(f"Model checkpoint size: {ckpt_size_mb:.2f} MB")
    print(f"Total parameters     : {n_params:,}")


if __name__ == "__main__":
    main()