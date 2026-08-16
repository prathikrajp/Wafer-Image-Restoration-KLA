"""
Train a denoising + 2x super-resolution model on the competition dataset.

Expected structure:
    semicon/
      train/
        GT/          <- 256x256 float32 .npy, clean, range [0,1]
        NoisyLR/     <- 128x128 float32 .npy, noisy/blurry, range can exceed 1.0
      Test_NoisyLR/  <- 128x128 float32 .npy, blind test set (no GT)

Task: given a 128x128 noisy/blurry input, predict the 256x256 clean image.

Run:
    python train_sr.py --data_dir . --epochs 40
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------- Dataset -----------------------------

class SRDataset(Dataset):
    """Loads paired (NoisyLR 128x128, GT 256x256) .npy files by matching filename."""

    def __init__(self, root: str, split_dir: str = "train"):
        self.gt_dir = Path(root) / split_dir / "GT"
        self.lr_dir = Path(root) / split_dir / "NoisyLR"
        self.filenames = sorted(f.name for f in self.gt_dir.glob("*.npy"))
        assert len(self.filenames) > 0, f"No .npy files found in {self.gt_dir}"

        lr_names = set(f.name for f in self.lr_dir.glob("*.npy"))
        missing = [f for f in self.filenames if f not in lr_names]
        assert not missing, f"{len(missing)} GT files have no matching NoisyLR file, e.g. {missing[:3]}"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        gt = np.load(self.gt_dir / fname).astype(np.float32)          # (256,256), [0,1]
        lr = np.load(self.lr_dir / fname).astype(np.float32)          # (128,128), can exceed 1.0

        lr = np.clip(lr, 0.0, 1.0)  # clip rare bright-noise spikes so input stays in a stable range
        gt = np.clip(gt, 0.0, 1.0)

        lr_t = torch.from_numpy(lr).unsqueeze(0)  # (1,128,128)
        gt_t = torch.from_numpy(gt).unsqueeze(0)  # (1,256,256)
        return lr_t, gt_t


class TestNoisyDataset(Dataset):
    """For the blind test set — NoisyLR only, no GT. Used later for generating final predictions."""

    def __init__(self, root: str, split_dir: str = "Test_NoisyLR"):
        self.lr_dir = Path(root) / split_dir
        self.filenames = sorted(f.name for f in self.lr_dir.glob("*.npy"))
        assert len(self.filenames) > 0, f"No .npy files found in {self.lr_dir}"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr = np.load(self.lr_dir / fname).astype(np.float32)
        lr = np.clip(lr, 0.0, 1.0)
        lr_t = torch.from_numpy(lr).unsqueeze(0)
        return lr_t, fname


# ----------------------------- Model -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PixelShuffleUpsample(nn.Module):
    """2x upsample using PixelShuffle (sub-pixel convolution) - generally cleaner
    than plain ConvTranspose2d for super-resolution, avoids checkerboard artifacts."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class SRUNet(nn.Module):
    """U-Net encoder/decoder at 128x128, then a PixelShuffle head that upsamples
    the final feature map to 256x256 to match the GT resolution."""

    def __init__(self, base_ch=32):
        super().__init__()
        # Encoder (operates at 128x128 input)
        self.enc1 = ConvBlock(1, base_ch)            # 128
        self.enc2 = ConvBlock(base_ch, base_ch*2)      # 64
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)    # 32
        self.enc4 = ConvBlock(base_ch*4, base_ch*8)    # 16

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch*8, base_ch*16)  # 8

        # Decoder back up to 128x128
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch*16, base_ch*8)

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)

        # Super-resolution head: 128x128 -> 256x256
        self.sr_up = PixelShuffleUpsample(base_ch, base_ch)
        self.sr_refine = ConvBlock(base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))   # (base_ch, 128, 128)

        sr = self.sr_up(d1)                          # (base_ch, 256, 256)
        sr = self.sr_refine(sr)
        out = torch.sigmoid(self.out_conv(sr))        # (1, 256, 256), range [0,1]
        return out


# ----------------------------- Metrics -----------------------------

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ----------------------------- Train loop -----------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    full_ds = SRDataset(args.data_dir, args.train_split_dir)
    n_val = int(len(full_ds) * args.val_frac)
    n_train = len(full_ds) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=generator)
    print(f"Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SRUNet(base_ch=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for lr_img, gt_img in train_loader:
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)
            optimizer.zero_grad()
            pred = model(lr_img)
            loss = F.l1_loss(pred, gt_img)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * lr_img.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img, gt_img = lr_img.to(device), gt_img.to(device)
                pred = model(lr_img)
                loss = F.l1_loss(pred, gt_img)
                val_loss += loss.item() * lr_img.size(0)
                val_psnr += psnr(pred, gt_img).item() * lr_img.size(0)
        val_loss /= len(val_ds)
        val_psnr /= len(val_ds)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_psnr={val_psnr:.2f} dB")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Path(args.checkpoint_dir) / "best_model.pth")
            print(f"  -> saved new best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"No improvement for {args.early_stop_patience} epochs — stopping early.")
                break

    print(f"Training done. Best val_loss={best_val_loss:.4f}. "
          f"Checkpoint saved at {args.checkpoint_dir}/best_model.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".",
                         help="Folder containing train/GT and train/NoisyLR")
    parser.add_argument("--train_split_dir", type=str, default="train",
                         help="Subfolder name containing GT/ and NoisyLR/")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()