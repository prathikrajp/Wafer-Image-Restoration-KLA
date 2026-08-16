# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**SEMICON India Hackathon 2026 - KLA Challenge Submission**

## 📌 Project Overview
Microscopic semiconductor inspection images are vulnerable to optics-induced Gaussian blur, speckle noise, and downsampling resolution loss. These artifacts mask sub-micron defects, reducing fab yield and slowing automated defect classification (ADC). 

This project implements an end-to-end deep learning framework performing simultaneous joint deblurring, despeckling, and 2x super-resolution (128x128 -> 256x256) to restore semiconductor wafer images.

## 🚀 Model Architecture
Our solution uses a modified **U-Net Architecture** featuring:
* **Skip Connections:** To preserve high-frequency edge boundaries (crucial for circuit trace lines).
* **Sub-pixel Convolution (PixelShuffle):** For highly efficient 2x upscaling without checkerboard artifacts.
* **Loss Function:** Optimized using `L1` loss to maintain sharp edge geometries, augmented by structural similarity awareness.

## 📊 Performance Metrics
Evaluated on the official validation split:
* **PSNR:** 27.20 dB (Improved from ~18.40 dB baseline)
* **SSIM:** 0.72 
* **LPIPS:** 0.3759
* **Inference Latency:** 4.00 ms/image (~250 FPS on RTX 3060)
* **Model Footprint:** 29.91 MB (7.82 Million parameters)

## 💻 How to Run

**1. Install Dependencies**
```bash
pip install -r requirements.txt