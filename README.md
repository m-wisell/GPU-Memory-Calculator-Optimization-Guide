# GPU Memory Calculator & Optimization Guide

A practical reference for training PyTorch vision models without running out of GPU memory. Written for people who are new to GPU training (no systems background assumed).

**[→ Live demo on GitHub Pages](https://m-wisell.github.io/GPU-Memory-Calculator-Optimization-Guide/)**

---

## What's included

### Memory Calculator
Estimate peak VRAM before you run an experiment. 23 model presets including:

- **Classification:** MobileViT-XXS/S, EfficientNet-B1, ResNet-50, Swin-T/B, ConvNeXt-B, ViT-Base/Large
- **Segmentation:** SAM ViT-B/L/H, SAM2-Tiny/Small/Base+/Large, Mask2Former, SegFormer-B5
- **Foundation models:** DINOv2 ViT-S/B/L, CLIP ViT-B/L

The calculator uses architecture-aware spatial reduction factors (CNN=20, Hybrid=10, Transformer=50) to produce realistic estimates rather than naive worst-case numbers. Batch size configurable up to 2048.

### Optimization Guide
A beginner-friendly reference covering:

- **Start here** — five changes that fix most GPU problems (AMP, DataLoader, no_grad, .item(), memory monitoring)
- What GPU memory actually is and why resolution is quadratic
- Mixed-precision training (AMP) and loss scaling
- Gradient accumulation, DataLoader tuning, CUDA prefetching
- Gradient checkpointing, frozen backbones, torch.compile()
- Classification vs. segmentation specifics (including SAM/SAM2)
- OOM troubleshooting with a prioritized fix order

---

## Deploying to GitHub Pages

1. Fork or clone this repo
2. Go to **Settings → Pages** in your GitHub repo
3. Under **Source**, select `main` branch, `/ (root)` folder
4. Click **Save**
5. Your site will be live at `https://YOUR-USERNAME.github.io/YOUR-REPO-NAME/`
6. Update the link in this README and in `GPU_Memory_Calculator.html`'s nav GitHub link (`id="gh-link"`)

No build step required. The entire site is a single `GPU_Memory_Calculator.html` file.

---

## Files

```
GPU_Memory_Calculator.html  # The entire site — calculator + guide in one file
GPU Optimization Guide.pdf  # Formatted Word document version (downloadable)
README.md                   # This file
```

---

## Formulas used

Activation memory (the dominant cost):

```
M_acts = B × (H×W ÷ sf) × C_avg × L × bytes × 1.5
```

Where:
- `B` = batch size
- `H×W` = input resolution
- `sf` = spatial reduction factor (CNN: 20, Hybrid: 10, Transformer: 50)
- `C_avg` = average channels per layer
- `L` = number of layers
- `bytes` = 2 for AMP/FP16, 4 for FP32
- `1.5` = overhead multiplier for autograd graph and intermediate buffers

The spatial factor corrects for the fact that CNNs progressively downsample (ResNet-50 goes 224→7px), so the average spatial footprint per layer is much smaller than the input resolution.

---

## Notes on accuracy

The calculator produces estimates, not guarantees. Actual VRAM usage varies with:
- PyTorch version and CUDA version
- Specific model implementation details
- Whether you're in training vs. inference mode
- Fragmentation and caching behavior

Treat the output as a planning tool — add a 15–20% safety margin for production runs.

---

## Contributing

Pull requests welcome. If you have calibrated measurements for a specific model that differ significantly from the estimates here, please open an issue with the model name, configuration, and measured peak VRAM.
