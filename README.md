# Hot-Peppers-Company Computer Vision

Minimal training pipeline for MMFW-UAV sequence detection.

## Quickstart

1) (Optional) Create a local sample subset from a raw copy:
```bash
python scripts/download_sample.py \
  --raw_dir data/MMFW-UAV/raw \
  --output_dir data/MMFW-UAV/sample \
  --clip_length 10 \
  --limit_per_type 200
```

2) Create train/val/test splits for the data root you will train on:
```bash
python scripts/prepare_data.py --data_root data/MMFW-UAV/sample
```
This writes `data/MMFW-UAV/sample/splits/*.json`.

3) Train:
```bash
python train.py --data_root data/MMFW-UAV/sample --epochs 10
```

## Project layout (simple)
- `data.py` - sequence dataset + transforms (consistent augmentations per sequence)
- `model.py` - backbone + Mamba/LSTM temporal model + detection head
- `train.py` - minimal training loop
- `scripts/prepare_data.py` - creates splits (by part if present, else by UAV type)
- `scripts/download_sample.py` - builds a contiguous sample subset

## Notes
- Splits live under the selected data root (e.g., `data/MMFW-UAV/raw/splits`).
- If your dataset is organized by parts, `prepare_data.py` splits by part to avoid leakage.
- Legacy notebooks and Lightning modules remain under `mamba/` but are no longer required.

# Current plan

**A. Baseline**

1. Select SOTA Backbone
2. Detection Head Swap: replace heavy Cascade Mask R-CNN with a lightweight YOLOv11 or YOLOv12 detection head for edge efficiency.
3. Do pipelines for training/tuning and benchmarking (FPS, mAP, and Frames-per-Watt on MS COCO, BDD100K day/night, ACDC adverse weather)

**B. MoE & LoRA Adaptation (Andrii)**

1. Determine where feature variance is highest.
2. Implement rank 7-8 LoRA adapters there
3. Train LoRA A for Day conditions (BDD100K day images)
4. Train LoRA B for Low-Light/Night (BDD100K night images)
5. Train LoRA C for Adversarial Noise/Weather (ACDC adverse weather)
6. Lightweight gating network (small CNN or MLP) to select the active LoRA in <20ms based on input frame statistics. Img Hashing??
7. Optimize the switching logic to keep adapters in shared memory, preventing I/O delays during domain shifts.

**C. Low-Level Calibration (Ivan)**

1. Experiment with making Δ (step size) and the A matrix dynamic during inference to trade off temporal resolution for speed.
2. Identify if the bottleneck is the Selective Scan (S6) kernel or the linear projections on the hardware.
3. Apply INT8 quantization to the CNN/MLP stages while maintaining FP16 for the SSM blocks to preserve precision.

**D. Evaluation & Contribution Framing**

1. Test "Standard Mamba-Vision" vs. "MoE-LoRA Mamba-Vision" on "broken" datasets (noisy/dark).
2. Measure total power consumption and latency (ms) for the gating + backbone + head pipeline.
3. To be continued ...