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

## Current plan
A) **ORIGINAL MAMBA**
1. ~~Implement detector with original mamba~~
2. Train/finetune & metrics
3. Test in deployment locally
4. Test in deployment on edge

B) **MAMBA-VISION**
1.  ~~Add just VisionMamba blocks~~
2. Combine with attention (one of the core ideas behind the vision mamba block)
3. Combine into original VisionMamba block
4. Train/finetune & metrics
5. Test in deployment locally
6. Test in deployment on edge

C) **CHANNEL/SPATIAL ATTENTION**

D) **VECTOR FIELDS**

E) **MoE & gating network (to handle different domains)**