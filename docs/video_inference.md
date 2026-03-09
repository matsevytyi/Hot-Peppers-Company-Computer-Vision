# Video Inference Guide

This document explains how to run video inference with the new pipeline CLI.

## Prerequisites

1. Install project dependencies:
```bash
pip install -r requirements.txt
pip install -e MambaVisionReengineering
```

2. Make sure model checkpoints referenced by your eval config exist.
   Default config:
   - `configs/eval/shared_eval.yaml`

3. Ensure runtime support for Mamba models (CUDA + `mamba_ssm`):
```bash
python scripts/pipelines/preflight_check.py
```

## Main Command

Run inference with a model entry from eval config:

```bash
python scripts/pipelines/infer_video.py \
  --config configs/eval/shared_eval.yaml \
  --model-name moe_full_router \
  --video-input /absolute/or/relative/path/input.mp4 \
  --video-output results/inference/output.mp4
```

The script reads frames, runs detection, draws boxes + labels + scores, and writes an annotated output video.
For MoE models, router top-domain and confidence are also shown on the frame.

## Common Model Names

From `configs/eval/shared_eval.yaml`:
- `base_coco`
- `lora_day`
- `lora_night`
- `lora_acdc`
- `moe_full_router`

## Useful Optional Flags

- `--classes-config configs/classes/common_8.yaml`
- `--device cuda`
- `--image-size 640`
- `--conf-threshold 0.25`
- `--nms-iou 0.5`
- `--max-frames 300`

## Quick Check

Print CLI help:

```bash
python scripts/pipelines/infer_video.py --help
```

## Troubleshooting

- `Model '<name>' not found in eval.models`:
  Check `--model-name` against `configs/eval/shared_eval.yaml`.

- `Could not open video`:
  Verify input path and file readability.

- `Could not create output video`:
  Check output directory permissions and codec support.

- `Mamba runtime is not supported`:
  Use Linux + NVIDIA CUDA runtime with required Mamba packages.
