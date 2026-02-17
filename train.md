# Train Guide (Notebook Workflow + Dataset Download)

This document explains how to **download datasets correctly** and run the **full notebook workflow**.

Main rule: use `notebooks/*.ipynb` as the primary path, not CLI.

## 1) Environment setup

Run from the repository root: `Hot-Peppers-Company-Computer-Vision`.

```bash
pip install -r requirements.txt

python scripts/pipelines/preflight_check.py
```

If `preflight_check.py` reports `MISSING`, install missing packages and run it again.

## 2) How to download datasets correctly

Datasets are loaded/exported directly from the training notebooks through FiftyOne (`prepare_zoo_split_export`).

What happens automatically during train notebooks:

1. Load split from FiftyOne Zoo (`coco-2017`, `bdd100k`, `acdc`).
2. Normalize labels to the shared 8-class policy.
3. Export to COCO format under `data/exports/...`.
4. Generate manifests under `configs/manifests/*.json`.

Required manifests after running notebooks:

- `configs/manifests/coco_train.json`
- `configs/manifests/coco_val.json`
- `configs/manifests/bdd_day_train.json`
- `configs/manifests/bdd_day_val.json`
- `configs/manifests/bdd_night_train.json`
- `configs/manifests/bdd_night_val.json`
- `configs/manifests/acdc_train.json`
- `configs/manifests/acdc_val.json`

### ACDC fallback (if Zoo download fails)

If dataset loading fails in `notebooks/14_train_lora_acdc.ipynb`, set local dataset fields in `configs/training/lora_acdc.yaml`:

- `data.local_dataset_dir: /path/to/acdc`
- `data.local_dataset_type: COCODetectionDataset`

Then re-run `notebooks/14_train_lora_acdc.ipynb` from the beginning.

### Zoo fallback policy (CLI/data export)

For `scripts/pipelines/export_with_fiftyone.py`, fallback behavior is controlled by:

- `--fallback_policy strict|permissive|off` (default: `strict`)
- `--zoo_load_retries` (default: `2`)

Policy semantics:

- `strict`: fallback to `local_dataset_dir` only for known Zoo/network/load errors
- `permissive`: fallback for any Zoo load error
- `off`: disable fallback completely

Examples:

```bash
# strict fallback (recommended)
python scripts/pipelines/export_with_fiftyone.py \
  --zoo_name coco-2017 \
  --split train \
  --dataset_name coco_base_train \
  --export_dir data/exports/coco/train \
  --manifest_path configs/manifests/coco_train.json \
  --local_dataset_dir /path/to/local/coco \
  --fallback_policy strict \
  --zoo_load_retries 2
```

```bash
# no fallback: fail fast on Zoo errors
python scripts/pipelines/export_with_fiftyone.py \
  --zoo_name coco-2017 \
  --split train \
  --dataset_name coco_base_train \
  --export_dir data/exports/coco/train \
  --manifest_path configs/manifests/coco_train.json \
  --fallback_policy off
```

## 3) Notebook workflow (strict order)

Open `notebooks/` and run in this exact order:

1. `notebooks/11_train_coco_base.ipynb`
2. `notebooks/12_train_lora_bdd_day.ipynb`
3. `notebooks/13_train_lora_bdd_night.ipynb`
4. `notebooks/14_train_lora_acdc.ipynb`
5. `notebooks/20_eval_shared.ipynb`

## 4) How to run each training notebook correctly

For each of the 4 training notebooks:

1. Check `CONFIG_PATH` (must point to the matching `configs/training/*.yaml`).
2. First set `RUN_MODE = 'pilot'` and run all cells.
3. After a successful pilot run, set `RUN_MODE = 'full'`.
4. Run all cells again.

Important: notebooks enforce a pilot-before-full flow. If `RUN_MODE='pilot'`, full training is skipped.

## 5) Dependency order between notebooks

1. `11_train_coco_base.ipynb` must finish first and create:
   `checkpoints/base/coco_base.ckpt`
2. Only then run LoRA notebooks (12/13/14), because they depend on that base checkpoint.
3. Run `20_eval_shared.ipynb` only after base checkpoint + all 3 LoRA adapters exist.

## 6) Expected outputs after full run

- `checkpoints/base/coco_base.ckpt`
- `checkpoints/lora/bdd_day.safetensors`
- `checkpoints/lora/bdd_night.safetensors`
- `checkpoints/lora/acdc_adverse.safetensors`
- `results/shared_eval/metrics.json`
- `results/shared_eval/metrics.csv`
- `results/shared_eval/telemetry/*.jsonl`

## 7) Quick troubleshooting

1. `Missing manifest` in train/eval notebook:
   dataset export step did not complete; re-run the notebook from the top.
2. `Missing base checkpoint` in LoRA notebook:
   complete `11_train_coco_base.ipynb` in `RUN_MODE='full'` first.
3. Import errors (`mambavision`, `fiftyone`, `pycocotools`):
   reinstall dependencies in `.venv`, then run `preflight_check.py` again.
4. CUDA OOM:
   reduce `train.batch_size` in the relevant `configs/training/*.yaml`.

## 8) CLI

Use `scripts/pipelines/*.py` only as a fallback. Recommended default path in this project is notebook-first.
