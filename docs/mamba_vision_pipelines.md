# Mamba-Vision Pipelines (COCO Base + LoRA)

This project now includes notebook-first pipelines for:

1. COCO base training
2. LoRA adaptation on BDD100K daytime
3. LoRA adaptation on BDD100K nighttime
4. LoRA adaptation on ACDC adverse-weather detection annotations
5. Shared evaluation across all trained variants

## Notebook entrypoints

- `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/notebooks/11_train_coco_base.ipynb`
- `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/notebooks/12_train_lora_bdd_day.ipynb`
- `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/notebooks/13_train_lora_bdd_night.ipynb`
- `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/notebooks/14_train_lora_acdc.ipynb`
- `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/notebooks/20_eval_shared.ipynb`

## Config files

- Train configs:
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/configs/training/coco_base.yaml`
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/configs/training/lora_bdd_day.yaml`
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/configs/training/lora_bdd_night.yaml`
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/configs/training/lora_acdc.yaml`
- Eval config:
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/configs/eval/shared_eval.yaml`

## Shared class policy

The current v1 shared label space is fixed to 8 classes:

- `person`
- `bicycle`
- `car`
- `motorcycle`
- `bus`
- `train`
- `truck`
- `traffic_light`

## Artifacts

- Base checkpoint:
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/checkpoints/base/coco_base.ckpt`
- LoRA adapters:
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/checkpoints/lora/bdd_day.safetensors`
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/checkpoints/lora/bdd_night.safetensors`
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/checkpoints/lora/acdc_adverse.safetensors`
- Evaluation outputs:
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/results/shared_eval/metrics.json`
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/results/shared_eval/metrics.csv`

## Optional CLI mirrors

The notebooks call the same building blocks as these scripts:

- export data with FiftyOne:
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/scripts/pipelines/export_with_fiftyone.py`
- train from config:
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/scripts/pipelines/train_from_config.py`
- evaluate from config:
  - `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/scripts/pipelines/eval_from_config.py`

## ACDC fallback

If `acdc` is not available through FiftyOne Zoo in your environment, set:

- `data.local_dataset_dir`
- `data.local_dataset_type` (default `COCODetectionDataset`)

in `/Users/ivantyshchenko/Projects/Python/Hot-Peppers-Company-Computer-Vision/configs/training/lora_acdc.yaml`.
