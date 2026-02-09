# Pipeline Configs

## TrainConfig files

- `training/coco_base.yaml`
- `training/lora_bdd_day.yaml`
- `training/lora_bdd_night.yaml`
- `training/lora_acdc.yaml`

Each train config includes:

- `model.backbone`
- `model.num_classes`
- `train.epochs`
- `train.batch_size`
- `train.lr`
- `train.weight_decay`
- `train.scheduler`
- `train.pilot_steps`
- `train.precision`
- `train.device`
- `ckpt.save_top_k`
- `ckpt.save_last`

LoRA runs add:

- `lora.rank`
- `lora.alpha`
- `lora.dropout`
- `lora.target_rule`
- `freeze.backbone_base`
- `freeze.neck`
- `freeze.head`

## EvalConfig file

- `eval/shared_eval.yaml`

Fields:

- `eval.datasets`
- `eval.metrics`
- `eval.conf_threshold`
- `eval.nms_iou`
- `eval.max_images_quick`
- `eval.save_predictions`

## Dataset manifests

Generated JSON manifests are stored in `manifests/` and consumed by training/eval.
