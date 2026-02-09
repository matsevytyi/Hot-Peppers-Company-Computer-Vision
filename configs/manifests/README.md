# Dataset Manifests

This folder stores generated `DatasetManifest` JSON files created after FiftyOne exports.

Required manifests for the current pipelines:

- `coco_train.json`
- `coco_val.json`
- `bdd_day_train.json`
- `bdd_day_val.json`
- `bdd_night_train.json`
- `bdd_night_val.json`
- `acdc_train.json`
- `acdc_val.json`

Each file follows this schema:

- `dataset_name`
- `source`
- `split`
- `class_list`
- `class_map`
- `root_dir`
- `images_dir`
- `labels_or_annotations`
- `num_images`
- `num_instances`
- `created_at`
