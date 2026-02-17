"""Evaluate base and LoRA models from a shared YAML config."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from pipelines.coco_dataset import build_dataloader  # noqa: E402
from pipelines.contracts import DatasetManifest, EvalConfig, ModelSection  # noqa: E402
from pipelines.evaluation import evaluate_model_detailed  # noqa: E402
from pipelines.lora import inject_lora_modules, load_lora_adapters  # noqa: E402
from pipelines.model_loader import create_model_from_config  # noqa: E402
from pipelines.training import load_checkpoint, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    return parser.parse_args()


def _load_model(model_cfg_dict: dict, device: str):
    section = ModelSection.from_dict(model_cfg_dict["model"])
    section.model_file = str((REPO_ROOT / section.model_file).resolve())
    model = create_model_from_config(section, device=device)

    base_checkpoint = model_cfg_dict.get("base_checkpoint")
    if base_checkpoint:
        base_path = Path(base_checkpoint)
        if not base_path.is_absolute():
            base_path = (REPO_ROOT / base_path).resolve()
        load_checkpoint(base_path, model)

    lora_adapter = model_cfg_dict.get("lora_adapter")
    if lora_adapter:
        lora_path = Path(lora_adapter)
        if not lora_path.is_absolute():
            lora_path = (REPO_ROOT / lora_path).resolve()
        lora_cfg = model_cfg_dict.get("lora", {})
        inject_lora_modules(
            model.backbone,
            rank=int(lora_cfg.get("rank", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            target_rule=str(lora_cfg.get("target_rule", "all_linear_except_head")),
        )
        missing, unexpected = load_lora_adapters(model, str(lora_path), strict=False)
        print(f"Loaded LoRA adapter: {lora_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    return model, section


def _safe_name(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return compact.strip("_") or "item"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    cfg = EvalConfig.from_yaml(config_path).eval
    device = str(resolve_device(cfg.get("device", "cuda")))
    output_dir = Path(cfg.get("output_dir", "results/shared_eval"))
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_images_quick = int(cfg.get("max_images_quick", 512))
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 4))
    conf_threshold = float(cfg.get("conf_threshold", 0.25))
    nms_iou = float(cfg.get("nms_iou", 0.5))
    power_cfg = dict(cfg.get("power", {}))
    power_enabled = bool(power_cfg.get("enabled", True))
    power_backend = str(power_cfg.get("backend", "auto"))
    power_gpu_index = int(power_cfg.get("gpu_index", 0))
    power_poll_interval_ms = int(power_cfg.get("poll_interval_ms", 100))
    warmup_batches = int(power_cfg.get("warmup_batches", 20))
    telemetry_enabled = bool(power_cfg.get("telemetry", True))
    telemetry_dir = Path(power_cfg.get("telemetry_dir", str(output_dir / "telemetry")))
    if not telemetry_dir.is_absolute():
        telemetry_dir = (REPO_ROOT / telemetry_dir).resolve()
    if telemetry_enabled:
        telemetry_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_cfg in cfg.get("datasets", []):
        dataset_name = dataset_cfg["name"]
        manifest_path = Path(dataset_cfg["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = (REPO_ROOT / manifest_path).resolve()
        manifest = DatasetManifest.from_json(manifest_path)
        loader = build_dataloader(
            manifest,
            image_size=int(dataset_cfg.get("image_size", 640)),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            max_samples=max_images_quick if args.mode == "quick" else None,
        )

        for model_cfg in cfg.get("models", []):
            model, section = _load_model(model_cfg, device=device)
            result = evaluate_model_detailed(
                model=model,
                dataloader=loader,
                num_classes=section.num_classes,
                image_size=int(dataset_cfg.get("image_size", 640)),
                device_preference=device,
                conf_threshold=conf_threshold,
                nms_iou=nms_iou,
                power_enabled=power_enabled,
                power_backend=power_backend,
                power_gpu_index=power_gpu_index,
                power_poll_interval_ms=power_poll_interval_ms,
                warmup_batches=warmup_batches,
                collect_telemetry=telemetry_enabled,
            )
            row = {"dataset": dataset_name, "model": model_cfg["name"], **result.metrics}
            if telemetry_enabled:
                telemetry_file = telemetry_dir / f"{_safe_name(dataset_name)}__{_safe_name(model_cfg['name'])}.jsonl"
                with open(telemetry_file, "w", encoding="utf-8") as f:
                    for entry in result.telemetry:
                        f.write(json.dumps(entry, ensure_ascii=True))
                        f.write("\n")
                row["telemetry_file"] = str(telemetry_file)
            rows.append(row)
            print(row)

    json_path = output_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")

    csv_path = output_dir / "metrics.csv"
    if rows:
        headers = sorted({k for row in rows for k in row.keys()})
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved metrics: {json_path}")
    print(f"Saved table: {csv_path}")


if __name__ == "__main__":
    main()
