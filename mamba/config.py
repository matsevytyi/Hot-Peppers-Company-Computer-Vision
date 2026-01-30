"""Configuration for Mamba UAV Detector."""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration."""

    data_root: str = "data/MMFW-UAV/sample"
    sensor_type: str = "Zoom"
    view: str = "Top_Down"
    sequence_length: int = 10
    stride: int = 5
    img_size: int = 640
    batch_size: int = 4
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""

    backbone: str = "mobilevit_s"
    d_model: int = 256
    d_state: int = 16
    mamba_layers: int = 4
    output_last_only: bool = True
    freeze_backbone: bool = False
    pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""

    max_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # cosine, step, none
    warmup_epochs: int = 5
    gradient_clip_val: float = 1.0

    bbox_loss_weight: float = 1.0
    conf_loss_weight: float = 1.0

    log_every_n_steps: int = 10
    val_check_interval: float = 1.0

    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class Config:
    """Main configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    project_name: str = "uav-mamba"
    experiment_name: str = "mamba-mobilevit-s10"
    use_wandb: bool = True

    accelerator: str = "gpu"  # gpu, cpu, mps
    devices: int = 1

    seed: int = 42
