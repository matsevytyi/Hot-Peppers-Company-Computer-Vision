from pathlib import Path
import torch
import sys

from .contracts import TrainConfig
from .model_loader import create_model_from_config
from .training import load_checkpoint, resolve_device

def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / '.git').exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(Path.cwd().resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():

    cfg_path = REPO_ROOT / 'configs/training/moe_mambavis_day_night_adverse.yaml'

    cfg = TrainConfig.from_yaml(cfg_path)
    cfg.model.model_file = str((REPO_ROOT / cfg.model.model_file).resolve())
    cfg.ckpt.output_path = str((REPO_ROOT / cfg.ckpt.output_path).resolve())
    if cfg.model.base_checkpoint:
        cfg.model.base_checkpoint = str((REPO_ROOT / cfg.model.base_checkpoint).resolve())
    if cfg.data.get('lora_output_path'):
        cfg.data['lora_output_path'] = str((REPO_ROOT / cfg.data['lora_output_path']).resolve())
    # print(cfg)
    
    device = resolve_device(cfg.train.device)
    model = create_model_from_config(cfg.model, device=str(device)).to(device)
    
    print("Loaded model type:", type(model))
    x = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    if isinstance(out, tuple):
        outputs, probs = out
        print("MoE outputs scales:", [o.shape for o in outputs])
        print("Router probs shape:", probs.shape)
    else:
        print("Base model outputs:", [o.shape for o in out])

if __name__ == "__main__":
    main()

# in comfig model must be base_model even for moe_model (moe_model is defined when loading)

# examples: 

# Base (no-MoE) config detected. Proceeding with base model
# Loaded model type: <class 'mamba_vision_ours_runtime.MambaVisionOurs'>
# Base model outputs: [torch.Size([2, 13, 28, 28]), torch.Size([2, 13, 14, 14]), torch.Size([2, 13, 7, 7])]


# MoE configuration detected. Wrapping base model...
# Loading checkpoint for domain 'day' from checkpoints/lora/bdd_day_train_blanket_epoch007.ckpt
# Loading checkpoint for domain 'night' from checkpoints/lora/bdd_night_train_blanket_epoch007.ckpt
# Loading checkpoint for domain 'adverse' from checkpoints/lora/acdc_train_blanket_epoch012.ckpt
# Detected LoRA rank: 8
# Loaded model type: <class 'mamba_vision_moe_runtime.MoEMambaVision'>
# MoE outputs scales: [torch.Size([2, 13, 28, 28]), torch.Size([2, 13, 14, 14]), torch.Size([2, 13, 7, 7])]
# Router probs shape: torch.Size([2, 3])