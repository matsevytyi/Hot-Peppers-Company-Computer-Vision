"""Export trained model to ONNX/TorchScript."""
import argparse
from pathlib import Path
import sys

import torch

# Allow running from any working directory
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from mamba.config import Config  # noqa: E402
from mamba.trainer import MambaDetectorModule  # noqa: E402


def export_model(checkpoint_path, output_dir, fmt="both"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    model = MambaDetectorModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    base_model = model.model

    dummy_input = torch.randn(
        1,
        config.data.sequence_length,
        3,
        config.data.img_size,
        config.data.img_size,
    )

    model_name = Path(checkpoint_path).stem

    if fmt in ["torchscript", "both"]:
        scripted_model = torch.jit.script(base_model)
        output_path = output_dir / f"{model_name}.pt"
        torch.jit.save(scripted_model, output_path)
        print(f"✅ TorchScript saved to: {output_path}")

    if fmt in ["onnx", "both"]:
        output_path = output_dir / f"{model_name}.onnx"
        torch.onnx.export(
            base_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"✅ ONNX saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output_dir", type=str, default="outputs/exported")
    parser.add_argument("--format", type=str, default="both", choices=["torchscript", "onnx", "both"])

    args = parser.parse_args()
    export_model(args.checkpoint, args.output_dir, args.format)
