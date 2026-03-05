from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from safetensors.torch import load_file

try:
    from .adapters.lora import inject_lora_modules, LoRALinear
    from .adapters.router import RouterMLP, _image_stats

    from .base_model import MambaVisionOurs, check_shapes

except ImportError:
    from adapters.lora import inject_lora_modules, LoRALinear
    from adapters.router import RouterMLP, _image_stats

    from base_model import MambaVisionOurs, check_shapes


class MoEMambaVision(nn.Module):
    """Wrapper that radds router to inject LoRA adapters at a runtime."""

    def __init__(self, base_model: MambaVisionOurs, adapter_paths: Dict[str, str], target_rule: str = "all_linear_except_head"):
        super().__init__()
        self.model = base_model
        self.router = RouterMLP()
        self.domains = list(adapter_paths.keys())

        # Load all raw checkpoints into memory first
        raw_states = {}
        for domain, path in adapter_paths.items():
            print(f"Loading checkpoint for domain '{domain}' from {path}")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            
            # If it's a standard PyTorch/Lightning save dict, extract the model weights
            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt # Assume it's a raw weight dict
                
            # Filter to keep ONLY LoRA keys
            lora_state = {k: v for k, v in state.items() if "lora_" in k}
            if not lora_state:
                raise ValueError(f"No LoRA weights found in checkpoint: {path}")
                
            raw_states[domain] = lora_state

        # Inject the multi-adapter LoRA layers into the base model
        # Infer rank from first adapter
        first_state = next(iter(raw_states.values()))
        rank = 8
        for k, v in first_state.items():
            if "lora_A" in k:
                rank = v.shape[0]
                print("""Detected LoRA rank: {}""".format(rank))
                break
                
        inject_lora_modules(
            self.model.backbone, 
            domains=self.domains, 
            rank=rank, 
            alpha=rank, 
            dropout=0.0, 
            target_rule=target_rule
        )

        # 3. Format and load the state dicts into the new ParameterDict structure
        for domain, state in raw_states.items():
            domain_state = {}
            for k, v in state.items():
                if "lora_A" in k:
                    domain_state[k.replace("lora_A", f"lora_A.{domain}")] = v
                elif "lora_B" in k:
                    domain_state[k.replace("lora_B", f"lora_B.{domain}")] = v
            
            # Load this specific domain's weights into the backbone
            missing, unexpected = self.model.backbone.load_state_dict(domain_state, strict=False)

    def _set_active_domain(self, domain_idx: int):
        """Helper to tell all LoRA layers which domain to use for the upcoming forward pass"""
        for module in self.model.backbone.modules():
            if isinstance(module, LoRALinear):
                # temporarily store the active domain on the module instance
                # so the forward pass knows which matrix to pick
                module.active_domain_idx = domain_idx

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        probs = self.router(x)
        top_domains = probs.argmax(dim=1) # [Batch_size]
        
        batch_outputs = [None] * x.shape[0]
        
        # Group batch by domain to preserve parallel processing
        for domain_idx, domain_name in enumerate(self.domains):
            # which images belong to this domain
            mask = top_domains == domain_idx
            indices = mask.nonzero(as_tuple=True)[0]
            
            if len(indices) == 0:
                continue
                
            # Tell the LoRA layers to use this domain
            self._set_active_domain(domain_idx)
            
            # Forward pass only the images for this domain
            domain_images = x[indices]
            domain_outs = self.model(domain_images)
            
            # Scatter the outputs back to their original batch positions
            for i, original_idx in enumerate(indices):
                batch_outputs[original_idx] = [scale[i:i+1] for scale in domain_outs]
                
        # Re-collate the outputs so it matches standard batch format ( List of 3 scales, each scale is [Batch, C, H, W] )
        final_outputs = []
        num_scales = len(batch_outputs[0])
        for scale_idx in range(num_scales):
            scale_tensors = [out[scale_idx] for out in batch_outputs]
            final_outputs.append(torch.cat(scale_tensors, dim=0))
            
        return final_outputs, probs

# utils   
def check_shapes(moe_model: MoEMambaVision, input_tensor: torch.Tensor):
    """
    Tests the MoE model by hooking into LoRA layers, checking the router,
    and verifying the standard Base Model multi-scale outputs.
    """

    moe_model.eval()
    
    # 0. Verify base model
    #check_shapes(moe_model.model, input_tensor)
    
    # 1. Verify LoRA Injection
    print("\n[1] LORA INJECTION:")
    lora_count = 0
    lora_layers = []
    
    for name, module in moe_model.model.backbone.named_modules():
        if module.__class__.__name__ == 'LoRALinear': # Or isinstance(module, LoRALinear)
            lora_count += 1
            lora_layers.append(name)
            
    print(f"Total LoRA layers found: {lora_count}")
    if lora_count >= 3:
        print(f"First 3 injected at: {lora_layers[:3]} ...")
        print(f"Last 3 injected at: {lora_layers[-3:]}")
    elif lora_count > 0:
        print(f"Injected at: {lora_layers}")
    else:
        print("WARNING: No LoRA layers found in the backbone!")

    # 2. Forward Hooks to spy on a specific LoRA layer during forward pass
    hook_handles = []
    lora_spy_results = {}

    def lora_hook_fn(module_name):
        def hook(module, inp, out):
            lora_spy_results[module_name] = {
                "input": inp[0].shape,
                "output": out.shape
            }
        return hook

    # Attach hook to the very first LoRA layer we found
    if lora_count > 0:
        target_layer_name = lora_layers[0]
        # Recursively find the actual module object to attach the hook
        for name, module in moe_model.model.backbone.named_modules():
            if name == target_layer_name:
                handle = module.register_forward_hook(lora_hook_fn(name))
                hook_handles.append(handle)
                break

    # 3. Test the Router standalone
    print("\n[2] ROUTER:")
    with torch.no_grad():
        stats = _image_stats(input_tensor)
        print(f"Image Stats shape: {stats.shape} (Expected: [Batch, 4])")
        
        probs = moe_model.router(input_tensor)
        print(f"Router Probs shape: {probs.shape} (Expected: [Batch, {len(moe_model.domains)}])")
        print(f"Sample Probs: {probs[0].tolist()}")

    # 4. Full MoE Forward Pass
    print("\n[3] FULL MoE FORWARD PASS:")
    with torch.no_grad():
        outputs, out_probs = moe_model(input_tensor)
        
    print(f"MoE returned outputs of type: {type(outputs)}")
    print(f"Number of detection scales: {len(outputs)}")
    
    for i, out in enumerate(outputs):
        print(f"  Scale {i + 1} shape: {out.shape}")
        # Verify batch size wasn't lost in grouping
        assert out.shape[0] == input_tensor.shape[0], f"Batch size mismatch on scale {i}"

    # 5. LoRA Hook Results
    if lora_spy_results:
        print("\n[4] LORA TENSOR FLOW (From Hook):")
        for name, shapes in lora_spy_results.items():
            print(f"Layer: {name}")
            print(f"  Input  shape: {shapes['input']}")
            print(f"  Output shape: {shapes['output']}")
            assert shapes['input'][:-1] == shapes['output'][:-1], "Batch/Seq dimensions changed unexpectedly in LoRA!"

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()
        
    print("\n=== ALL TESTS PASSED SUCCESSFULLY ===")


# ==========================================
# Mock Loader implementation and Test Script
# ==========================================

def build_moe_from_config(cfg) -> nn.Module:
    """Example of how to construct the model from your config logic"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building base model on {device}...")
    
    # 1. Build Base
    base_model = MambaVisionOurs(
        model_type=cfg.model.backbone,
        device=device,
        num_output_classes=cfg.model.num_classes,
        pretrained=False # Set to false for testing shapes
    ).to(device)

    # 2. Check if config requests MoE routing
    if hasattr(cfg.model, "moe_adapters") and cfg.model.moe_adapters:
        print("\n=== MoE CONFIGURATION DETECTED ===")
        print(f"Found adapters for domains: {list(cfg.model.moe_adapters.keys())}")
        print("Wrapping base model with MoEMambaVision router...")
        
        # In a real environment, you would use load_moe_wrapper_class from model_loader.py
        # Here we just instantiate the class directly since it's in the same file
        moe_model = MoEMambaVision(
            base_model=base_model,
            adapter_paths=cfg.model.moe_adapters,
            target_rule="all_linear_except_head"
        ).to(device)
        return moe_model

    print("No MoE configuration found. Returning base model.")
    return base_model


if __name__ == "__main__":

    from pathlib import Path
    
    # Get the absolute path to 'Hot-Peppers-Company-Computer-Vision'
    repo_root = Path(__file__).resolve().parent.parent 
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    # Optional: explicitly set the package name so relative imports resolve
    __package__ = "mamba-vision-ours"

    # 1. Create a mock config that perfectly mimics YAML structure
    class MockModelSection:
        model_file = "mamba-vision-ours/model.py"
        backbone = "mamba_vision_T2"
        num_classes = 8
        pretrained = False
        checkpoint_path = ""
        base_checkpoint = "checkpoints/base/coco_base_epoch009.ckpt"
        
        # MOCK ADAPTER PATHS
        moe_adapters = {
            "day": "checkpoints/lora/bdd_day_train_blanket_epoch007.ckpt",
            "night": "checkpoints/lora/bdd_night_train_blanket_epoch007.ckpt",
            "adverse": "checkpoints/lora/acdc_train_blanket_epoch012.ckpt"
        }

    class MockConfig:
        model = MockModelSection()
    
    cfg = MockConfig()
    
    try:
        # Build the model using the mock config
        model = build_moe_from_config(cfg)
        
        device = next(model.parameters()).device
        dummy_input = torch.randn(8, 3, 224, 224).to(device) # Batch of 8
        
        # Run the shape checks
        check_shapes(model, dummy_input)
        
    except FileNotFoundError as e:
        print(f"\n[TEST FAILED - FILE NOT FOUND]: {e}")
        print("-> To run this test locally, ensure the 'moe_adapters' paths in MockModelSection point to actual .ckpt files!")
    except Exception as e:
        print(f"\n[TEST FAILED]: {e}")

    # EXAMPLE OF OUTPUT

    # Building base model on cuda...

    # === MoE CONFIGURATION DETECTED ===
    # Found adapters for domains: ['day', 'night', 'adverse']
    # Wrapping base model with MoEMambaVision router...
    # Loading checkpoint for domain 'day' from checkpoints/lora/bdd_day_train_blanket_epoch007.ckpt
    # Loading checkpoint for domain 'night' from checkpoints/lora/bdd_night_train_blanket_epoch007.ckpt
    # Loading checkpoint for domain 'adverse' from checkpoints/lora/acdc_train_blanket_epoch012.ckpt
    # Detected LoRA rank: 8

    # [1] LORA INJECTION:
    # Total LoRA layers found: 76
    # First 3 injected at: ['levels.2.blocks.0.mixer.in_proj', 'levels.2.blocks.0.mixer.x_proj', 'levels.2.blocks.0.mixer.dt_proj'] ...
    # Last 3 injected at: ['levels.3.blocks.3.mixer.proj', 'levels.3.blocks.3.mlp.fc1', 'levels.3.blocks.3.mlp.fc2']

    # [2] ROUTER:
    # Image Stats shape: torch.Size([8, 4]) (Expected: [Batch, 4])
    # Router Probs shape: torch.Size([8, 3]) (Expected: [Batch, 3])
    # Sample Probs: [0.4196569621562958, 0.3114374577999115, 0.26890552043914795]

    # [3] FULL MoE FORWARD PASS:
    # MoE returned outputs of type: <class 'list'>
    # Number of detection scales: 3
    #   Scale 1 shape: torch.Size([8, 13, 28, 28])
    #   Scale 2 shape: torch.Size([8, 13, 14, 14])
    #   Scale 3 shape: torch.Size([8, 13, 7, 7])

    # [4] LORA TENSOR FLOW (From Hook):
    # Layer: levels.2.blocks.0.mixer.in_proj
    #   Input  shape: torch.Size([1, 196, 320])
    #   Output shape: torch.Size([1, 196, 320])

    # === ALL TESTS PASSED SUCCESSFULLY ===
