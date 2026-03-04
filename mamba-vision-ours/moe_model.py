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

from .adapters.lora import inject_lora_modules, LoRALinear
from .adapters.router import _image_stats, RouterMLP

from .base_model import MambaVisionOurs

class MoEMambaVision(nn.Module):
    def __init__(self, base_model: MambaVisionOurs, adapter_paths: Dict[str, str], target_rule: str = "all_linear_except_head"):
        super().__init__()
        self.model = base_model
        self.router = RouterMLP()
        self.domains = list(adapter_paths.keys())

        # Inject the multi-adapter LoRA layers FIRST
        # Infer rank from first adapter
        first_state = load_file(str(list(adapter_paths.values())[0]))
        rank = 8
        for k, v in first_state.items():
            if k.endswith("lora_A"):
                rank = v.shape[0]
                break
                
        inject_lora_modules(
            self.model.backbone, 
            domains=self.domains, 
            rank=rank, 
            alpha=rank, 
            dropout=0.0, 
            target_rule=target_rule
        )

        # Load the state dicts directly into the respective parameter dicts ONCE
        for domain, path in adapter_paths.items():
            state = load_file(str(path))
            # Rename keys from saved 'lora_A' to our new 'lora_A.domain' format
            domain_state = {}
            for k, v in state.items():
                if "lora_A" in k:
                    domain_state[k.replace("lora_A", f"lora_A.{domain}")] = v
                elif "lora_B" in k:
                    domain_state[k.replace("lora_B", f"lora_B.{domain}")] = v
            
            # Load this specific domain's weights into the backbone
            self.model.backbone.load_state_dict(domain_state, strict=False)


    def _set_active_domain(self, domain_idx: int):
        """Helper to tell all LoRA layers which domain to use for the upcoming forward pass"""
        for module in self.model.backbone.modules():
            if isinstance(module, LoRALinear):
                # temporarily store the active domain on the module instance
                # so the forward pass knows which matrix to pick
                module.active_domain_idx = domain_idx

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:

        domain_probabilities = self.router(x)
        top_domains = domain_probabilities.argmax(dim=1) # [Batch_size]
        
        # Determine number of output scales
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
            
        return final_outputs, domain_probabilities
