"""LoRA low-rank adapters for nn.Linear layers.

Usage:
    inject_lora(model, target_substrings=["qkv", "proj"], rank=8, alpha=16)
    # train with base weights frozen
    merge_lora(model)  # in-place fold adapters back into nn.Linear
"""
import math
import copy
from typing import Iterable, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap nn.Linear: y = base(x) + (alpha/rank) * B(A(x)).

    Initialization: A Kaiming-uniform, B zeros — initial forward == base(x).
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling

    def merged_linear(self) -> nn.Linear:
        """Return a standalone nn.Linear with LoRA folded into the weight."""
        fused = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
        )
        delta = self.lora_B.weight @ self.lora_A.weight * self.scaling
        fused.weight.data = self.base.weight.data.detach().clone() + delta.detach()
        if self.base.bias is not None:
            fused.bias.data = self.base.bias.data.detach().clone()
        return fused


def inject_lora(
    model: nn.Module,
    target_substrings: Iterable[str],
    rank: int,
    alpha: float,
) -> List[str]:
    """Replace every `nn.Linear` whose *leaf* module name contains any of
    target_substrings with a `LoRALinear(m, rank, alpha)`.

    Returns dotted paths of replaced modules for logging.
    """
    targets = tuple(target_substrings)
    replaced: List[str] = []

    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if isinstance(child, LoRALinear):
                continue
            if not any(s in child_name for s in targets):
                continue
            wrapper = LoRALinear(child, rank=rank, alpha=alpha)
            wrapper = wrapper.to(child.weight.device, dtype=child.weight.dtype)
            setattr(parent, child_name, wrapper)
            dotted = f"{parent_name}.{child_name}" if parent_name else child_name
            replaced.append(dotted)

    return replaced


def merge_lora(model: nn.Module) -> nn.Module:
    """In-place: replace every LoRALinear in `model` with its merged nn.Linear."""
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                fused = child.merged_linear().to(
                    child.base.weight.device, dtype=child.base.weight.dtype
                )
                setattr(parent, child_name, fused)
    return model
