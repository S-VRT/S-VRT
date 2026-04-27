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

    @property
    def weight(self) -> torch.Tensor:
        """Expose an nn.Linear-compatible effective weight view.

        Some third-party modules access `.weight` directly instead of calling the
        layer as a module. Return the base weight plus the current LoRA delta so
        those callers observe the adapted projection.
        """
        delta = self.lora_B.weight @ self.lora_A.weight
        return self.base.weight + delta * self.scaling

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

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
            dotted = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(
                part in dotted
                for part in (
                    "fusion_adapter",
                    "fusion_operator",
                    "mamba_token_mixer",
                    "pa_deform",
                    ".dcn.",
                    "spynet",
                )
            ):
                continue
            wrapper = LoRALinear(child, rank=rank, alpha=alpha)
            wrapper = wrapper.to(child.weight.device, dtype=child.weight.dtype)
            setattr(parent, child_name, wrapper)
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


def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Export a LoRA-folded state_dict without deepcopying the live module tree.

    This is robust when the training model carries runtime tensor attributes that
    are not deepcopy-compatible, such as non-leaf tensors cached during forward.
    """
    merged = {}
    lora_module_names = set()
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        lora_module_names.add(name)
        fused = module.merged_linear()
        weight_key = f"{name}.weight" if name else "weight"
        merged[weight_key] = fused.weight.detach().cpu()
        if fused.bias is not None:
            bias_key = f"{name}.bias" if name else "bias"
            merged[bias_key] = fused.bias.detach().cpu()

    state_dict = model.state_dict()
    exported = {}
    for key, value in state_dict.items():
        if ".lora_A." in key or ".lora_B." in key:
            continue
        if any(key == f"{name}.base.weight" or key == f"{name}.base.bias" for name in lora_module_names):
            continue
        exported[key] = merged.get(key, value.detach().cpu())
    exported.update(merged)
    return exported
