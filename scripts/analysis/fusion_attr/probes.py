from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class FusionProbeRecord:
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor
    module_name: str


def _detach_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (tuple, list)):
        return tuple(_detach_tensor(v) for v in value)
    return value


def find_fusion_adapter(model: nn.Module) -> nn.Module:
    if hasattr(model, "fusion_adapter"):
        return getattr(model, "fusion_adapter")
    if hasattr(model, "netG"):
        net = getattr(model, "netG")
        if hasattr(net, "fusion_adapter"):
            return getattr(net, "fusion_adapter")
    for _, module in model.named_modules():
        if module.__class__.__name__.lower().endswith("fusionadapter"):
            return module
    raise ValueError("Could not find fusion_adapter on model")


class FusionProbe:
    def __init__(self, module: nn.Module):
        self.module = module
        self.record: FusionProbeRecord | None = None
        self._handle = None

    def attach(self) -> None:
        self.close()
        self._handle = self.module.register_forward_hook(self._hook)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        tensor_output = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor_output, torch.Tensor):
            return
        if tensor_output.requires_grad:
            tensor_output.retain_grad()
        tensor_inputs = tuple(v for v in _detach_tensor(inputs) if isinstance(v, torch.Tensor))
        self.record = FusionProbeRecord(
            inputs=tensor_inputs,
            output=tensor_output,
            module_name=module.__class__.__name__,
        )


def _channel_norm(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.detach().float()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim == 3:
        return torch.linalg.vector_norm(data, dim=0)
    if data.ndim == 2:
        return data
    raise ValueError(f"Expected 2D, 3D, 4D, or 5D tensor, got shape {tuple(tensor.shape)}")


def reduce_operator_explanations(explanations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    reduced: dict[str, torch.Tensor] = {}
    if "gate" in explanations:
        gate = explanations["gate"].detach().float()
        if gate.ndim == 4:
            reduced["gate_mean"] = gate[0].mean(dim=0)
        elif gate.ndim == 5:
            reduced["gate_mean"] = gate[0, gate.shape[1] // 2].mean(dim=0)
    if "correction" in explanations:
        reduced["correction_norm"] = _channel_norm(explanations["correction"])
    if "effective_update" in explanations:
        reduced["effective_update"] = _channel_norm(explanations["effective_update"])
    for name, value in explanations.items():
        if name not in {"gate", "correction", "effective_update"}:
            reduced[name] = _channel_norm(value)
    return reduced
