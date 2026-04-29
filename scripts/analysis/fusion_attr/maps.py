from __future__ import annotations

import numpy as np
import torch


def reduce_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.detach().float()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim == 3:
        return torch.linalg.vector_norm(data, dim=0)
    if data.ndim == 2:
        return data
    raise ValueError(f"Expected 2D, 3D, 4D, or 5D tensor, got {tuple(tensor.shape)}")


def normalize_map(values: torch.Tensor, low: float = 1.0, high: float = 99.0) -> torch.Tensor:
    arr = values.detach().float().cpu().numpy()
    lo = float(np.percentile(arr, low))
    hi = float(np.percentile(arr, high))
    if hi <= lo:
        return torch.zeros_like(values, dtype=torch.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return torch.from_numpy(out).to(dtype=torch.float32, device=values.device)


def compute_fusion_delta(fusion_output: torch.Tensor, rgb_reference: torch.Tensor) -> torch.Tensor:
    if fusion_output.shape != rgb_reference.shape:
        raise ValueError(
            f"fusion_output and rgb_reference shapes differ: {tuple(fusion_output.shape)} vs {tuple(rgb_reference.shape)}"
        )
    return reduce_to_2d(fusion_output - rgb_reference)


def compute_error_map(output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if output.shape != gt.shape:
        raise ValueError(f"output and gt shapes differ: {tuple(output.shape)} vs {tuple(gt.shape)}")
    data = (output.detach().float() - gt.detach().float()).abs()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Expected image tensor with channels, got {tuple(output.shape)}")
    return data.mean(dim=0)


def gradient_activation_cam(activation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if activation.grad is not None:
        activation.grad.zero_()
    target.backward(retain_graph=True)
    if activation.grad is None:
        raise RuntimeError("Activation gradient was not retained")
    grad = activation.grad.detach()
    act = activation.detach()
    if act.ndim == 5:
        weights = grad.mean(dim=(-1, -2), keepdim=True)
        cam = (weights * act).sum(dim=2)
        cam = cam[0, cam.shape[1] // 2]
    elif act.ndim == 4:
        weights = grad.mean(dim=(-1, -2), keepdim=True)
        cam = (weights * act).sum(dim=1)[0]
    else:
        raise ValueError(f"Expected 4D or 5D activation, got {tuple(act.shape)}")
    return torch.relu(cam.detach())


def _clone_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def integrated_gradients_map(
    model,
    inputs,
    baselines,
    target_fn,
    steps: int = 32,
    input_index: int = 0,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")

    tuple_input = isinstance(inputs, (tuple, list))
    if not tuple_input:
        inputs = (inputs,)
        baselines = (baselines,)

    if len(inputs) != len(baselines):
        raise ValueError("inputs and baselines must have the same arity")

    total_grad = None
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=inputs[input_index].device)[1:]

    for alpha in alphas:
        scaled_inputs = []
        tracked_tensor = None
        for idx, (inp, base) in enumerate(zip(inputs, baselines)):
            scaled = base + alpha * (inp - base)
            if idx == input_index:
                tracked_tensor = _clone_with_grad(scaled)
                scaled_inputs.append(tracked_tensor)
            else:
                scaled_inputs.append(scaled.detach())
        output = model(*scaled_inputs)
        target = target_fn(output)
        if target.ndim != 0:
            raise ValueError("target_fn must return a scalar tensor")
        grad = torch.autograd.grad(target, tracked_tensor, retain_graph=False, create_graph=False)[0]
        total_grad = grad if total_grad is None else total_grad + grad

    avg_grad = total_grad / float(steps)
    attr = (inputs[input_index] - baselines[input_index]).detach() * avg_grad.detach()
    return reduce_to_2d(attr.abs())
