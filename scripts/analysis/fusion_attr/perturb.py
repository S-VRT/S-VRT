from __future__ import annotations

import torch


def perturb_spike(spike: torch.Tensor, mode: str) -> torch.Tensor:
    normalized = str(mode).strip().lower()
    if normalized == "zero":
        return torch.zeros_like(spike)
    if normalized == "noise":
        return torch.randn_like(spike) * spike.detach().float().std().clamp_min(1e-6)
    if normalized == "shuffle":
        flat = spike.reshape(-1)
        perm = torch.randperm(flat.numel(), device=spike.device)
        return flat[perm].reshape_as(spike)
    if normalized == "temporal-drop":
        out = spike.clone()
        if out.ndim < 5:
            raise ValueError("temporal-drop expects spike tensor [B,T,C,H,W]")
        out[:, out.shape[1] // 2] = 0
        return out
    raise ValueError(f"Unsupported spike perturbation mode: {mode}")
