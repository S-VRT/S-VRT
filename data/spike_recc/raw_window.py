from __future__ import annotations

import numpy as np


def extract_centered_raw_window(
    spike_matrix: np.ndarray,
    window_length: int,
    center_index: int | None = None,
) -> np.ndarray:
    if spike_matrix.ndim != 3:
        raise ValueError(f"spike_matrix must be 3D (T, H, W), got shape {spike_matrix.shape}")
    if window_length <= 0 or window_length % 2 == 0:
        raise ValueError(f"window_length must be a positive odd integer, got {window_length}")

    total_steps = int(spike_matrix.shape[0])
    if window_length > total_steps:
        raise ValueError(
            f"window_length={window_length} exceeds available spike steps T={total_steps}"
        )

    resolved_center = total_steps // 2 if center_index is None else int(center_index)
    half = window_length // 2
    start = resolved_center - half
    end = resolved_center + half + 1
    if start < 0 or end > total_steps:
        raise ValueError(
            f"Centered raw window [{start}:{end}] is out of bounds for T={total_steps}"
        )

    return spike_matrix[start:end].astype(np.float32, copy=False)


__all__ = ["extract_centered_raw_window"]
