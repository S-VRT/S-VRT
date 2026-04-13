from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

WINDOW_LENGTH = 25
WINDOW_HALF = WINDOW_LENGTH // 2


def validate_encoding25_tensor(tensor: np.ndarray) -> None:
    """Validate strict SCFlow encoding tensor shape [25, H, W]."""
    if tensor.ndim != 3:
        raise ValueError(f"encoding25 tensor must be [25,H,W], got ndim={tensor.ndim}")
    if int(tensor.shape[0]) != WINDOW_LENGTH:
        raise ValueError(
            f"encoding25 tensor expected {WINDOW_LENGTH} channels, got {tensor.shape[0]}"
        )


def build_output_dir(clip_dir: Path, dt: int) -> Path:
    """Return dataset-local artifact directory: <clip>/encoding25_dt{dt}."""
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    return Path(clip_dir) / f"encoding25_dt{int(dt)}"


def compute_center_index(
    frame_index: int,
    clip_start_frame: int,
    dt: int,
    center_offset: int = 40,
) -> int:
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    local_index = int(frame_index) - int(clip_start_frame)
    return int(center_offset) + local_index * int(dt)


def validate_center_bounds(
    *,
    center: int,
    total_length: int,
    edge_margin: int = 40,
    length: int = WINDOW_LENGTH,
    clip: Optional[str] = None,
    frame: Optional[int] = None,
    dt: Optional[int] = None,
    center_offset: Optional[int] = None,
) -> None:
    if length != WINDOW_LENGTH:
        raise ValueError("SCFlow strict mode requires length=25")

    if center - WINDOW_HALF < edge_margin or center + WINDOW_HALF >= total_length - edge_margin:
        clip_name = clip if clip is not None else "<unknown-clip>"
        frame_name = frame if frame is not None else "<unknown-frame>"
        raise ValueError(
            "Invalid encoding25 center bounds: "
            f"clip={clip_name}, frame={frame_name}, center={center}, T={total_length}, "
            f"edge_margin={edge_margin}, length={length}, dt={dt}, center_offset={center_offset}."
        )


def build_centered_window(
    spike_matrix: np.ndarray,
    center: int,
    length: int = WINDOW_LENGTH,
) -> np.ndarray:
    if spike_matrix.ndim != 3:
        raise ValueError(f"spike_matrix must be [T,H,W], got shape={tuple(spike_matrix.shape)}")
    if length != WINDOW_LENGTH:
        raise ValueError("SCFlow strict mode requires length=25")

    st = int(center) - WINDOW_HALF
    ed = int(center) + WINDOW_HALF + 1
    if st < 0 or ed > int(spike_matrix.shape[0]):
        raise ValueError(
            f"center={center} out of valid range for length={length}, T={spike_matrix.shape[0]}"
        )

    window = spike_matrix[st:ed].astype(np.float32)
    validate_encoding25_tensor(window)
    return window
