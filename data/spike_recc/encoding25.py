from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

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


def compute_subframe_centers(
    t_raw: int,
    num_subframes: int,
    margin: int = WINDOW_HALF,
) -> List[int]:
    """Compute S evenly-spaced sub-centers within a single .dat file."""
    if t_raw < WINDOW_LENGTH:
        raise ValueError(f"t_raw={t_raw} too short for a {WINDOW_LENGTH}-wide window")
    if num_subframes < 1:
        raise ValueError(f"num_subframes must be >= 1, got {num_subframes}")
    lo = margin
    hi = t_raw - margin - 1
    if hi < lo:
        raise ValueError(
            f"t_raw={t_raw} with margin={margin} yields no valid center range "
            f"[{lo}, {hi}]"
        )
    if num_subframes == 1:
        return [(lo + hi) // 2]
    raw_centers = np.linspace(lo, hi, num_subframes)
    return [int(round(c)) for c in raw_centers]


def validate_subframes_tensor(tensor: np.ndarray, num_subframes: int) -> None:
    """Validate shape [S, 25, H, W] for multi-subframe encoding."""
    if tensor.ndim != 4:
        raise ValueError(f"subframes tensor must be ndim=4 [S,25,H,W], got ndim={tensor.ndim}")
    if tensor.shape[0] != num_subframes:
        raise ValueError(f"subframes tensor expected {num_subframes} subframes, got {tensor.shape[0]}")
    if tensor.shape[1] != WINDOW_LENGTH:
        raise ValueError(f"subframes tensor expected {WINDOW_LENGTH} channels, got {tensor.shape[1]}")


def build_output_dir_subframes(clip_dir: Path, dt: int, num_subframes: int) -> Path:
    """Return artifact directory with subframe suffix when S > 1."""
    if num_subframes <= 1:
        return build_output_dir(clip_dir, dt)
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    return Path(clip_dir) / f"encoding25_dt{int(dt)}_s{int(num_subframes)}"


def resolve_encoding25_format(
    format_value: str,
    *,
    base_path: Path,
) -> Tuple[str, Path]:
    """Resolve artifact format and concrete path from a base path without extension."""
    normalized = str(format_value or "auto").strip().lower()
    if normalized not in {"auto", "npy", "dat"}:
        raise ValueError(f"Unsupported encoding25 format: {format_value!r}")

    npy_path = base_path.with_suffix(".npy")
    dat_path = base_path.with_suffix(".dat")

    if normalized == "npy":
        return "npy", npy_path
    if normalized == "dat":
        return "dat", dat_path

    if npy_path.exists():
        return "npy", npy_path
    if dat_path.exists():
        return "dat", dat_path
    raise FileNotFoundError(f"Missing encoding25 artifact: {npy_path} or {dat_path}")


def _validate_binary_tensor(arr: np.ndarray) -> None:
    if arr.dtype == np.bool_:
        return
    if np.issubdtype(arr.dtype, np.integer):
        unique = np.unique(arr)
        if unique.size <= 2 and np.all((unique == 0) | (unique == 1)):
            return
    if np.issubdtype(arr.dtype, np.floating):
        unique = np.unique(arr)
        if unique.size <= 2 and np.all((unique == 0.0) | (unique == 1.0)):
            return
    raise ValueError(f"encoding25 .dat save expects binary spike tensor, got dtype={arr.dtype}")


def save_encoding25_artifact(path: Path, arr: np.ndarray, artifact_format: str) -> None:
    """Save encoding25 tensor as .npy or packed-binary .dat."""
    normalized = str(artifact_format).strip().lower()
    if normalized == "npy":
        np.save(path.with_suffix(".npy"), arr)
        return
    if normalized != "dat":
        raise ValueError(f"Unsupported encoding25 artifact format: {artifact_format!r}")

    _validate_binary_tensor(arr)
    bool_arr = arr.astype(bool, copy=False)
    packed = np.packbits(bool_arr.reshape(-1).astype(np.uint8), bitorder="little")
    path.with_suffix(".dat").write_bytes(packed.tobytes())


def load_encoding25_artifact_with_shape(
    base_path: Path,
    *,
    artifact_format: str = "auto",
    num_subframes: int = 1,
    spike_h: int,
    spike_w: int,
) -> np.ndarray:
    """Load encoding25 tensor from .npy or packed-binary .dat with explicit spatial shape."""
    resolved_format, path = resolve_encoding25_format(artifact_format, base_path=base_path)
    if resolved_format == "npy":
        arr = np.load(path).astype(np.float32)
    else:
        raw = np.fromfile(path, dtype=np.uint8)
        expected_values = int(num_subframes) * WINDOW_LENGTH * int(spike_h) * int(spike_w)
        unpacked = np.unpackbits(raw, bitorder="little")
        if unpacked.size < expected_values:
            raise ValueError(
                f"encoding25 packed artifact too small: {path}, expected at least {expected_values} values, "
                f"got {unpacked.size}"
            )
        if num_subframes > 1:
            arr = unpacked[:expected_values].reshape(num_subframes, WINDOW_LENGTH, spike_h, spike_w).astype(np.float32)
        else:
            arr = unpacked[:expected_values].reshape(WINDOW_LENGTH, spike_h, spike_w).astype(np.float32)
    if num_subframes > 1:
        validate_subframes_tensor(arr, num_subframes)
    else:
        validate_encoding25_tensor(arr)
    return arr
