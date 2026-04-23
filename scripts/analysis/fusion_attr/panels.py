from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


PANEL_COLUMNS = [
    "Blurry RGB",
    "Spike cue",
    "Restored",
    "Error reduction",
    "Attribution heatmap",
    "Fusion-specific map",
]


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    raise ValueError(f"Unsupported panel image shape: {image.shape}")


def make_six_column_panel(path: str | Path, images: dict[str, np.ndarray]) -> None:
    missing = [name for name in PANEL_COLUMNS if name not in images]
    if missing:
        raise ValueError(f"Missing panel columns: {missing}")
    cells = [_as_bgr(images[name]) for name in PANEL_COLUMNS]
    height = max(cell.shape[0] for cell in cells)
    width = max(cell.shape[1] for cell in cells)
    rendered = []
    for name, cell in zip(PANEL_COLUMNS, cells):
        resized = cv2.resize(cell, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.full((height + 28, width, 3), 255, dtype=np.uint8)
        canvas[:height] = resized
        cv2.putText(canvas, name, (4, height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        rendered.append(canvas)
    panel = np.concatenate(rendered, axis=1)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), panel)
