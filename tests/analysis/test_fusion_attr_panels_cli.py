from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.analysis.fusion_attr.io import save_gray_map_png, save_rgb_tensor_png
from scripts.analysis.fusion_attr.panels import make_six_column_panel


def test_save_rgb_tensor_png_writes_bgr_file(tmp_path: Path):
    tensor = torch.zeros(3, 4, 5)
    tensor[0] = 1.0
    path = tmp_path / "rgb.png"
    save_rgb_tensor_png(path, tensor)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img.shape == (4, 5, 3)
    assert int(img[:, :, 2].max()) == 255


def test_save_gray_map_png_writes_uint8_file(tmp_path: Path):
    path = tmp_path / "map.png"
    save_gray_map_png(path, torch.ones(4, 5))
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (4, 5)


def test_make_six_column_panel_writes_panel(tmp_path: Path):
    images = {
        "Blurry RGB": np.zeros((12, 16, 3), dtype=np.uint8),
        "Spike cue": np.ones((12, 16, 3), dtype=np.uint8) * 20,
        "Restored": np.ones((12, 16, 3), dtype=np.uint8) * 40,
        "Error reduction": np.ones((12, 16, 3), dtype=np.uint8) * 60,
        "Attribution heatmap": np.ones((12, 16, 3), dtype=np.uint8) * 80,
        "Fusion-specific map": np.ones((12, 16, 3), dtype=np.uint8) * 100,
    }
    out = tmp_path / "panel.png"
    make_six_column_panel(out, images)
    panel = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert panel is not None
    assert panel.shape[0] > 12
    assert panel.shape[1] > 16 * 5
