import numpy as np
import utils.utils_video as utils_video


def test_returns_crop_params_dict():
    """paired_random_crop must return (gts, lqs, crop_params) 3-tuple."""
    h, w = 64, 80
    patch = 16
    gt = np.random.rand(h, w, 3).astype(np.float32)
    lq = np.random.rand(h, w, 3).astype(np.float32)
    result = utils_video.paired_random_crop(gt, lq, patch, scale=1)
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
    gts, lqs, crop_params = result
    assert isinstance(crop_params, dict)
    assert set(crop_params.keys()) == {"top", "left", "lq_patch_size"}
    assert crop_params["lq_patch_size"] == patch


def test_crop_params_coordinates_in_bounds():
    """top/left must be non-negative and within (h - patch, w - patch)."""
    h, w = 64, 80
    patch = 16
    gt = np.random.rand(h, w, 3).astype(np.float32)
    lq = np.random.rand(h, w, 3).astype(np.float32)
    for _ in range(50):
        _, _, crop_params = utils_video.paired_random_crop(gt, lq, patch, scale=1)
        assert 0 <= crop_params["top"] <= h - patch
        assert 0 <= crop_params["left"] <= w - patch


def test_crop_params_with_scale_factor():
    """With scale > 1, top/left are in LQ coordinate space, lq_patch_size = gt_patch_size // scale."""
    scale = 4
    gt_patch = 64
    lq_h, lq_w = 32, 40
    gt_h, gt_w = lq_h * scale, lq_w * scale
    gt = np.random.rand(gt_h, gt_w, 3).astype(np.float32)
    lq = np.random.rand(lq_h, lq_w, 3).astype(np.float32)
    _, _, crop_params = utils_video.paired_random_crop(gt, lq, gt_patch, scale=scale)
    assert crop_params["lq_patch_size"] == gt_patch // scale  # 16
    assert 0 <= crop_params["top"] <= lq_h - crop_params["lq_patch_size"]
    assert 0 <= crop_params["left"] <= lq_w - crop_params["lq_patch_size"]
