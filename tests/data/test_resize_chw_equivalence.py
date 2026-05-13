import cv2
import numpy as np

from data.dataset_video_train_rgbspike import resize_chw_image


def _resize_chw_channelwise(arr_chw, size):
    resized = [
        cv2.resize(arr_chw[ch], size, interpolation=cv2.INTER_LINEAR)
        for ch in range(arr_chw.shape[0])
    ]
    return np.stack(resized, axis=0).astype(np.float32)


def test_resize_chw_image_matches_channelwise_for_spike_voxels():
    rng = np.random.default_rng(19931005)
    arr = rng.random((4, 37, 53), dtype=np.float32)

    actual = resize_chw_image(arr, (19, 23))
    expected = _resize_chw_channelwise(arr, (19, 23))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-6)


def test_resize_chw_image_matches_channelwise_for_encoding25_flow():
    rng = np.random.default_rng(19931006)
    arr = rng.random((25, 41, 29), dtype=np.float32)

    actual = resize_chw_image(arr, (17, 31))
    expected = _resize_chw_channelwise(arr, (17, 31))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-6)
