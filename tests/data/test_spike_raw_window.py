import numpy as np
import pytest

from data.spike_recc import extract_centered_raw_window


def test_extract_centered_raw_window_uses_middle_by_default():
    spike_matrix = np.arange(9 * 2 * 3, dtype=np.float32).reshape(9, 2, 3)

    window = extract_centered_raw_window(spike_matrix, window_length=5)

    assert window.shape == (5, 2, 3)
    assert np.array_equal(window, spike_matrix[2:7])


def test_extract_centered_raw_window_accepts_explicit_center():
    spike_matrix = np.arange(11, dtype=np.float32).reshape(11, 1, 1)

    window = extract_centered_raw_window(spike_matrix, window_length=3, center_index=7)

    assert window.shape == (3, 1, 1)
    assert np.array_equal(window[:, 0, 0], np.array([6.0, 7.0, 8.0], dtype=np.float32))


@pytest.mark.parametrize("bad_length", [0, 4, -3])
def test_extract_centered_raw_window_rejects_non_positive_or_even_lengths(bad_length):
    spike_matrix = np.zeros((9, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="window_length"):
        extract_centered_raw_window(spike_matrix, window_length=bad_length)


def test_extract_centered_raw_window_rejects_window_larger_than_available_time_axis():
    spike_matrix = np.zeros((7, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="window_length=9"):
        extract_centered_raw_window(spike_matrix, window_length=9)
