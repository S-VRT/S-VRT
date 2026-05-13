import numpy as np

from scripts.data_preparation.spike.prepare_spike_raw_window import (
    build_output_base,
    save_artifact,
)


def test_raw_window_preprocess_writes_raw_window_length_directory(tmp_path):
    base = build_output_base(tmp_path, "clipA", 9, "00000001")

    assert base == tmp_path / "clipA" / "raw_window_l9" / "00000001"


def test_raw_window_preprocess_saves_uint8_npy(tmp_path):
    base = tmp_path / "clipA" / "raw_window_l5" / "00000001"
    window = (np.arange(5 * 2 * 2, dtype=np.float32).reshape(5, 2, 2) % 2)

    out_path = save_artifact(base, window, artifact_format="npy", storage_dtype="uint8")

    loaded = np.load(out_path)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, window.astype(np.uint8))
