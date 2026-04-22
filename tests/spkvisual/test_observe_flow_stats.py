import json
from pathlib import Path

import numpy as np
import pytest

from spkvisual import observe_flow_stats as ofs


def test_strip_json_comments_preserves_urls_inside_strings():
    raw = '{"path": "http://example.test/a//b", // comment\n "value": 2}'

    parsed = json.loads(ofs.strip_json_comments(raw))

    assert parsed["path"] == "http://example.test/a//b"
    assert parsed["value"] == 2


def test_summarize_values_reports_percentiles_and_low_ratio():
    values = np.array([0.0, 1.0, 2.0, 10.0], dtype=np.float32)

    summary = ofs.summarize_values(values, low_threshold=1.5)

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(3.25)
    assert summary["max"] == pytest.approx(10.0)
    assert summary["p50"] == pytest.approx(1.5)
    assert summary["low_ratio"] == pytest.approx(0.5)


def test_select_subframe_supports_middle_mean_and_index():
    arr = np.stack(
        [
            np.full((25, 2, 2), 1.0, dtype=np.float32),
            np.full((25, 2, 2), 3.0, dtype=np.float32),
            np.full((25, 2, 2), 5.0, dtype=np.float32),
        ],
        axis=0,
    )

    middle, label = ofs.select_subframe(arr, "middle")
    meaned, mean_label = ofs.select_subframe(arr, "mean")
    indexed, index_label = ofs.select_subframe(arr, "2")

    assert label == "middle:1"
    assert mean_label == "mean"
    assert index_label == "index:2"
    assert np.all(middle == 3.0)
    assert np.all(meaned == 3.0)
    assert np.all(indexed == 5.0)


def test_select_subframe_rejects_invalid_index():
    arr = np.zeros((2, 25, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="out of range"):
        ofs.select_subframe(arr, "3")


def test_compare_flows_persists_console_conclusions(tmp_path):
    spynet = np.zeros((2, 2, 2), dtype=np.float32)
    scflow = np.zeros((2, 2, 2), dtype=np.float32)
    scflow[0, :, :] = 2.0
    spike_window = np.zeros((25, 2, 2), dtype=np.float32)
    spike_window[:, 0, :] = 1.0

    pair = ofs.analyze_pair_arrays(
        key="clip/000000",
        next_key="clip/000001",
        spynet_flow=spynet,
        scflow_flow=scflow,
        spike_window=spike_window,
        low_flow_threshold=0.5,
        active_threshold=0.1,
    )
    summary = ofs.build_run_summary(
        config={"device": "cpu"},
        pair_rows=[pair],
        histograms={"spynet_mag": [0.0], "scflow_mag": [2.0], "diff_mag": [2.0]},
    )
    out_path = tmp_path / "summary.json"

    ofs.write_json(out_path, summary)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))

    assert loaded["config"]["device"] == "cpu"
    assert "console_conclusions" in loaded
    assert "spynet" in loaded["aggregate"]
    assert "scflow" in loaded["aggregate"]
    assert "diff" in loaded["aggregate"]
    assert "active_spike_regions" in loaded["aggregate"]["scflow"]
