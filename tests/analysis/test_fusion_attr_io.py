import json
from pathlib import Path

import pytest

from scripts.analysis.fusion_attr.io import (
    AnalysisSample,
    build_sample_output_dir,
    load_samples_file,
    strip_json_comments,
    write_json,
)


def test_strip_json_comments_keeps_urls_and_removes_comments():
    text = '{"url": "http://example.test/a//b", /* block */ "value": 3 // line\n}'
    cleaned = strip_json_comments(text)
    data = json.loads(cleaned)
    assert data == {"url": "http://example.test/a//b", "value": 3}


def test_load_samples_file_parses_box_mask(tmp_path: Path):
    samples_path = tmp_path / "fusion_samples.json"
    samples_path.write_text(
        """
        {
          "samples": [
            {
              "clip": "GOPR0384_11_02",
              "frame": "001301",
              "frame_index": 3,
              "mask": {"type": "box", "xyxy": [1, 2, 9, 10], "label": "motion_boundary"},
              "reason": "fast motion edge"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    samples = load_samples_file(samples_path)
    assert samples == [
        AnalysisSample(
            clip="GOPR0384_11_02",
            frame="001301",
            frame_index=3,
            mask_type="box",
            xyxy=(1, 2, 9, 10),
            mask_label="motion_boundary",
            reason="fast motion edge",
        )
    ]


def test_load_samples_file_rejects_invalid_box(tmp_path: Path):
    samples_path = tmp_path / "fusion_samples.json"
    samples_path.write_text(
        '{"samples":[{"clip":"a","frame":"b","frame_index":0,"mask":{"type":"box","xyxy":[1,2,3]},"reason":"x"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="xyxy must contain four integers"):
        load_samples_file(samples_path)


def test_build_sample_output_dir_is_stable(tmp_path: Path):
    sample = AnalysisSample(
        clip="GOPR0384_11_02",
        frame="001301",
        frame_index=3,
        mask_type="box",
        xyxy=(1, 2, 9, 10),
        mask_label="motion_boundary",
        reason="fast motion edge",
    )
    assert build_sample_output_dir(tmp_path, sample) == tmp_path / "samples" / "GOPR0384_11_02_001301"


def test_write_json_creates_parent_directory(tmp_path: Path):
    target = tmp_path / "nested" / "metadata.json"
    write_json(target, {"b": 2, "a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
