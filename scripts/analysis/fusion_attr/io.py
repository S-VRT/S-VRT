from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class AnalysisSample:
    clip: str
    frame: str
    frame_index: int
    mask_type: str
    xyxy: tuple[int, int, int, int]
    mask_label: str
    reason: str


def strip_json_comments(text: str) -> str:
    result: list[str] = []
    i = 0
    in_string = False
    string_char = ""
    while i < len(text):
        ch = text[i]
        if in_string:
            result.append(ch)
            if ch == "\\" and i + 1 < len(text):
                result.append(text[i + 1])
                i += 2
                continue
            if ch == string_char:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            result.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < len(text) and text[i] not in "\n\r":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def _require_str(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _parse_xyxy(values: Iterable[Any]) -> tuple[int, int, int, int]:
    parsed = tuple(int(v) for v in values)
    if len(parsed) != 4:
        raise ValueError("xyxy must contain four integers")
    x1, y1, x2, y2 = parsed
    if x2 <= x1 or y2 <= y1:
        raise ValueError("xyxy must satisfy x2 > x1 and y2 > y1")
    return (x1, y1, x2, y2)


def load_samples_file(path: str | Path) -> list[AnalysisSample]:
    sample_path = Path(path)
    data = json.loads(strip_json_comments(sample_path.read_text(encoding="utf-8")))
    raw_samples = data.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("samples must be a list")
    samples: list[AnalysisSample] = []
    for item in raw_samples:
        if not isinstance(item, dict):
            raise ValueError("each sample must be an object")
        mask = item.get("mask")
        if not isinstance(mask, dict):
            raise ValueError("sample mask must be an object")
        mask_type = str(mask.get("type", "")).strip().lower()
        if mask_type != "box":
            raise ValueError(f"unsupported mask type: {mask_type}")
        samples.append(
            AnalysisSample(
                clip=_require_str(item, "clip"),
                frame=_require_str(item, "frame"),
                frame_index=int(item.get("frame_index", 0)),
                mask_type=mask_type,
                xyxy=_parse_xyxy(mask.get("xyxy", [])),
                mask_label=str(mask.get("label", "box")),
                reason=_require_str(item, "reason"),
            )
        )
    return samples


def build_sample_output_dir(out_root: str | Path, sample: AnalysisSample) -> Path:
    return Path(out_root) / "samples" / f"{sample.clip}_{sample.frame}"


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
