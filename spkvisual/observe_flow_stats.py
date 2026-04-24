#!/usr/bin/env python3
"""Compare SpyNet and SCFlow flow statistics on real RGB/spike frame pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # Keep --help usable on machines without the runtime env.
    np = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPT = PROJECT_ROOT / "options" / "gopro_rgbspike_server.json"
DEFAULT_SPYNET_CKPT = PROJECT_ROOT / "weights" / "optical_flow" / "spynet" / "spynet_sintel_final-3d2a1287.pth"
DEFAULT_SCFLOW_CKPT = PROJECT_ROOT / "weights" / "optical_flow" / "dt10_e40.pth"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def strip_json_comments(json_text: str) -> str:
    """Strip // comments while preserving // inside quoted strings."""
    out: List[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(json_text):
        ch = json_text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < len(json_text) and json_text[i + 1] == "/":
            while i < len(json_text) and json_text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_options(path: Path) -> Dict[str, Any]:
    return json.loads(strip_json_comments(Path(path).read_text(encoding="utf-8")))


def summarize_values(values: np.ndarray, low_threshold: Optional[float] = None) -> Dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        empty = {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
        }
        if low_threshold is not None:
            empty["low_threshold"] = float(low_threshold)
            empty["low_ratio"] = None
        return empty

    pct = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])
    result: Dict[str, Any] = {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p01": float(pct[0]),
        "p05": float(pct[1]),
        "p25": float(pct[2]),
        "p50": float(pct[3]),
        "p75": float(pct[4]),
        "p95": float(pct[5]),
        "p99": float(pct[6]),
    }
    if low_threshold is not None:
        result["low_threshold"] = float(low_threshold)
        result["low_ratio"] = float(np.mean(flat < float(low_threshold)))
    return result


def _flow_mag(flow_chw: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow_chw, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"flow must be [2,H,W], got shape={tuple(flow.shape)}")
    return np.sqrt(np.sum(flow * flow, axis=0))


def resize_flow_to(flow_chw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    flow = np.asarray(flow_chw, dtype=np.float32)
    src_h, src_w = flow.shape[1:]
    if (src_h, src_w) == (target_h, target_w):
        return flow.copy()
    import cv2

    u = cv2.resize(flow[0], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(flow[1], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    u = u * (float(target_w) / float(src_w))
    v = v * (float(target_h) / float(src_h))
    return np.stack([u, v], axis=0).astype(np.float32)


def select_subframe(arr: np.ndarray, selector: str) -> Tuple[np.ndarray, str]:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 3:
        if data.shape[0] != 25:
            raise ValueError(f"encoding25 tensor must be [25,H,W], got shape={tuple(data.shape)}")
        return data, "single"
    if data.ndim != 4 or data.shape[1] != 25:
        raise ValueError(f"subframe encoding25 tensor must be [S,25,H,W], got shape={tuple(data.shape)}")

    normalized = str(selector).strip().lower()
    if normalized == "middle":
        idx = data.shape[0] // 2
        return data[idx], f"middle:{idx}"
    if normalized == "mean":
        return data.mean(axis=0), "mean"
    if re.fullmatch(r"\d+", normalized):
        idx = int(normalized)
        if idx < 0 or idx >= data.shape[0]:
            raise ValueError(f"subframe index {idx} out of range for S={data.shape[0]}")
        return data[idx], f"index:{idx}"
    raise ValueError("subframe selector must be 'middle', 'mean', or a zero-based integer index")


def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask.shape == (target_h, target_w):
        return mask.astype(bool)
    import cv2

    resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _active_inactive_summary(
    flow_mag: np.ndarray,
    spike_window: np.ndarray,
    active_threshold: float,
) -> Dict[str, Any]:
    activity = np.asarray(spike_window, dtype=np.float32).sum(axis=0)
    mask = _resize_mask(activity > float(active_threshold), flow_mag.shape[0], flow_mag.shape[1])
    active = flow_mag[mask]
    inactive = flow_mag[~mask]
    return {
        "activity_threshold": float(active_threshold),
        "active_pixel_ratio": float(mask.mean()) if mask.size else None,
        "active": summarize_values(active),
        "inactive": summarize_values(inactive),
    }


def analyze_pair_arrays(
    *,
    key: str,
    next_key: str,
    spynet_flow: np.ndarray,
    scflow_flow: np.ndarray,
    spike_window: np.ndarray,
    low_flow_threshold: float,
    active_threshold: float,
) -> Dict[str, Any]:
    scflow_mag = _flow_mag(scflow_flow)
    spynet_resized = resize_flow_to(spynet_flow, scflow_flow.shape[1], scflow_flow.shape[2])
    spynet_mag = _flow_mag(spynet_resized)
    diff = np.asarray(scflow_flow, dtype=np.float32) - spynet_resized
    diff_mag = _flow_mag(diff)

    denom = np.maximum(scflow_mag * spynet_mag, 1e-12)
    cosine = np.sum(np.asarray(scflow_flow) * spynet_resized, axis=0) / denom
    valid_cosine = cosine[(scflow_mag > 1e-6) & (spynet_mag > 1e-6)]

    return {
        "key": key,
        "next_key": next_key,
        "spynet": {
            "mag": summarize_values(spynet_mag, low_threshold=low_flow_threshold),
            "u": summarize_values(spynet_resized[0]),
            "v": summarize_values(spynet_resized[1]),
            "active_spike_regions": _active_inactive_summary(spynet_mag, spike_window, active_threshold),
        },
        "scflow": {
            "mag": summarize_values(scflow_mag, low_threshold=low_flow_threshold),
            "u": summarize_values(scflow_flow[0]),
            "v": summarize_values(scflow_flow[1]),
            "active_spike_regions": _active_inactive_summary(scflow_mag, spike_window, active_threshold),
        },
        "diff": {
            "mag": summarize_values(diff_mag, low_threshold=low_flow_threshold),
            "u_signed": summarize_values(diff[0]),
            "v_signed": summarize_values(diff[1]),
            "cosine": summarize_values(valid_cosine),
        },
        "_arrays": {
            "spynet_mag": spynet_mag.reshape(-1),
            "scflow_mag": scflow_mag.reshape(-1),
            "diff_mag": diff_mag.reshape(-1),
            "spynet_u": spynet_resized[0].reshape(-1),
            "spynet_v": spynet_resized[1].reshape(-1),
            "scflow_u": np.asarray(scflow_flow[0]).reshape(-1),
            "scflow_v": np.asarray(scflow_flow[1]).reshape(-1),
        },
    }


def _concat_hist_values(histograms: Mapping[str, Sequence[float]], key: str) -> np.ndarray:
    return np.asarray(histograms.get(key, []), dtype=np.float32).reshape(-1)


def build_run_summary(
    *,
    config: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    histograms: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    aggregate = {
        "spynet": {
            "mag": summarize_values(_concat_hist_values(histograms, "spynet_mag"), config.get("low_flow_threshold")),
            "u": summarize_values(_concat_hist_values(histograms, "spynet_u")),
            "v": summarize_values(_concat_hist_values(histograms, "spynet_v")),
        },
        "scflow": {
            "mag": summarize_values(_concat_hist_values(histograms, "scflow_mag"), config.get("low_flow_threshold")),
            "u": summarize_values(_concat_hist_values(histograms, "scflow_u")),
            "v": summarize_values(_concat_hist_values(histograms, "scflow_v")),
        },
        "diff": {
            "mag": summarize_values(_concat_hist_values(histograms, "diff_mag"), config.get("low_flow_threshold")),
        },
    }
    for model_name in ("spynet", "scflow"):
        active_items = [row[model_name]["active_spike_regions"] for row in pair_rows if model_name in row]
        if active_items:
            aggregate[model_name]["active_spike_regions"] = {
                "mean_active_pixel_ratio": float(np.mean([x["active_pixel_ratio"] for x in active_items])),
                "per_pair": active_items,
            }

    console_conclusions = {
        "spynet_magnitude": aggregate["spynet"]["mag"],
        "scflow_magnitude": aggregate["scflow"]["mag"],
        "scflow_vs_spynet_difference": aggregate["diff"]["mag"],
        "spynet_active_vs_inactive_spike_regions": aggregate["spynet"].get("active_spike_regions"),
        "scflow_active_vs_inactive_spike_regions": aggregate["scflow"].get("active_spike_regions"),
    }
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": dict(config),
        "num_pairs": len(pair_rows),
        "aggregate": aggregate,
        "console_conclusions": console_conclusions,
        "pairs": [_without_arrays(row) for row in pair_rows],
    }


def _without_arrays(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k != "_arrays"}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _flatten_row(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _flatten_row(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = value


def write_pair_csv(path: Path, pair_rows: Sequence[Mapping[str, Any]]) -> None:
    flat_rows: List[Dict[str, Any]] = []
    for row in pair_rows:
        flat: Dict[str, Any] = {}
        _flatten_row("", _without_arrays(row), flat)
        flat_rows.append(flat)
    if not flat_rows:
        return
    fieldnames = sorted({key for row in flat_rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_active_region_csv(path: Path, pair_rows: Sequence[Mapping[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for row in pair_rows:
        for model_name in ("spynet", "scflow"):
            region = row[model_name]["active_spike_regions"]
            rows.append(
                {
                    "key": row["key"],
                    "next_key": row["next_key"],
                    "model": model_name,
                    "active_pixel_ratio": region["active_pixel_ratio"],
                    "active_mean": region["active"]["mean"],
                    "active_p50": region["active"]["p50"],
                    "active_p95": region["active"]["p95"],
                    "inactive_mean": region["inactive"]["mean"],
                    "inactive_p50": region["inactive"]["p50"],
                    "inactive_p95": region["inactive"]["p95"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "key",
            "next_key",
            "model",
            "active_pixel_ratio",
            "active_mean",
            "active_p50",
            "active_p95",
            "inactive_mean",
            "inactive_p50",
            "inactive_p95",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_histograms(out_dir: Path, histograms: Mapping[str, Sequence[float]]) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    def save_hist(path: Path, items: Sequence[Tuple[str, np.ndarray]], bins: int = 80) -> None:
        plt.figure(figsize=(8, 5))
        for label, values in items:
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            if arr.size:
                plt.hist(arr, bins=bins, alpha=0.55, label=label)
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    mag_path = out_dir / "hist_flow_mag.png"
    save_hist(
        mag_path,
        [
            ("SpyNet |flow|", _concat_hist_values(histograms, "spynet_mag")),
            ("SCFlow |flow|", _concat_hist_values(histograms, "scflow_mag")),
        ],
    )
    outputs["hist_flow_mag"] = str(mag_path)

    uv_path = out_dir / "hist_u_v.png"
    save_hist(
        uv_path,
        [
            ("SpyNet u", _concat_hist_values(histograms, "spynet_u")),
            ("SpyNet v", _concat_hist_values(histograms, "spynet_v")),
            ("SCFlow u", _concat_hist_values(histograms, "scflow_u")),
            ("SCFlow v", _concat_hist_values(histograms, "scflow_v")),
        ],
    )
    outputs["hist_u_v"] = str(uv_path)

    diff_path = out_dir / "diff_mag_hist.png"
    save_hist(diff_path, [("SCFlow - SpyNet |diff|", _concat_hist_values(histograms, "diff_mag"))])
    outputs["diff_mag_hist"] = str(diff_path)
    return outputs


def _parse_meta_pairs(ds_cfg: Mapping[str, Any], num_pairs: int, start_index: int) -> List[Tuple[str, str, int, int]]:
    filename_tmpl = ds_cfg.get("filename_tmpl", "08d")
    pairs: List[Tuple[str, str, int, int]] = []
    meta_path = Path(ds_cfg["meta_info_file"])
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        folder, frame_num, _shape, start_frame = line.split(" ", 3)
        begin = int(start_frame)
        end = begin + int(frame_num) - 1
        for idx in range(begin, end):
            key = f"{folder}/{idx:{filename_tmpl}}"
            next_key = f"{folder}/{idx + 1:{filename_tmpl}}"
            pairs.append((key, next_key, idx, idx + 1))
    return pairs[int(start_index): int(start_index) + int(num_pairs)]


def _read_rgb_frame(ds_cfg: Mapping[str, Any], clip_name: str, frame_idx: int) -> np.ndarray:
    import cv2
    from utils import utils_video

    root = Path(ds_cfg["dataroot_lq"])
    filename_tmpl = ds_cfg.get("filename_tmpl", "08d")
    ext = ds_cfg.get("filename_ext", "png")
    rel_key = f"{clip_name}/{frame_idx:{filename_tmpl}}"
    disk_path = root / clip_name / f"{frame_idx:{filename_tmpl}}.{ext}"

    backend = str(ds_cfg.get("io_backend", {}).get("type", "disk")).lower()
    if backend == "lmdb" and "db_paths" in ds_cfg.get("io_backend", {}):
        io_opt = dict(ds_cfg["io_backend"])
        io_type = io_opt.pop("type")
        client = utils_video.FileClient(io_type, **io_opt)
        content = client.get(rel_key, "lq")
    else:
        if not disk_path.exists():
            raise FileNotFoundError(f"Missing RGB frame: {disk_path}")
        content = disk_path.read_bytes()

    img_bgr = utils_video.imfrombytes(content, float32=True)
    if img_bgr is None:
        raise ValueError(f"Failed to decode RGB frame: {disk_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_flow_spike(ds_cfg: Mapping[str, Any], clip_name: str, frame_idx: int, selector: str) -> Tuple[np.ndarray, str, Path]:
    from data.spike_recc.encoding25 import load_encoding25_artifact_with_shape

    spike_flow = ds_cfg.get("spike_flow", {})
    dt = int(spike_flow.get("dt", 10))
    subframes = int(spike_flow.get("subframes", 1))
    artifact_format = str(spike_flow.get("format", "auto"))
    flow_root = Path(ds_cfg["dataroot_spike"]) if str(spike_flow.get("root", "auto")).lower() == "auto" else Path(spike_flow["root"])
    dir_name = f"encoding25_dt{dt}_s{subframes}" if subframes > 1 else f"encoding25_dt{dt}"
    filename_tmpl = ds_cfg.get("filename_tmpl", "08d")
    base_path = flow_root / clip_name / dir_name / f"{frame_idx:{filename_tmpl}}"
    try:
        arr = load_encoding25_artifact_with_shape(
            base_path,
            artifact_format=artifact_format,
            num_subframes=subframes,
            spike_h=int(ds_cfg["spike_h"]),
            spike_w=int(ds_cfg["spike_w"]),
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing encoding25 artifact: {base_path}.npy or .dat") from exc
    selected, label = select_subframe(arr, selector)
    return selected, label, base_path


def _to_tensor_pair(frame1: np.ndarray, frame2: np.ndarray, device: str):
    import torch

    t1 = torch.from_numpy(frame1.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    t2 = torch.from_numpy(frame2.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return t1, t2


def _spike_to_tensor(spike: np.ndarray, device: str):
    import torch

    return torch.from_numpy(np.asarray(spike, dtype=np.float32)).unsqueeze(0).float().to(device)


def _flow_output_to_numpy(output: Any) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        output = output[0]
    arr = output.detach().float().cpu().numpy()
    if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[1] != 2:
        raise ValueError(f"Expected model flow [1,2,H,W], got shape={tuple(arr.shape)}")
    return arr[0]


def _append_histograms(histograms: Dict[str, List[float]], row: Mapping[str, Any]) -> None:
    for key, arr in row["_arrays"].items():
        histograms.setdefault(key, []).extend(np.asarray(arr, dtype=np.float32).reshape(-1).tolist())


def _print_console(summary: Mapping[str, Any], out_dir: Path) -> None:
    conclusions = summary["console_conclusions"]

    def brief(name: str, item: Mapping[str, Any]) -> None:
        print(
            f"{name}: mean={item['mean']:.4f} p50={item['p50']:.4f} "
            f"p95={item['p95']:.4f} max={item['max']:.4f} low_ratio={item.get('low_ratio')}",
            flush=True,
        )

    print("[flow-observe] persisted conclusions:", out_dir / "summary.json", flush=True)
    brief("SpyNet |flow|", conclusions["spynet_magnitude"])
    brief("SCFlow |flow|", conclusions["scflow_magnitude"])
    brief("SCFlow-SpyNet |diff|", conclusions["scflow_vs_spynet_difference"])
    active = conclusions.get("scflow_active_vs_inactive_spike_regions")
    if active:
        print(f"SCFlow active_pixel_ratio_mean={active['mean_active_pixel_ratio']:.4f}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, default=DEFAULT_OPT)
    parser.add_argument("--dataset", choices=["train", "test"], default="train")
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Default: cuda if available else cpu")
    parser.add_argument("--spynet-ckpt", type=Path, default=DEFAULT_SPYNET_CKPT)
    parser.add_argument("--scflow-ckpt", type=Path, default=DEFAULT_SCFLOW_CKPT)
    parser.add_argument("--subframe", type=str, default="middle", help="'middle', 'mean', or zero-based index")
    parser.add_argument("--low-flow-threshold", type=float, default=0.5)
    parser.add_argument("--active-threshold", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if np is None:
        raise ModuleNotFoundError("NumPy is required to run flow observation. Install the project runtime environment first.")

    import torch
    from models.optical_flow import create_optical_flow
    opts = load_options(args.opt)
    ds_cfg = opts["datasets"][args.dataset]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out or PROJECT_ROOT / "spkvisual" / "flow_observations" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    spike_flow = ds_cfg.get("spike_flow", {})
    dt = int(spike_flow.get("dt", 10))
    pairs = _parse_meta_pairs(ds_cfg, args.num_pairs, args.start_index)
    if not pairs:
        raise ValueError("No adjacent pairs selected. Check --num-pairs, --start-index, and meta_info_file.")

    print(f"[flow-observe] loading models on {device}", flush=True)
    spynet = create_optical_flow("spynet", checkpoint=str(args.spynet_ckpt), device=device)
    scflow = create_optical_flow("scflow", checkpoint=str(args.scflow_ckpt), device=device, dt=dt)
    spynet.eval()
    scflow.eval()

    pair_rows: List[Dict[str, Any]] = []
    histograms: Dict[str, List[float]] = {}
    for pair_idx, (key, next_key, frame_idx, next_frame_idx) in enumerate(pairs):
        clip_name = key.split("/")[0]
        print(f"[flow-observe] pair {pair_idx + 1}/{len(pairs)} {key}->{next_key}", flush=True)
        rgb1 = _read_rgb_frame(ds_cfg, clip_name, frame_idx)
        rgb2 = _read_rgb_frame(ds_cfg, clip_name, next_frame_idx)
        spike1, selector_label, spike_path1 = _load_flow_spike(ds_cfg, clip_name, frame_idx, args.subframe)
        spike2, _, spike_path2 = _load_flow_spike(ds_cfg, clip_name, next_frame_idx, args.subframe)

        with torch.no_grad():
            rgb_t1, rgb_t2 = _to_tensor_pair(rgb1, rgb2, device)
            spike_t1 = _spike_to_tensor(spike1, device)
            spike_t2 = _spike_to_tensor(spike2, device)
            spynet_flow = _flow_output_to_numpy(spynet(rgb_t1, rgb_t2))
            scflow_flow = _flow_output_to_numpy(scflow(spike_t1, spike_t2))

        row = analyze_pair_arrays(
            key=key,
            next_key=next_key,
            spynet_flow=spynet_flow,
            scflow_flow=scflow_flow,
            spike_window=spike1,
            low_flow_threshold=args.low_flow_threshold,
            active_threshold=args.active_threshold,
        )
        row["rgb1"] = str(Path(ds_cfg["dataroot_lq"]) / clip_name / f"{frame_idx:{ds_cfg.get('filename_tmpl', '08d')}}.{ds_cfg.get('filename_ext', 'png')}")
        row["rgb2"] = str(Path(ds_cfg["dataroot_lq"]) / clip_name / f"{next_frame_idx:{ds_cfg.get('filename_tmpl', '08d')}}.{ds_cfg.get('filename_ext', 'png')}")
        row["spike1"] = str(spike_path1)
        row["spike2"] = str(spike_path2)
        row["subframe_selector"] = selector_label
        pair_rows.append(row)
        _append_histograms(histograms, row)

    hist_paths = write_histograms(out_dir, histograms)
    config = {
        "opt": str(args.opt),
        "dataset": args.dataset,
        "num_pairs": args.num_pairs,
        "start_index": args.start_index,
        "device": device,
        "spynet_ckpt": str(args.spynet_ckpt),
        "scflow_ckpt": str(args.scflow_ckpt),
        "subframe": args.subframe,
        "low_flow_threshold": args.low_flow_threshold,
        "active_threshold": args.active_threshold,
        "histograms": hist_paths,
    }
    summary = build_run_summary(config=config, pair_rows=pair_rows, histograms=histograms)
    write_json(out_dir / "summary.json", summary)
    write_pair_csv(out_dir / "per_pair.csv", pair_rows)
    write_active_region_csv(out_dir / "active_vs_inactive_spike.csv", pair_rows)
    _print_console(summary, out_dir)


if __name__ == "__main__":
    main()
