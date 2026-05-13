#!/usr/bin/env python3
"""
Visualize spike .dat clip as an image sequence using SpikeCV.

Default config is read from `options/gopro_rgbspike_local.json` (test dataset).
If you do not pass a clip, it will pick the first test clip found under
`dataroot_spike` and drop PNG frames in `spkvisual/output_frames/<clip_name>`.
Supports TFP (时间窗口叠加) reconstruction to reduce flicker.

Usage examples:
  python spkvisual/visualize_spike_clip.py
  python spkvisual/visualize_spike_clip.py --clip /path/to/clip_dir_or_dat --out out_frames
  python spkvisual/visualize_spike_clip.py --dataset train --begin 100 --length 200
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Iterable

import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPT = PROJECT_ROOT / "options" / "gopro_rgbspike_local.json"

# Ensure SpikeCV importable
SPIKECV_ROOT = PROJECT_ROOT / "SpikeCV"
if str(SPIKECV_ROOT) not in sys.path:
    sys.path.insert(0, str(SPIKECV_ROOT))

from SpikeCV.spkData.load_dat import SpikeStream  # noqa: E402
from SpikeCV.spkProc.reconstruction.tfp import TFP  # noqa: E402


def strip_json_comments(text: str) -> str:
    """Remove // comments from json-like text."""
    cleaned_lines = []
    for line in text.splitlines():
        # simple removal; config does not contain // inside strings
        cleaned_lines.append(re.sub(r"//.*", "", line))
    return "\n".join(cleaned_lines)


def load_options(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    cleaned = strip_json_comments(content)
    return json.loads(cleaned)


def pick_dataset_cfg(opts: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    ds_all = opts.get("datasets", {})
    if dataset not in ds_all:
        raise KeyError(f"Dataset '{dataset}' not found in options")
    return ds_all[dataset]


def _iter_candidate_dat_files(base: Path, folder_name: Optional[str], ext: str) -> Iterable[Path]:
    """Yield candidate dat files given a dataset root."""
    if base.is_file() and base.suffix == f".{ext}":
        yield base
        return

    # If the clip folder contains a spike subfolder, search there
    if folder_name and (base / folder_name).exists():
        yield from sorted((base / folder_name).glob(f"*.{ext}"))
        return

    # Otherwise search .dat files directly under the folder
    yield from sorted(base.glob(f"*.{ext}"))

    # Walk one level deeper for typical dataset structure root/clip_name/spike/xxxx.dat
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        if folder_name and (sub / folder_name).exists():
            yield from sorted((sub / folder_name).glob(f"*.{ext}"))
        else:
            yield from sorted(sub.glob(f"*.{ext}"))


def resolve_clip_path(ds_cfg: Dict[str, Any], user_clip: Optional[str]) -> Path:
    base = Path(user_clip) if user_clip else Path(ds_cfg["dataroot_spike"])
    if base.is_file():
        return base

    ext = ds_cfg.get("spike_filename_ext", "dat")
    folder_name = ds_cfg.get("spike_folder_name")

    for file in _iter_candidate_dat_files(base, folder_name, ext):
        return file

    raise FileNotFoundError(f"No *.{ext} files found under {base}")


def load_spikes(dat_path: Path, ds_cfg: Dict[str, Any], begin: int, length: int) -> np.ndarray:
    stream = SpikeStream(
        filepath=str(dat_path),
        spike_h=ds_cfg["spike_h"],
        spike_w=ds_cfg["spike_w"],
        print_dat_detail=False,
    )
    flipud = bool(ds_cfg.get("spike_flipud", False))
    if length > 0:
        return stream.get_block_spikes(begin_idx=begin, block_len=length, flipud=flipud)
    return stream.get_spike_matrix(flipud=flipud)


def save_frames(spikes: np.ndarray, out_dir: Path) -> Tuple[int, Tuple[int, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    num, h, w = spikes.shape
    for idx in range(num):
        frame = (spikes[idx].astype(np.uint8)) * 255
        cv2.imwrite(str(out_dir / f"{idx:06d}.png"), frame)
    return num, (h, w)


def apply_colormap(frames: np.ndarray, cmap_name: str) -> np.ndarray:
    """Apply matplotlib colormap to grayscale frames; returns BGR uint8."""
    cmap = plt.get_cmap(cmap_name)
    colored = []
    for f in frames:
        norm = f.astype(np.float32)
        if norm.max() > 0:
            norm = norm / 255.0
        mapped = cmap(norm)[:, :, :3]  # RGBA -> RGB
        bgr = (mapped[:, :, ::-1] * 255).astype(np.uint8)
        colored.append(bgr)
    return np.stack(colored, axis=0)


def decay_trails(spikes: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential decay accumulation to show motion trails."""
    acc = np.zeros(spikes.shape[1:], dtype=np.float32)  # H x W
    out = np.zeros_like(spikes, dtype=np.uint8)
    for i in range(spikes.shape[0]):
        acc = acc * alpha + spikes[i].astype(np.float32)
        maxv = acc.max()
        frame = acc / (maxv + 1e-6) * 255.0 if maxv > 0 else acc
        out[i] = frame.astype(np.uint8)
    return out


def window_sum(spikes: np.ndarray, win: int) -> np.ndarray:
    """Simple sliding window sum (box filter) to show density."""
    T = spikes.shape[0]
    if win <= 0 or win > T:
        return spikes.copy()
    out_len = T - win + 1
    out = np.zeros((out_len, spikes.shape[1], spikes.shape[2]), dtype=np.uint8)
    for i in range(out_len):
        window = spikes[i:i + win].sum(axis=0)
        frame = window / (window.max() + 1e-6) * 255.0 if window.max() > 0 else window
        out[i] = frame.astype(np.uint8)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize spike .dat clip to image sequence")
    parser.add_argument("--opt", type=Path, default=DEFAULT_OPT, help="Path to options JSON")
    parser.add_argument("--dataset", choices=["train", "test"], default="test", help="Dataset section to read")
    parser.add_argument("--clip", type=str, default=None,
                        help="Clip path (folder or .dat file). Defaults to first clip under dataroot_spike")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for image sequence. Defaults to spkvisual/output_frames/<clip_name>")
    parser.add_argument("--begin", type=int, default=0, help="Start frame index")
    parser.add_argument("--length", type=int, default=-1, help="Number of frames to export (-1 for all)")
    parser.add_argument("--tfp", action="store_true", default=True,
                        help="Use TFP windowed accumulation (default: on)")
    parser.add_argument("--tfp_half_win", type=int, default=None,
                        help="Half window length for TFP (default: read from options, else 20)")
    parser.add_argument("--save_raw", action="store_true",
                        help="Also save raw binary frames alongside TFP")
    parser.add_argument("--cmap", type=str, default="turbo",
                        help="Matplotlib colormap name for pseudo-color (e.g., turbo, inferno)")
    parser.add_argument("--window_sum", type=int, default=0,
                        help="Sliding window size for density sum (0 to disable)")
    parser.add_argument("--decay_alpha", type=float, default=0.9,
                        help="Exponential decay factor for motion trails (0-1, closer to 1 keeps longer)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opts = load_options(args.opt)
    ds_cfg = pick_dataset_cfg(opts, args.dataset)

    dat_path = resolve_clip_path(ds_cfg, args.clip)
    print(f"[info] using clip: {dat_path}")

    default_out = PROJECT_ROOT / "spkvisual" / "output_frames" / dat_path.stem
    out_dir = args.out or default_out

    spikes = load_spikes(dat_path, ds_cfg, args.begin, args.length)

    if args.tfp:
        half_win = args.tfp_half_win or int(ds_cfg.get("tfp_half_win_length", 20))
        device = ds_cfg.get("tfp_device", "cpu")
        tfp = TFP(spike_h=ds_cfg["spike_h"], spike_w=ds_cfg["spike_w"], device=device)
        tfp_imgs = tfp.spikes2images(np.ascontiguousarray(spikes), half_win)
        tfp_dir = out_dir / "tfp"
        total_tfp, (h, w) = save_frames(tfp_imgs, tfp_dir)
        print(f"[done] TFP saved {total_tfp} frames at {w}x{h} to {tfp_dir} (half_win={half_win}, device={device})")
    else:
        total_tfp = 0
        h = spikes.shape[1]
        w = spikes.shape[2]

    if args.save_raw or not args.tfp:
        raw_dir = out_dir / "raw"
        total_raw, _ = save_frames(spikes, raw_dir)
        print(f"[info] raw saved {total_raw} frames at {w}x{h} to {raw_dir}")

    # Sliding window density sum
    if args.window_sum and args.window_sum > 0:
        win_imgs = window_sum(spikes, args.window_sum)
        win_dir = out_dir / f"window_sum_{args.window_sum}"
        total_win, _ = save_frames(win_imgs, win_dir)
        print(f"[info] window-sum({args.window_sum}) saved {total_win} frames to {win_dir}")

    # Exponential decay trails
    if args.decay_alpha is not None and 0 < args.decay_alpha < 1:
        decay_imgs = decay_trails(spikes, args.decay_alpha)
        decay_dir = out_dir / f"decay_{args.decay_alpha}"
        total_decay, _ = save_frames(decay_imgs, decay_dir)
        print(f"[info] decay(alpha={args.decay_alpha}) saved {total_decay} frames to {decay_dir}")

    # Pseudo color on TFP (preferred) else on raw
    base_for_cmap = tfp_imgs if args.tfp else spikes.astype(np.uint8) * 255
    cmap_imgs = apply_colormap(base_for_cmap, args.cmap)
    cmap_dir = out_dir / f"cmap_{args.cmap}"
    cmap_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(cmap_imgs):
        cv2.imwrite(str(cmap_dir / f"{idx:06d}.png"), frame)
    print(f"[info] cmap({args.cmap}) saved {len(cmap_imgs)} frames to {cmap_dir}")

    if args.tfp:
        print(f"[done] output dir: {out_dir}")


if __name__ == "__main__":
    main()

