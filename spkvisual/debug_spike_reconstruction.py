#!/usr/bin/env python3
"""
Debug spike reconstruction for S-VRT without GUI popups.

This script follows the project's current dataset config, loads spike .dat clips,
reconstructs frames with the S-VRT TFP path, and writes results to disk.
It can also save raw binary spike frames and compare flipud on/off outputs.

Examples:
  python spkvisual/debug_spike_reconstruction.py
  python spkvisual/debug_spike_reconstruction.py --opt options/gopro_rgbspike_server.json --dataset train
  python spkvisual/debug_spike_reconstruction.py --clip /path/to/000001.dat --length 128 --save-raw
  python spkvisual/debug_spike_reconstruction.py --compare-flipud
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPT = PROJECT_ROOT / "options" / "gopro_rgbspike_server.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPIKECV_ROOT = PROJECT_ROOT / "SpikeCV"
if str(SPIKECV_ROOT) not in sys.path:
    sys.path.insert(0, str(SPIKECV_ROOT))

from SpikeCV.spkData.load_dat import SpikeStream  # noqa: E402
from SpikeCV.spkProc.reconstruction.tfp import TFP as SpikeCVTFP  # noqa: E402
from data.spike_recc.spikecv.reconstruction.tfp import TFP as SVRTTFP  # noqa: E402


def strip_json_comments(text: str) -> str:
    return "\n".join(re.sub(r"//.*", "", line) for line in text.splitlines())


def load_options(path: Path) -> Dict[str, Any]:
    return json.loads(strip_json_comments(path.read_text(encoding="utf-8")))


def pick_dataset_cfg(opts: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    datasets = opts.get("datasets", {})
    if dataset not in datasets:
        raise KeyError(f"Dataset '{dataset}' not found in options")
    return datasets[dataset]


def iter_candidate_dat_files(base: Path, folder_name: Optional[str], ext: str) -> Iterable[Path]:
    if base.is_file() and base.suffix == f".{ext}":
        yield base
        return

    if folder_name and (base / folder_name).exists():
        yield from sorted((base / folder_name).glob(f"*.{ext}"))
        return

    yield from sorted(base.glob(f"*.{ext}"))

    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        if folder_name and (sub / folder_name).exists():
            yield from sorted((sub / folder_name).glob(f"*.{ext}"))
        else:
            yield from sorted(sub.glob(f"*.{ext}"))


def resolve_clip_path(ds_cfg: Dict[str, Any], user_clip: Optional[str]) -> Path:
    base = Path(user_clip) if user_clip else Path(ds_cfg["dataroot_spike"])
    ext = ds_cfg.get("spike_filename_ext", "dat")
    folder_name = ds_cfg.get("spike_folder_name")

    for dat_path in iter_candidate_dat_files(base, folder_name, ext):
        return dat_path

    raise FileNotFoundError(f"No *.{ext} found under {base}")


def load_spikes(dat_path: Path, ds_cfg: Dict[str, Any], begin: int, length: int, flipud: bool) -> np.ndarray:
    stream = SpikeStream(
        filepath=str(dat_path),
        spike_h=ds_cfg["spike_h"],
        spike_w=ds_cfg["spike_w"],
        print_dat_detail=False,
    )
    spikes = stream.get_spike_matrix(flipud=flipud)
    if begin < 0:
        raise ValueError(f"begin must be >= 0, got {begin}")
    if begin >= spikes.shape[0]:
        raise ValueError(f"begin={begin} out of range for spike length {spikes.shape[0]}")
    if length > 0:
        end = min(begin + length, spikes.shape[0])
        spikes = spikes[begin:end]
    elif begin > 0:
        spikes = spikes[begin:]
    return np.ascontiguousarray(spikes)


def save_grayscale_frames(frames: np.ndarray, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{idx:06d}.png"), frame.astype(np.uint8))
    return int(frames.shape[0])


def compare_frame_stacks(frames_a: np.ndarray, frames_b: np.ndarray, diff_dir: Optional[Path] = None) -> Dict[str, Any]:
    if frames_a.shape != frames_b.shape:
        return {
            "same_shape": False,
            "shape_a": list(frames_a.shape),
            "shape_b": list(frames_b.shape),
            "identical": False,
        }

    diff = np.abs(frames_a.astype(np.int16) - frames_b.astype(np.int16))
    identical = bool(np.array_equal(frames_a, frames_b))
    nonzero_mask = diff > 0
    diff_frame_indices = np.where(nonzero_mask.reshape(nonzero_mask.shape[0], -1).any(axis=1))[0].tolist()

    if diff_dir is not None and not identical:
        diff_dir.mkdir(parents=True, exist_ok=True)
        diff_visual = np.clip(diff.astype(np.uint16) * 16, 0, 255).astype(np.uint8)
        save_grayscale_frames(diff_visual, diff_dir)

    return {
        "same_shape": True,
        "shape": list(frames_a.shape),
        "identical": identical,
        "num_diff_pixels": int(nonzero_mask.sum()),
        "num_diff_frames": len(diff_frame_indices),
        "diff_frame_indices": diff_frame_indices[:20],
        "max_abs_diff": int(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def reconstruct_tfp(tfp_cls, spikes: np.ndarray, ds_cfg: Dict[str, Any], half_win: int, device: str) -> np.ndarray:
    tfp = tfp_cls(spike_h=ds_cfg["spike_h"], spike_w=ds_cfg["spike_w"], device=device)
    return tfp.spikes2images(spikes, half_win)


def make_manifest(
    out_root: Path,
    dat_path: Path,
    ds_cfg: Dict[str, Any],
    begin: int,
    length: int,
    half_win: int,
    device: str,
    flipud: bool,
    spikes: np.ndarray,
    outputs: Dict[str, Dict[str, Any]],
) -> None:
    manifest = {
        "dat_path": str(dat_path),
        "out_root": str(out_root),
        "spike_shape": list(spikes.shape),
        "spike_flipud": flipud,
        "begin": begin,
        "length": length,
        "tfp_half_win_length": half_win,
        "tfp_device": device,
        "spike_h": ds_cfg["spike_h"],
        "spike_w": ds_cfg["spike_w"],
        "outputs": outputs,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug S-VRT spike reconstruction and save outputs to disk")
    parser.add_argument("--opt", type=Path, default=DEFAULT_OPT, help="Path to options JSON")
    parser.add_argument("--dataset", choices=["train", "test"], default="train", help="Dataset section to use")
    parser.add_argument("--clip", type=str, default=None, help="Clip dir or .dat path")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--begin", type=int, default=0, help="Start spike index")
    parser.add_argument("--length", type=int, default=-1, help="Number of spike frames to load; -1 means full clip")
    parser.add_argument("--tfp-half-win", type=int, default=None, help="TFP half window length override")
    parser.add_argument("--device", type=str, default=None, help="Torch device for TFP, default cpu")
    parser.add_argument("--save-raw", action="store_true", help="Also save raw binary spike frames")
    parser.add_argument("--compare-flipud", action="store_true", help="Export both flipud=true and flipud=false reconstructions")
    return parser.parse_args()


def run_one(
    dat_path: Path,
    ds_cfg: Dict[str, Any],
    out_root: Path,
    begin: int,
    length: int,
    half_win: int,
    device: str,
    flipud: bool,
    save_raw: bool,
) -> None:
    spikes = load_spikes(dat_path, ds_cfg, begin, length, flipud=flipud)
    output_specs = {
        "s_vrt_tfp": (SVRTTFP, np.ascontiguousarray(spikes)),
        "spikecv_tfp": (SpikeCVTFP, spikes),
    }
    outputs: Dict[str, Dict[str, Any]] = {}
    rendered_frames: Dict[str, np.ndarray] = {}
    for name, (tfp_cls, spikes_input) in output_specs.items():
        tfp_frames = reconstruct_tfp(tfp_cls, spikes_input, ds_cfg, half_win, device)
        tfp_dir = out_root / name
        save_grayscale_frames(tfp_frames, tfp_dir)
        rendered_frames[name] = tfp_frames
        outputs[name] = {
            "dir": str(tfp_dir),
            "shape": list(tfp_frames.shape),
        }

    if save_raw:
        raw_dir = out_root / "raw"
        raw_frames = (spikes.astype(np.uint8)) * 255
        save_grayscale_frames(raw_frames, raw_dir)
        outputs["raw"] = {
            "dir": str(raw_dir),
            "shape": list(raw_frames.shape),
        }

    comparison = compare_frame_stacks(
        rendered_frames["s_vrt_tfp"],
        rendered_frames["spikecv_tfp"],
        diff_dir=out_root / "diff_s_vrt_vs_spikecv",
    )
    outputs["comparison_s_vrt_vs_spikecv"] = comparison

    make_manifest(out_root, dat_path, ds_cfg, begin, length, half_win, device, flipud, spikes, outputs)
    print(
        f"[done] flipud={flipud} spikes={tuple(spikes.shape)} identical={comparison.get('identical')} "
        f"saved to {out_root}"
    )


def main() -> None:
    args = parse_args()
    opts = load_options(args.opt)
    ds_cfg = pick_dataset_cfg(opts, args.dataset)
    dat_path = resolve_clip_path(ds_cfg, args.clip)

    default_flipud = bool(ds_cfg.get("spike_flipud", True))
    half_win = args.tfp_half_win if args.tfp_half_win is not None else int(ds_cfg.get("tfp_half_win_length", 20))
    device = args.device or ds_cfg.get("tfp_device") or "cpu"

    base_out = args.out or (PROJECT_ROOT / "spkvisual" / "debug_outputs" / dat_path.stem)
    print(f"[info] dat_path={dat_path}")
    print(f"[info] out_root={base_out}")
    print(f"[info] default flipud from config={default_flipud}, half_win={half_win}, device={device}")

    if args.compare_flipud:
        run_one(dat_path, ds_cfg, base_out / "flipud_true", args.begin, args.length, half_win, device, True, args.save_raw)
        run_one(dat_path, ds_cfg, base_out / "flipud_false", args.begin, args.length, half_win, device, False, args.save_raw)
    else:
        run_one(dat_path, ds_cfg, base_out, args.begin, args.length, half_win, device, default_flipud, args.save_raw)


if __name__ == "__main__":
    main()
