#!/usr/bin/env python3
"""
Verify how many spike frames are inside a .dat file.

Default: use the first test clip specified by options/gopro_rgbspike_local.json,
print T (number of spike frames) and shape.

Usage:
  python spkvisual/verify_dat_frames.py
  python spkvisual/verify_dat_frames.py --clip /path/to/000001.dat
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPT = PROJECT_ROOT / "options" / "gopro_rgbspike_local.json"

SPIKECV_ROOT = PROJECT_ROOT / "SpikeCV"
if str(SPIKECV_ROOT) not in sys.path:
    sys.path.insert(0, str(SPIKECV_ROOT))

from SpikeCV.spkData.load_dat import SpikeStream  # noqa: E402


def strip_json_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r"//.*", "", line))
    return "\n".join(lines)


def load_options(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    return json.loads(strip_json_comments(content))


def pick_dataset_cfg(opts: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    ds_all = opts.get("datasets", {})
    if dataset not in ds_all:
        raise KeyError(f"Dataset '{dataset}' not found")
    return ds_all[dataset]


def resolve_clip_path(ds_cfg: Dict[str, Any], user_clip: Optional[str]) -> Path:
    base = Path(user_clip) if user_clip else Path(ds_cfg["dataroot_spike"])
    if base.is_file():
        return base
    ext = ds_cfg.get("spike_filename_ext", "dat")
    folder = ds_cfg.get("spike_folder_name", "spike")
    # try base/spike/*.dat then base/*.dat, else scan one level deeper (base/*/spike/*.dat)
    cand_dir = base / folder if (base / folder).exists() else base
    files = sorted(cand_dir.glob(f"*.{ext}"))
    if files:
        return files[0]
    # one level deeper
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        deeper = sub / folder if (sub / folder).exists() else sub
        files = sorted(deeper.glob(f"*.{ext}"))
        if files:
            return files[0]
    raise FileNotFoundError(f"No .{ext} under {base} (and subfolders)")


def collect_dat_files(ds_cfg: Dict[str, Any], clip_path: Path) -> Dict[str, Path]:
    """Return dict name->path for all dat files under a clip folder or direct file."""
    ext = ds_cfg.get("spike_filename_ext", "dat")
    folder = ds_cfg.get("spike_folder_name", "spike")
    results = {}
    if clip_path.is_file() and clip_path.suffix == f".{ext}":
        results[clip_path.stem] = clip_path
        return results

    # Try clip/spike/*.dat then clip/*.dat
    cand_dirs = []
    if clip_path.is_dir():
        cand_dirs.append(clip_path / folder)
        cand_dirs.append(clip_path)
    for d in cand_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob(f"*.{ext}")):
            results[p.stem] = p
        if results:
            return results

    # One level deeper: clip/*/spike/*.dat
    if clip_path.is_dir():
        for sub in sorted(clip_path.iterdir()):
            if not sub.is_dir():
                continue
            deeper = sub / folder if (sub / folder).exists() else sub
            for p in sorted(deeper.glob(f"*.{ext}")):
                results[p.stem] = p

    if not results:
        raise FileNotFoundError(f"No .{ext} found under {clip_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify frame count inside a spike .dat")
    parser.add_argument("--opt", type=Path, default=DEFAULT_OPT, help="options json")
    parser.add_argument("--dataset", choices=["train", "test"], default="test")
    parser.add_argument("--clip", type=str, default=None, help="clip dir or .dat path")
    args = parser.parse_args()

    opts = load_options(args.opt)
    ds_cfg = pick_dataset_cfg(opts, args.dataset)
    first_dat = resolve_clip_path(ds_cfg, args.clip)
    dat_map = collect_dat_files(ds_cfg, first_dat.parent if first_dat.is_file() else first_dat)
    print(f"[info] found {len(dat_map)} dat files under {first_dat.parent if first_dat.is_file() else first_dat}")

    for name, dat_path in dat_map.items():
        stream = SpikeStream(
            filepath=str(dat_path),
            spike_h=ds_cfg["spike_h"],
            spike_w=ds_cfg["spike_w"],
            print_dat_detail=False,
        )
        spikes = stream.get_spike_matrix(flipud=bool(ds_cfg.get("spike_flipud", False)))
        T, H, W = spikes.shape
        print(f"{name}: {T} x {H} x {W}")


if __name__ == "__main__":
    main()

