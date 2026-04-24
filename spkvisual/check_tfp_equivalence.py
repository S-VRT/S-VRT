#!/usr/bin/env python3
"""
Batch-check whether S-VRT TFP and original SpikeCV TFP are behaviorally equivalent.

It loads real spike .dat files, reconstructs frames through both implementations,
and reports whether outputs are pixel-identical.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spkvisual.debug_spike_reconstruction import (  # noqa: E402
    SPIKECV_ROOT,
    SVRTTFP,
    SpikeCVTFP,
    compare_frame_stacks,
    load_spikes,
    load_options,
    pick_dataset_cfg,
    reconstruct_tfp,
)

if str(SPIKECV_ROOT) not in sys.path:
    sys.path.insert(0, str(SPIKECV_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-check equivalence of S-VRT TFP vs SpikeCV TFP")
    parser.add_argument("--opt", type=Path, default=PROJECT_ROOT / "options" / "gopro_rgbspike_server.json")
    parser.add_argument("--dataset", choices=["train", "test", "both"], default="both")
    parser.add_argument("--half-win", type=int, default=20)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "spkvisual" / "debug_outputs" / "equivalence_report.json")
    parser.add_argument("clips", nargs="+", help="One or more .dat paths")
    return parser.parse_args()


def infer_dataset_name(path: Path) -> str:
    text = str(path)
    if "/train/" in text:
        return "train"
    if "/test/" in text:
        return "test"
    raise ValueError(f"Cannot infer dataset split from path: {path}")


def check_one(dat_path: Path, ds_cfg: Dict[str, Any], half_win: int, length: int, device: str, flipud: bool) -> Dict[str, Any]:
    spikes = load_spikes(dat_path, ds_cfg, begin=0, length=length, flipud=flipud)
    svrt_frames = reconstruct_tfp(SVRTTFP, spikes, ds_cfg, half_win, device)
    spikecv_frames = reconstruct_tfp(SpikeCVTFP, spikes, ds_cfg, half_win, device)
    comparison = compare_frame_stacks(svrt_frames, spikecv_frames)
    return {
        "dat_path": str(dat_path),
        "flipud": flipud,
        "spike_shape": list(spikes.shape),
        "svrt_shape": list(svrt_frames.shape),
        "spikecv_shape": list(spikecv_frames.shape),
        "comparison": comparison,
    }


def main() -> None:
    args = parse_args()
    opts = load_options(args.opt)
    results: List[Dict[str, Any]] = []

    for clip in args.clips:
        dat_path = Path(clip)
        dataset_name = infer_dataset_name(dat_path)
        if args.dataset != "both" and dataset_name != args.dataset:
            continue
        ds_cfg = pick_dataset_cfg(opts, dataset_name)
        for flipud in (True, False):
            result = check_one(dat_path, ds_cfg, args.half_win, args.length, args.device, flipud)
            results.append(result)
            cmp_info = result["comparison"]
            print(
                f"{dataset_name} | flipud={flipud} | {dat_path.name} | "
                f"identical={cmp_info.get('identical')} | max_abs_diff={cmp_info.get('max_abs_diff')}"
            )

    summary = {
        "num_cases": len(results),
        "all_identical": all(item["comparison"].get("identical", False) for item in results),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] cases={summary['num_cases']} all_identical={summary['all_identical']}")
    print(f"[summary] report={args.out}")


if __name__ == "__main__":
    main()
