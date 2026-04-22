from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from models.utils.flow import flow_warp
from scripts.data_preparation.spike_flow.prepare_scflow_encoding25 import decode_spike_dat
from scripts.diagnostics.scflow_dataset_probe import (
    build_consecutive_pairs,
    flow_oob_ratio,
    iter_clips,
    load_frame_artifact,
)
from models.optical_flow.scflow.wrapper import SCFlowWrapper


def flatten_subframes(items: Sequence[np.ndarray], subframes: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for arr in items:
        if subframes > 1:
            out.extend([arr[i] for i in range(arr.shape[0])])
        else:
            out.append(arr)
    return out


def sample_clip_artifacts(clip_dir: Path, dt: int, subframes: int, frames: int) -> Tuple[List[str], List[np.ndarray]]:
    if subframes > 1:
        artifact_dir = clip_dir / f"encoding25_dt{dt}_s{subframes}"
    else:
        artifact_dir = clip_dir / f"encoding25_dt{dt}"
    frame_files = sorted(artifact_dir.glob("*.npy"))
    if len(frame_files) < frames:
        raise ValueError(f"Need at least {frames} artifacts in {artifact_dir}, found {len(frame_files)}")
    picked = frame_files[:frames]
    names = [p.stem for p in picked]
    arrays = [load_frame_artifact(clip_dir, name, dt=dt, subframes=subframes) for name in names]
    return names, arrays


def coarse_metrics(seq1: torch.Tensor, seq2: torch.Tensor, flow: torch.Tensor) -> Dict[str, float]:
    warped = flow_warp(seq1, flow.permute(0, 2, 3, 1).contiguous(), interp_mode="bilinear", padding_mode="zeros")
    pre_diff = (seq1 - seq2).abs()
    post_diff = (warped - seq2).abs()
    zero_mask = (warped.abs().sum(dim=1) == 0)

    improvement = (pre_diff.mean() - post_diff.mean()) / max(pre_diff.mean().item(), 1e-8)
    return {
        "pre_diff_mean": float(pre_diff.mean().item()),
        "post_diff_mean": float(post_diff.mean().item()),
        "relative_improvement": float(improvement.item()),
        "zero_fill_ratio": float(zero_mask.float().mean().item()),
        "flow_oob_ratio": flow_oob_ratio(flow),
    }


def probe_clip(
    *,
    clip_dir: Path,
    model: SCFlowWrapper,
    dt: int,
    subframes: int,
    frames: int,
    device: str,
) -> Dict[str, object]:
    frame_names, arrays = sample_clip_artifacts(clip_dir, dt=dt, subframes=subframes, frames=frames)
    windows = flatten_subframes(arrays, subframes=subframes)
    seq1, seq2 = build_consecutive_pairs(windows)
    flows = model(seq1.to(device), seq2.to(device))
    flow = flows[0].detach().cpu()
    metrics = coarse_metrics(seq1, seq2, flow)
    first_dat = clip_dir / "spike" / f"{frame_names[0]}.dat"
    spike_matrix = decode_spike_dat(first_dat, spike_h=360, spike_w=640, device="cpu", flipud=True)
    metrics["clip"] = clip_dir.name
    metrics["raw_t"] = int(spike_matrix.shape[0])
    metrics["frame_names"] = frame_names
    metrics["num_pairs"] = int(seq1.shape[0])
    return metrics


def summarize_group(reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    def collect(key: str) -> List[float]:
        return [float(report[key]) for report in reports]

    def stats(values: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    return {
        "num_clips": len(reports),
        "pre_diff_mean": stats(collect("pre_diff_mean")),
        "post_diff_mean": stats(collect("post_diff_mean")),
        "relative_improvement": stats(collect("relative_improvement")),
        "zero_fill_ratio": stats(collect("zero_fill_ratio")),
        "flow_oob_ratio": stats(collect("flow_oob_ratio")),
    }


def summarize_reports(reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not reports:
        return {"num_clips": 0}
    by_raw_t: Dict[str, List[Dict[str, object]]] = {}
    for report in reports:
        key = str(int(report["raw_t"]))
        by_raw_t.setdefault(key, []).append(report)

    return {
        "overall": summarize_group(reports),
        "by_raw_t": {key: summarize_group(group) for key, group in sorted(by_raw_t.items(), key=lambda item: int(item[0]))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe coarse warp quality for SCFlow flow on encoding25 artifacts.")
    parser.add_argument("--spike-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/optical_flow/dt10_e40.pth"))
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument("--subframes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = SCFlowWrapper(checkpoint=str(args.checkpoint), device=args.device, dt=args.dt)
    clip_dirs = list(iter_clips(args.spike_root))
    reports = [
        probe_clip(
            clip_dir=clip_dir,
            model=model,
            dt=args.dt,
            subframes=args.subframes,
            frames=args.frames,
            device=args.device,
        )
        for clip_dir in clip_dirs[: args.clips]
    ]

    print(json.dumps({"summary": summarize_reports(reports), "reports": reports}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
