from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from data.spike_recc.encoding25 import compute_subframe_centers
from models.optical_flow.scflow.wrapper import SCFlowWrapper
from scripts.data_preparation.spike_flow.prepare_scflow_encoding25 import decode_spike_dat


def iter_clips(spike_root: Path) -> Iterable[Path]:
    for item in sorted(spike_root.iterdir()):
        if item.is_dir():
            yield item


def load_frame_artifact(clip_dir: Path, frame_name: str, dt: int, subframes: int) -> np.ndarray:
    if subframes > 1:
        path = clip_dir / f"encoding25_dt{dt}_s{subframes}" / f"{frame_name}.npy"
    else:
        path = clip_dir / f"encoding25_dt{dt}" / f"{frame_name}.npy"
    arr = np.load(path).astype(np.float32)
    if subframes > 1 and arr.ndim != 4:
        raise ValueError(f"Expected [S,25,H,W] at {path}, got {arr.shape}")
    if subframes == 1 and arr.ndim != 3:
        raise ValueError(f"Expected [25,H,W] at {path}, got {arr.shape}")
    return arr


def flatten_subframes(items: Sequence[np.ndarray], subframes: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for arr in items:
        if subframes > 1:
            out.extend([arr[i] for i in range(arr.shape[0])])
        else:
            out.append(arr)
    return out


def build_consecutive_pairs(windows: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(windows) < 2:
        raise ValueError("Need at least 2 windows to build SCFlow pairs.")
    seq1 = torch.from_numpy(np.stack(windows[:-1], axis=0)).float()
    seq2 = torch.from_numpy(np.stack(windows[1:], axis=0)).float()
    return seq1, seq2


def flow_oob_ratio(flow: torch.Tensor) -> float:
    _, _, h, w = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    x2 = xx.unsqueeze(0) + flow[:, 0]
    y2 = yy.unsqueeze(0) + flow[:, 1]
    oob = (x2 < 0) | (x2 > (w - 1)) | (y2 < 0) | (y2 > (h - 1))
    return float(oob.float().mean().item())


def summarize_tensor(name: str, tensor: torch.Tensor) -> Dict[str, float]:
    flat = tensor.reshape(-1)
    return {
        f"{name}_mean": float(flat.mean().item()),
        f"{name}_std": float(flat.std().item()),
        f"{name}_abs_mean": float(flat.abs().mean().item()),
        f"{name}_abs_max": float(flat.abs().max().item()),
        f"{name}_p95": float(torch.quantile(flat, 0.95).item()),
    }


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


def probe_clip(
    *,
    clip_dir: Path,
    model: SCFlowWrapper,
    dt: int,
    subframes: int,
    frames: int,
    spike_h: int,
    spike_w: int,
    device: str,
) -> Dict[str, object]:
    frame_names, arrays = sample_clip_artifacts(clip_dir, dt=dt, subframes=subframes, frames=frames)
    windows = flatten_subframes(arrays, subframes=subframes)
    seq1, seq2 = build_consecutive_pairs(windows)
    flows = model(seq1.to(device), seq2.to(device))
    full_flow = flows[0].detach().cpu()
    flow_mag = torch.linalg.vector_norm(full_flow, dim=1)

    first_dat = clip_dir / "spike" / f"{frame_names[0]}.dat"
    spike_matrix = decode_spike_dat(first_dat, spike_h=spike_h, spike_w=spike_w, device="cpu", flipud=True)
    centers = compute_subframe_centers(spike_matrix.shape[0], num_subframes=subframes)
    sub_dt = None
    if len(centers) >= 2:
        diffs = np.diff(centers).astype(np.float32)
        sub_dt = float(diffs.mean())

    stats: Dict[str, object] = {
        "clip": clip_dir.name,
        "frame_names": frame_names,
        "raw_t": int(spike_matrix.shape[0]),
        "subframe_centers": [int(x) for x in centers],
        "sub_dt_mean": sub_dt,
        "artifact_shape": list(arrays[0].shape),
        "artifact_mean": float(arrays[0].mean()),
        "artifact_std": float(arrays[0].std()),
        "artifact_nonzero_ratio": float((arrays[0] > 0).mean()),
        "num_scflow_pairs": int(seq1.shape[0]),
        "flow_fullres_shape": list(full_flow.shape),
        "flow_oob_ratio": flow_oob_ratio(full_flow),
    }
    stats.update(summarize_tensor("flow_x", full_flow[:, 0]))
    stats.update(summarize_tensor("flow_y", full_flow[:, 1]))
    stats.update(summarize_tensor("flow_mag", flow_mag))
    return stats


def summarize_reports(reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not reports:
        return {"num_clips": 0}

    def collect_float(key: str) -> List[float]:
        values: List[float] = []
        for report in reports:
            value = report.get(key)
            if value is not None:
                values.append(float(value))
        return values

    def stats(values: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    summary = {
        "num_clips": len(reports),
        "sub_dt_mean": stats(collect_float("sub_dt_mean")),
        "artifact_nonzero_ratio": stats(collect_float("artifact_nonzero_ratio")),
        "flow_oob_ratio": stats(collect_float("flow_oob_ratio")),
        "flow_mag_mean": stats(collect_float("flow_mag_mean")),
        "flow_mag_p95": stats(collect_float("flow_mag_p95")),
        "flow_mag_abs_max": stats(collect_float("flow_mag_abs_max")),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe SCFlow checkpoint and encoding25 artifacts on GoPro spike data.")
    parser.add_argument("--spike-root", type=Path, required=True, help="Path to split root, e.g. .../GOPRO_Large_spike_seq/train")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/optical_flow/dt10_e40.pth"))
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument("--subframes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=3, help="Number of frame artifacts to sample per clip before flattening subframes.")
    parser.add_argument("--clips", type=int, default=2, help="How many clips to probe.")
    parser.add_argument("--spike-h", type=int, default=360)
    parser.add_argument("--spike-w", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    model = SCFlowWrapper(checkpoint=str(args.checkpoint), device=args.device, dt=args.dt)
    clip_dirs = list(iter_clips(args.spike_root))
    if not clip_dirs:
        raise ValueError(f"No clip directories under {args.spike_root}")

    reports: List[Dict[str, object]] = []
    for clip_dir in clip_dirs[: args.clips]:
        reports.append(
            probe_clip(
                clip_dir=clip_dir,
                model=model,
                dt=args.dt,
                subframes=args.subframes,
                frames=args.frames,
                spike_h=args.spike_h,
                spike_w=args.spike_w,
                device=args.device,
            )
        )

    print(
        json.dumps(
            {
                "summary": summarize_reports(reports),
                "reports": reports,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
