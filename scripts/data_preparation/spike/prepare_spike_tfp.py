from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.spike_recc import SpikeStream, voxelize_spikes_tfp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute spike TFP voxel artifacts for training/test datasets.")
    parser.add_argument("--spike-root", required=True, help="Root directory containing clip folders and spike/*.dat files.")
    parser.add_argument("--spike-h", type=int, default=360)
    parser.add_argument("--spike-w", type=int, default=640)
    parser.add_argument("--num-bins", type=int, default=4, help="Number of TFP output channels.")
    parser.add_argument("--half-win-length", type=int, default=20)
    parser.add_argument("--device", default="cpu", help="TFP reconstruction device, e.g. cpu or cuda:0.")
    parser.add_argument("--format", choices=["npy", "npz"], default="npy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clip", default=None, help="Optional single clip name to process.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of .dat files to process.")
    return parser.parse_args()


def iter_dat_files(spike_root: Path, clip_name: str | None):
    clip_dirs = [spike_root / clip_name] if clip_name else sorted(p for p in spike_root.iterdir() if p.is_dir())
    for clip_dir in clip_dirs:
        spike_dir = clip_dir / "spike"
        if not spike_dir.is_dir():
            continue
        for dat_path in sorted(spike_dir.glob("*.dat")):
            yield clip_dir.name, dat_path


def build_output_base(spike_root: Path, clip_name: str, num_bins: int, half_win_length: int, frame_stem: str) -> Path:
    out_dir = spike_root / clip_name / f"tfp_b{num_bins}_hw{half_win_length}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / frame_stem


def save_artifact(base_path: Path, spike_voxel: np.ndarray, artifact_format: str) -> Path:
    if artifact_format == "npy":
        out_path = base_path.with_suffix(".npy")
        np.save(out_path, spike_voxel.astype(np.float32))
        return out_path
    out_path = base_path.with_suffix(".npz")
    np.savez_compressed(out_path, spike_voxel=spike_voxel.astype(np.float32))
    return out_path


def main() -> None:
    args = parse_args()
    spike_root = Path(args.spike_root)
    processed = 0
    skipped = 0

    for clip_name, dat_path in iter_dat_files(spike_root, args.clip):
        frame_stem = dat_path.stem
        out_base = build_output_base(spike_root, clip_name, args.num_bins, args.half_win_length, frame_stem)
        out_path = out_base.with_suffix(f".{args.format}")
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        spike_stream = SpikeStream(
            offline=True,
            filepath=str(dat_path),
            spike_h=args.spike_h,
            spike_w=args.spike_w,
            print_dat_detail=False,
        )
        spike_matrix = spike_stream.get_spike_matrix(flipud=True)
        spike_voxel = voxelize_spikes_tfp(
            spike_matrix,
            num_channels=args.num_bins,
            device=args.device,
            half_win_length=args.half_win_length,
        )
        save_artifact(out_base, spike_voxel, args.format)
        processed += 1
        if processed % 100 == 0:
            print(f"[prepare_spike_tfp] processed={processed} skipped={skipped} last={dat_path}")
        if args.limit and processed >= args.limit:
            break

    print(f"[prepare_spike_tfp] done processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
