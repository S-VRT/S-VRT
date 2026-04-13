from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from data.spike_recc import SpikeStream
from data.spike_recc.encoding25 import (
    build_centered_window,
    build_output_dir,
    compute_center_index,
    validate_center_bounds,
)


@dataclass
class EncodeResult:
    generated: int = 0
    skipped_existing: int = 0
    dry_run: int = 0
    failed: int = 0


def parse_meta_info(meta_info_file: Optional[Path]) -> Dict[str, int]:
    if meta_info_file is None:
        return {}
    mapping: Dict[str, int] = {}
    if not meta_info_file.exists():
        raise ValueError(f"meta_info_file not found: {meta_info_file}")
    with meta_info_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            clip, _frame_num, _shape, start_frame = parts[:4]
            mapping[clip] = int(start_frame)
    return mapping


def iter_frame_indices(spike_dir: Path) -> Iterable[Tuple[int, Path]]:
    for dat_path in sorted(spike_dir.glob("*.dat")):
        try:
            frame_idx = int(dat_path.stem)
        except ValueError:
            continue
        yield frame_idx, dat_path


def resolve_clip_start_frame(
    clip_name: str,
    frame_indices: List[int],
    clip_start_mapping: Dict[str, int],
) -> int:
    if clip_name in clip_start_mapping:
        return int(clip_start_mapping[clip_name])
    if not frame_indices:
        raise ValueError(f"No frame indices found for clip={clip_name}")
    return min(frame_indices)


def encode_one_frame(
    *,
    spike_matrix: np.ndarray,
    clip_name: str,
    frame_index: int,
    clip_start_frame: int,
    dt: int,
    center_offset: int,
    edge_margin: int,
) -> np.ndarray:
    center = compute_center_index(
        frame_index=frame_index,
        clip_start_frame=clip_start_frame,
        dt=dt,
        center_offset=center_offset,
    )
    validate_center_bounds(
        center=center,
        total_length=int(spike_matrix.shape[0]),
        edge_margin=edge_margin,
        clip=clip_name,
        frame=frame_index,
        dt=dt,
        center_offset=center_offset,
    )
    return build_centered_window(spike_matrix, center=center, length=25)


def process_clip(
    *,
    clip_dir: Path,
    dt: int,
    center_offset: int,
    edge_margin: int,
    dry_run: bool,
    overwrite: bool,
    clip_start_mapping: Dict[str, int],
    spike_h: int,
    spike_w: int,
) -> Tuple[EncodeResult, List[str]]:
    clip_name = clip_dir.name
    spike_dir = clip_dir / "spike"
    out_dir = build_output_dir(clip_dir, dt=dt)
    result = EncodeResult()
    missing: List[str] = []

    if not spike_dir.exists():
        missing.append(f"missing spike dir: {spike_dir}")
        return result, missing

    entries = list(iter_frame_indices(spike_dir))
    frame_indices = [idx for idx, _ in entries]
    if not entries:
        missing.append(f"no .dat frames: {spike_dir}")
        return result, missing

    clip_start_frame = resolve_clip_start_frame(clip_name, frame_indices, clip_start_mapping)

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, dat_path in entries:
        out_path = out_dir / f"{frame_idx:0{len(dat_path.stem)}d}.npy"
        if out_path.exists() and not overwrite:
            result.skipped_existing += 1
            continue

        if dry_run:
            result.dry_run += 1
            continue

        try:
            spike_stream = SpikeStream(
                offline=True,
                filepath=str(dat_path),
                spike_h=spike_h,
                spike_w=spike_w,
                print_dat_detail=False,
            )
            spike_matrix = spike_stream.get_spike_matrix(flipud=True)
            encoded = encode_one_frame(
                spike_matrix=spike_matrix,
                clip_name=clip_name,
                frame_index=frame_idx,
                clip_start_frame=clip_start_frame,
                dt=dt,
                center_offset=center_offset,
                edge_margin=edge_margin,
            )
            np.save(out_path, encoded.astype(np.float32))
            result.generated += 1
        except Exception as exc:
            result.failed += 1
            missing.append(f"clip={clip_name} frame={frame_idx} error={exc}")

    return result, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SCFlow strict encoding25 artifacts.")
    parser.add_argument("--spike-root", required=True, type=Path)
    parser.add_argument("--meta-info-file", type=Path, default=None)
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument("--center-offset", type=int, default=40)
    parser.add_argument("--edge-margin", type=int, default=40)
    parser.add_argument("--spike-h", type=int, default=360)
    parser.add_argument("--spike-w", type=int, default=640)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    spike_root = Path(args.spike_root)
    if not spike_root.exists():
        raise ValueError(f"spike root not found: {spike_root}")
    if args.dt <= 0:
        raise ValueError(f"--dt must be > 0, got {args.dt}")

    clip_start_mapping = parse_meta_info(args.meta_info_file)

    all_result = EncodeResult()
    all_missing: List[str] = []

    clip_dirs = [p for p in sorted(spike_root.iterdir()) if p.is_dir()]
    for clip_dir in clip_dirs:
        result, missing = process_clip(
            clip_dir=clip_dir,
            dt=args.dt,
            center_offset=args.center_offset,
            edge_margin=args.edge_margin,
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
            clip_start_mapping=clip_start_mapping,
            spike_h=int(args.spike_h),
            spike_w=int(args.spike_w),
        )
        all_result.generated += result.generated
        all_result.skipped_existing += result.skipped_existing
        all_result.dry_run += result.dry_run
        all_result.failed += result.failed
        all_missing.extend(missing)

    print("[prepare_scflow_encoding25] Summary")
    print(f"  clips={len(clip_dirs)}")
    print(f"  generated={all_result.generated}")
    print(f"  skipped_existing={all_result.skipped_existing}")
    print(f"  dry_run={all_result.dry_run}")
    print(f"  failed={all_result.failed}")

    if all_missing:
        print("[prepare_scflow_encoding25] Missing/Errors:")
        for item in all_missing:
            print(f"  - {item}")

    if all_result.failed > 0 and not args.dry_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
