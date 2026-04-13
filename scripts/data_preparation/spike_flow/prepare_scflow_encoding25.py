from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _merge_result(dst: EncodeResult, src: EncodeResult) -> None:
    dst.generated += src.generated
    dst.skipped_existing += src.skipped_existing
    dst.dry_run += src.dry_run
    dst.failed += src.failed


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


def _process_clip_worker(args: Tuple[Path, int, int, int, bool, bool, Dict[str, int], int, int]) -> Tuple[str, EncodeResult, List[str]]:
    clip_dir, dt, center_offset, edge_margin, dry_run, overwrite, clip_start_mapping, spike_h, spike_w = args
    clip_name = clip_dir.name
    try:
        result, missing = process_clip(
            clip_dir=clip_dir,
            dt=dt,
            center_offset=center_offset,
            edge_margin=edge_margin,
            dry_run=dry_run,
            overwrite=overwrite,
            clip_start_mapping=clip_start_mapping,
            spike_h=spike_h,
            spike_w=spike_w,
        )
        return clip_name, result, missing
    except Exception as exc:
        failed = EncodeResult(failed=1)
        return clip_name, failed, [f"clip={clip_name} fatal_error={exc}"]


def process_all_clips(
    *,
    clip_dirs: List[Path],
    dt: int,
    center_offset: int,
    edge_margin: int,
    dry_run: bool,
    overwrite: bool,
    clip_start_mapping: Dict[str, int],
    spike_h: int,
    spike_w: int,
    num_workers: int,
) -> Tuple[EncodeResult, List[str]]:
    all_result = EncodeResult()
    all_missing: List[str] = []
    if not clip_dirs:
        return all_result, all_missing

    worker_count = min(max(1, int(num_workers)), len(clip_dirs))

    if worker_count == 1:
        for clip_dir in clip_dirs:
            result, missing = process_clip(
                clip_dir=clip_dir,
                dt=dt,
                center_offset=center_offset,
                edge_margin=edge_margin,
                dry_run=dry_run,
                overwrite=overwrite,
                clip_start_mapping=clip_start_mapping,
                spike_h=spike_h,
                spike_w=spike_w,
            )
            _merge_result(all_result, result)
            all_missing.extend(missing)
        return all_result, all_missing

    task_args = [
        (
            clip_dir,
            dt,
            center_offset,
            edge_margin,
            dry_run,
            overwrite,
            clip_start_mapping,
            spike_h,
            spike_w,
        )
        for clip_dir in clip_dirs
    ]

    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_process_clip_worker, item) for item in task_args]
        for future in as_completed(futures):
            clip_name, result, missing = future.result()
            _merge_result(all_result, result)
            if missing:
                all_missing.extend(missing)
            print(
                "[prepare_scflow_encoding25] clip_done "
                f"clip={clip_name} generated={result.generated} "
                f"skipped_existing={result.skipped_existing} dry_run={result.dry_run} failed={result.failed}"
            )

    return all_result, all_missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SCFlow strict encoding25 artifacts.")
    parser.add_argument("--spike-root", required=True, type=Path)
    parser.add_argument("--meta-info-file", type=Path, default=None)
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument("--center-offset", type=int, default=40)
    parser.add_argument("--edge-margin", type=int, default=40)
    parser.add_argument("--spike-h", type=int, default=360)
    parser.add_argument("--spike-w", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    spike_root = Path(args.spike_root)
    if not spike_root.exists():
        raise ValueError(f"spike root not found: {spike_root}")
    if args.dt <= 0:
        raise ValueError(f"--dt must be > 0, got {args.dt}")
    if args.num_workers <= 0:
        raise ValueError(f"--num-workers must be > 0, got {args.num_workers}")
    if args.max_clips < 0:
        raise ValueError(f"--max-clips must be >= 0, got {args.max_clips}")

    clip_start_mapping = parse_meta_info(args.meta_info_file)

    clip_dirs = [p for p in sorted(spike_root.iterdir()) if p.is_dir()]
    if args.max_clips > 0:
        clip_dirs = clip_dirs[: args.max_clips]

    requested_workers = int(args.num_workers)
    if requested_workers > 1 and len(clip_dirs) > 1:
        cpu_cnt = os.cpu_count() or requested_workers
        requested_workers = min(requested_workers, cpu_cnt)

    all_result, all_missing = process_all_clips(
        clip_dirs=clip_dirs,
        dt=int(args.dt),
        center_offset=int(args.center_offset),
        edge_margin=int(args.edge_margin),
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        clip_start_mapping=clip_start_mapping,
        spike_h=int(args.spike_h),
        spike_w=int(args.spike_w),
        num_workers=requested_workers,
    )

    print("[prepare_scflow_encoding25] Summary")
    print(f"  clips={len(clip_dirs)}")
    print(f"  workers={requested_workers}")
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
