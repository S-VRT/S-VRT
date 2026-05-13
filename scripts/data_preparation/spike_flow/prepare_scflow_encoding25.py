from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

from data.spike_recc.encoding25 import (
    build_output_dir,
    build_output_dir_subframes,
    compute_subframe_centers,
    build_centered_window,
    save_encoding25_artifact,
    validate_subframes_tensor,
    validate_encoding25_tensor,
)


@dataclass
class EncodeResult:
    generated: int = 0
    skipped_existing: int = 0
    dry_run: int = 0
    failed: int = 0
    exact25: int = 0
    cropped: int = 0
    padded: int = 0


@dataclass
class SpaceEstimate:
    dat_files: int
    bytes_per_file: int
    estimated_bytes: int
    free_bytes: int
    existing_output_bytes: int
    projected_additional_bytes: int


def bytes_per_artifact(
    *,
    spike_h: int,
    spike_w: int,
    num_subframes: int,
    artifact_format: str,
    npy_dtype: str,
) -> int:
    values = int(num_subframes) * 25 * int(spike_h) * int(spike_w)
    normalized = str(artifact_format).strip().lower()
    if normalized == "npy":
        npy_dtype = str(npy_dtype).strip().lower()
        if npy_dtype == "bool":
            return values * np.dtype(np.bool_).itemsize
        if npy_dtype == "float32":
            return values * np.dtype(np.float32).itemsize
        raise ValueError(f"Unsupported npy dtype: {npy_dtype!r}")
    if normalized == "dat":
        return (values + 7) // 8
    raise ValueError(f"Unsupported artifact format: {artifact_format!r}")


def _merge_result(dst: EncodeResult, src: EncodeResult) -> None:
    dst.generated += src.generated
    dst.skipped_existing += src.skipped_existing
    dst.dry_run += src.dry_run
    dst.failed += src.failed
    dst.exact25 += src.exact25
    dst.cropped += src.cropped
    dst.padded += src.padded


def iter_frame_indices(spike_dir: Path) -> Iterable[Tuple[int, Path]]:
    for dat_path in sorted(spike_dir.glob("*.dat")):
        try:
            frame_idx = int(dat_path.stem)
        except ValueError:
            continue
        yield frame_idx, dat_path


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(max(0, int(num_bytes)))
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def detect_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device=cuda requested but torch.cuda.is_available() is False")
    return device


def decode_spike_dat(
    dat_path: Path,
    spike_h: int,
    spike_w: int,
    *,
    device: str,
    flipud: bool = True,
) -> np.ndarray:
    """Decode packed 1-bit spike .dat into [T,H,W] bool matrix."""
    raw = np.fromfile(dat_path, dtype=np.uint8)
    img_size = int(spike_h) * int(spike_w)
    if img_size <= 0:
        raise ValueError(f"invalid spike resolution {spike_h}x{spike_w}")
    if img_size % 8 != 0:
        raise ValueError(f"spike_h*spike_w must be divisible by 8, got {img_size}")

    frame_bytes = img_size // 8
    img_num = raw.size // frame_bytes
    if img_num <= 0:
        raise ValueError(f"empty or invalid dat file: {dat_path}")

    usable = raw[: img_num * frame_bytes]

    if device == "cuda":
        tensor = torch.from_numpy(usable).to(device="cuda", dtype=torch.uint8, non_blocking=True)
        bits = torch.arange(8, device="cuda", dtype=torch.uint8)
        unpacked = tensor.unsqueeze(1).bitwise_right_shift(bits).bitwise_and_(1)
        spike_matrix = unpacked.reshape(-1).reshape(img_num, spike_h, spike_w)
        if flipud:
            spike_matrix = torch.flip(spike_matrix, dims=(1,))
        return spike_matrix.cpu().numpy().astype(bool, copy=False)

    unpacked = np.unpackbits(usable, bitorder="little")
    spike_matrix = unpacked.reshape(img_num, spike_h, spike_w).astype(bool, copy=False)
    if flipud:
        spike_matrix = spike_matrix[:, ::-1, :]
    return spike_matrix


def count_dat_files(clip_dirs: List[Path]) -> int:
    total = 0
    for clip_dir in clip_dirs:
        spike_dir = clip_dir / "spike"
        if not spike_dir.exists():
            continue
        total += sum(1 for _ in spike_dir.glob("*.dat"))
    return total


def get_existing_output_bytes(clip_dirs: List[Path], dt: int, num_subframes: int) -> int:
    total = 0
    for clip_dir in clip_dirs:
        out_dir = build_output_dir_subframes(clip_dir, dt=dt, num_subframes=num_subframes)
        if not out_dir.exists():
            continue
        for ext in ("*.npy", "*.dat"):
            for artifact_path in out_dir.glob(ext):
                try:
                    total += artifact_path.stat().st_size
                except FileNotFoundError:
                    continue
    return total


def estimate_required_space(
    *,
    clip_dirs: List[Path],
    dt: int,
    spike_root: Path,
    spike_h: int,
    spike_w: int,
    num_subframes: int,
    artifact_format: str,
    npy_dtype: str,
) -> SpaceEstimate:
    dat_files = count_dat_files(clip_dirs)
    bytes_per_file = bytes_per_artifact(
        spike_h=spike_h,
        spike_w=spike_w,
        num_subframes=num_subframes,
        artifact_format=artifact_format,
        npy_dtype=npy_dtype,
    )
    estimated_bytes = dat_files * bytes_per_file
    existing_output_bytes = get_existing_output_bytes(clip_dirs, dt=dt, num_subframes=num_subframes)
    free_bytes = shutil.disk_usage(spike_root).free
    projected_additional_bytes = max(0, estimated_bytes - existing_output_bytes)
    return SpaceEstimate(
        dat_files=dat_files,
        bytes_per_file=bytes_per_file,
        estimated_bytes=estimated_bytes,
        free_bytes=free_bytes,
        existing_output_bytes=existing_output_bytes,
        projected_additional_bytes=projected_additional_bytes,
    )


def build_scflow_window(spike_matrix: np.ndarray, short_policy: str) -> Tuple[np.ndarray, str]:
    if spike_matrix.ndim != 3:
        raise ValueError(f"spike_matrix must be [T,H,W], got shape={tuple(spike_matrix.shape)}")

    t = int(spike_matrix.shape[0])
    if t <= 0:
        raise ValueError("spike_matrix time length T must be > 0")

    target = 25
    if t == target:
        window = spike_matrix.astype(np.float32)
        validate_encoding25_tensor(window)
        return window, "exact25"

    if t > target:
        st = (t - target) // 2
        ed = st + target
        window = spike_matrix[st:ed].astype(np.float32)
        validate_encoding25_tensor(window)
        return window, "cropped"

    if short_policy == "strict":
        raise ValueError(f"T={t} < 25 under short_policy=strict")

    pad_total = target - t
    left = pad_total // 2
    right = pad_total - left
    if short_policy == "pad_edge":
        window = np.pad(spike_matrix, ((left, right), (0, 0), (0, 0)), mode="edge").astype(np.float32)
    elif short_policy == "pad_zero":
        window = np.pad(
            spike_matrix,
            ((left, right), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        ).astype(np.float32)
    else:
        raise ValueError(f"unknown short_policy={short_policy}")

    validate_encoding25_tensor(window)
    return window, "padded"


def build_scflow_subframe_windows(
    spike_matrix: np.ndarray,
    num_subframes: int,
) -> np.ndarray:
    """Extract S sub-windows from a spike matrix based on its own T_raw."""
    if spike_matrix.ndim != 3:
        raise ValueError(f"spike_matrix must be [T,H,W], got shape={tuple(spike_matrix.shape)}")
    centers = compute_subframe_centers(
        t_raw=spike_matrix.shape[0],
        num_subframes=num_subframes,
    )
    windows = [build_centered_window(spike_matrix, center) for center in centers]
    result = np.stack(windows, axis=0)
    validate_subframes_tensor(result, num_subframes)
    return result


def process_clip(
    *,
    clip_dir: Path,
    dt: int,
    short_policy: str,
    dry_run: bool,
    overwrite: bool,
    spike_h: int,
    spike_w: int,
    device: str,
    artifact_format: str,
    npy_dtype: str,
    num_subframes: int = 1,
) -> Tuple[EncodeResult, List[str]]:
    clip_name = clip_dir.name
    spike_dir = clip_dir / "spike"
    out_dir = build_output_dir_subframes(clip_dir, dt=dt, num_subframes=num_subframes)
    result = EncodeResult()
    missing: List[str] = []

    if not spike_dir.exists():
        missing.append(f"missing spike dir: {spike_dir}")
        return result, missing

    entries = list(iter_frame_indices(spike_dir))
    if not entries:
        missing.append(f"no .dat frames: {spike_dir}")
        return result, missing

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, dat_path in entries:
        out_base = out_dir / f"{frame_idx:0{len(dat_path.stem)}d}"
        out_path = out_base.with_suffix(f".{artifact_format}")
        if out_path.exists() and not overwrite:
            result.skipped_existing += 1
            continue

        if dry_run:
            result.dry_run += 1
            continue

        try:
            spike_matrix = decode_spike_dat(
                dat_path=dat_path,
                spike_h=spike_h,
                spike_w=spike_w,
                device=device,
                flipud=True,
            )
            if num_subframes > 1:
                encoded = build_scflow_subframe_windows(spike_matrix, num_subframes)
                if artifact_format == "npy" and npy_dtype == "bool":
                    encoded = encoded.astype(bool, copy=False)
                save_encoding25_artifact(out_base, encoded, artifact_format)
                result.generated += 1
                result.exact25 += 1
            else:
                encoded, kind = build_scflow_window(spike_matrix=spike_matrix, short_policy=short_policy)
                if artifact_format == "npy" and npy_dtype == "bool":
                    encoded = encoded.astype(bool, copy=False)
                save_encoding25_artifact(out_base, encoded, artifact_format)
                result.generated += 1
                if kind == "exact25":
                    result.exact25 += 1
                elif kind == "cropped":
                    result.cropped += 1
                else:
                    result.padded += 1
        except Exception as exc:
            result.failed += 1
            missing.append(f"clip={clip_name} frame={frame_idx} error={exc}")

    return result, missing


def _process_clip_worker(args: Tuple[Path, int, str, bool, bool, int, int, str, str, str, int]) -> Tuple[str, EncodeResult, List[str]]:
    clip_dir, dt, short_policy, dry_run, overwrite, spike_h, spike_w, device, artifact_format, npy_dtype, num_subframes = args
    clip_name = clip_dir.name
    try:
        result, missing = process_clip(
            clip_dir=clip_dir,
            dt=dt,
            short_policy=short_policy,
            dry_run=dry_run,
            overwrite=overwrite,
            spike_h=spike_h,
            spike_w=spike_w,
            device=device,
            artifact_format=artifact_format,
            npy_dtype=npy_dtype,
            num_subframes=num_subframes,
        )
        return clip_name, result, missing
    except Exception as exc:
        failed = EncodeResult(failed=1)
        return clip_name, failed, [f"clip={clip_name} fatal_error={exc}"]


def process_all_clips(
    *,
    clip_dirs: List[Path],
    dt: int,
    short_policy: str,
    dry_run: bool,
    overwrite: bool,
    spike_h: int,
    spike_w: int,
    device: str,
    artifact_format: str,
    npy_dtype: str,
    num_workers: int,
    num_subframes: int = 1,
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
                short_policy=short_policy,
                dry_run=dry_run,
                overwrite=overwrite,
                spike_h=spike_h,
                spike_w=spike_w,
                device=device,
                artifact_format=artifact_format,
                npy_dtype=npy_dtype,
                num_subframes=num_subframes,
            )
            _merge_result(all_result, result)
            all_missing.extend(missing)
        return all_result, all_missing

    task_args = [
        (
            clip_dir,
            dt,
            short_policy,
            dry_run,
            overwrite,
            spike_h,
            spike_w,
            device,
            artifact_format,
            npy_dtype,
            num_subframes,
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
                f"skipped_existing={result.skipped_existing} dry_run={result.dry_run} "
                f"exact25={result.exact25} cropped={result.cropped} padded={result.padded} failed={result.failed}"
            )

    return all_result, all_missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SCFlow strict encoding25 artifacts.")
    parser.add_argument("--spike-root", required=True, type=Path)
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument(
        "--short-policy",
        type=str,
        default="pad_edge",
        choices=["strict", "pad_edge", "pad_zero"],
        help="Policy when T<25 for a .dat spike matrix.",
    )
    parser.add_argument("--spike-h", type=int, default=360)
    parser.add_argument("--spike-w", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--artifact-format",
        type=str,
        default="npy",
        choices=["npy", "dat"],
        help="Output format for encoding25 artifacts. 'dat' uses lossless packed binary spikes.",
    )
    parser.add_argument(
        "--npy-dtype",
        type=str,
        default="bool",
        choices=["bool", "float32"],
        help="When --artifact-format=npy, save as bool or float32. bool is lossless for binary spikes and 4x smaller.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Decode .dat on CUDA when available. Default: auto.",
    )
    parser.add_argument(
        "--space-only",
        action="store_true",
        help="Only estimate required output space and exit without generating files.",
    )
    parser.add_argument(
        "--allow-insufficient-space",
        action="store_true",
        help="Proceed even when the estimated output size exceeds currently free disk space.",
    )
    parser.add_argument(
        "--subframes",
        type=int,
        default=4,
        help="Number of sub-windows per .dat file (S). Default: 4.",
    )
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
    device = detect_device(str(args.device))
    artifact_format = str(args.artifact_format).strip().lower()
    npy_dtype = str(args.npy_dtype).strip().lower()

    clip_dirs = [p for p in sorted(spike_root.iterdir()) if p.is_dir()]
    if args.max_clips > 0:
        clip_dirs = clip_dirs[: args.max_clips]

    requested_workers = int(args.num_workers)
    if requested_workers > 1 and len(clip_dirs) > 1:
        cpu_cnt = os.cpu_count() or requested_workers
        requested_workers = min(requested_workers, cpu_cnt)
    if device == "cuda":
        requested_workers = 1

    estimate = estimate_required_space(
        clip_dirs=clip_dirs,
        dt=int(args.dt),
        spike_root=spike_root,
        spike_h=int(args.spike_h),
        spike_w=int(args.spike_w),
        num_subframes=int(args.subframes),
        artifact_format=artifact_format,
        npy_dtype=npy_dtype,
    )
    print("[prepare_scflow_encoding25] Space Estimate")
    print(f"  clips={len(clip_dirs)}")
    print(f"  dat_files={estimate.dat_files}")
    print(f"  bytes_per_file={format_bytes(estimate.bytes_per_file)}")
    print(f"  estimated_output={format_bytes(estimate.estimated_bytes)}")
    print(f"  existing_output={format_bytes(estimate.existing_output_bytes)}")
    print(f"  projected_additional={format_bytes(estimate.projected_additional_bytes)}")
    print(f"  free_space={format_bytes(estimate.free_bytes)}")
    print(f"  device={device}")
    print(f"  artifact_format={artifact_format}")
    if artifact_format == "npy":
        print(f"  npy_dtype={npy_dtype}")

    if args.space_only:
        return

    if estimate.projected_additional_bytes > estimate.free_bytes and not args.allow_insufficient_space:
        raise SystemExit(
            "[prepare_scflow_encoding25] insufficient free space: "
            f"need about {format_bytes(estimate.projected_additional_bytes)} additional space, "
            f"but only {format_bytes(estimate.free_bytes)} is free. "
            "Delete old artifacts, use --max-clips for batches, or pass "
            "--allow-insufficient-space to override."
        )

    all_result, all_missing = process_all_clips(
        clip_dirs=clip_dirs,
        dt=int(args.dt),
        short_policy=str(args.short_policy),
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        spike_h=int(args.spike_h),
        spike_w=int(args.spike_w),
        device=device,
        artifact_format=artifact_format,
        npy_dtype=npy_dtype,
        num_workers=requested_workers,
        num_subframes=int(args.subframes),
    )

    print("[prepare_scflow_encoding25] Summary")
    print(f"  clips={len(clip_dirs)}")
    print(f"  workers={requested_workers}")
    print(f"  device={device}")
    print(f"  artifact_format={artifact_format}")
    if artifact_format == "npy":
        print(f"  npy_dtype={npy_dtype}")
    print(f"  short_policy={args.short_policy}")
    print(f"  generated={all_result.generated}")
    print(f"  skipped_existing={all_result.skipped_existing}")
    print(f"  dry_run={all_result.dry_run}")
    print(f"  exact25={all_result.exact25}")
    print(f"  cropped={all_result.cropped}")
    print(f"  padded={all_result.padded}")
    print(f"  failed={all_result.failed}")

    if all_missing:
        print("[prepare_scflow_encoding25] Missing/Errors:")
        for item in all_missing:
            print(f"  - {item}")

    if all_result.failed > 0 and not args.dry_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
