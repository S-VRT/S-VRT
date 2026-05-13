from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.spike_recc import SpikeStream, extract_centered_raw_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute centered raw spike-window artifacts.")
    parser.add_argument("--spike-root", required=True, help="Root directory containing clip folders and spike/*.dat files.")
    parser.add_argument("--spike-h", type=int, default=360)
    parser.add_argument("--spike-w", type=int, default=640)
    parser.add_argument("--window-length", type=int, default=9)
    parser.add_argument("--format", choices=["npy", "npz"], default="npy")
    parser.add_argument(
        "--storage-dtype",
        choices=["uint8", "float32"],
        default="uint8",
        help="On-disk dtype for raw-window artifacts. uint8 stores binary spikes compactly.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clip", default=None, help="Optional single clip name to process.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of .dat files to process.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min((os.cpu_count() or 1), 64)),
        help="Number of worker processes. Each worker uses a single CPU thread internally.",
    )
    return parser.parse_args()


def iter_dat_files(spike_root: Path, clip_name: str | None):
    clip_dirs = [spike_root / clip_name] if clip_name else sorted(p for p in spike_root.iterdir() if p.is_dir())
    for clip_dir in clip_dirs:
        spike_dir = clip_dir / "spike"
        if not spike_dir.is_dir():
            continue
        for dat_path in sorted(spike_dir.glob("*.dat")):
            yield clip_dir.name, dat_path


def build_output_base(spike_root: Path, clip_name: str, window_length: int, frame_stem: str) -> Path:
    out_dir = spike_root / clip_name / f"raw_window_l{window_length}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / frame_stem


def _encode_raw_window(raw_window: np.ndarray, storage_dtype: str) -> np.ndarray:
    if storage_dtype == "uint8":
        return np.rint(np.clip(np.asarray(raw_window, dtype=np.float32), 0.0, 1.0)).astype(np.uint8)
    return np.asarray(raw_window, dtype=np.float32)


def save_artifact(base_path: Path, raw_window: np.ndarray, artifact_format: str, storage_dtype: str) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encode_raw_window(raw_window, storage_dtype)
    if artifact_format == "npy":
        out_path = base_path.with_suffix(".npy")
        np.save(out_path, encoded)
        return out_path
    out_path = base_path.with_suffix(".npz")
    np.savez_compressed(out_path, spike_voxel=encoded)
    return out_path


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _process_one(
    spike_root_str: str,
    clip_name: str,
    dat_path_str: str,
    spike_h: int,
    spike_w: int,
    window_length: int,
    artifact_format: str,
    storage_dtype: str,
    overwrite: bool,
) -> str:
    spike_root = Path(spike_root_str)
    dat_path = Path(dat_path_str)
    out_base = build_output_base(spike_root, clip_name, window_length, dat_path.stem)
    out_path = out_base.with_suffix(f".{artifact_format}")
    if out_path.exists() and not overwrite:
        return "skipped"

    spike_stream = SpikeStream(
        offline=True,
        filepath=str(dat_path),
        spike_h=spike_h,
        spike_w=spike_w,
        print_dat_detail=False,
    )
    spike_matrix = spike_stream.get_spike_matrix(flipud=True)
    raw_window = extract_centered_raw_window(spike_matrix, window_length=window_length)
    save_artifact(out_base, raw_window, artifact_format, storage_dtype)
    return "processed"


def main() -> None:
    args = parse_args()
    spike_root = Path(args.spike_root)
    jobs = list(iter_dat_files(spike_root, args.clip))
    if args.limit:
        jobs = jobs[: args.limit]

    processed = 0
    skipped = 0
    worker_count = max(1, int(args.workers))
    print(
        f"[prepare_spike_raw_window] start jobs={len(jobs)} workers={worker_count} "
        f"window_length={args.window_length} format={args.format} storage_dtype={args.storage_dtype}"
    )

    with ProcessPoolExecutor(max_workers=worker_count, initializer=_worker_init) as executor:
        futures = {
            executor.submit(
                _process_one,
                str(spike_root),
                clip_name,
                str(dat_path),
                args.spike_h,
                args.spike_w,
                args.window_length,
                args.format,
                args.storage_dtype,
                args.overwrite,
            ): (clip_name, dat_path)
            for clip_name, dat_path in jobs
        }

        for idx, future in enumerate(as_completed(futures), start=1):
            clip_name, dat_path = futures[future]
            result = future.result()
            if result == "processed":
                processed += 1
            elif result == "skipped":
                skipped += 1
            if idx % 100 == 0 or idx == len(jobs):
                print(
                    f"[prepare_spike_raw_window] completed={idx}/{len(jobs)} processed={processed} "
                    f"skipped={skipped} last={clip_name}/{dat_path.name}"
                )

    print(f"[prepare_spike_raw_window] done processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
