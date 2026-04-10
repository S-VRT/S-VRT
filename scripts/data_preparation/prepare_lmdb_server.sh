#!/usr/bin/env bash
set -euo pipefail

# Robust LMDB builder for GoPro dataset on server.
# - Uses single-process image reading to avoid multiprocessing key-loss issues.
# - Skips unreadable/corrupted images instead of crashing.
#
# Usage:
#   bash scripts/data_preparation/prepare_lmdb_server.sh
#   GOPRO_ROOT=/path/to/GOPRO_Large bash scripts/data_preparation/prepare_lmdb_server.sh
#   SPLITS="train test" bash scripts/data_preparation/prepare_lmdb_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GOPRO_ROOT="${GOPRO_ROOT:-/storage/main/users/hengyuanwu/Datasets/gopro_spike/GOPRO_Large}"
SPLITS="${SPLITS:-train}"
TARGET_SUFFIXES="${TARGET_SUFFIXES:-GT GT_blurred}"

cd "${REPO_ROOT}"

echo "============================================================"
echo "Server LMDB Preparation (robust)"
echo "Repo: ${REPO_ROOT}"
echo "GoPro root: ${GOPRO_ROOT}"
echo "Splits: ${SPLITS}"
echo "Targets: ${TARGET_SUFFIXES}"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import os
import shutil
import cv2
from PIL import Image, UnidentifiedImageError

from scripts.data_preparation import create_lmdb as create_lmdb_module
from utils.utils_lmdb import LmdbMaker, read_img_worker

gopro_root = Path(os.environ.get("GOPRO_ROOT", "/storage/main/users/hengyuanwu/Datasets/gopro_spike/GOPRO_Large"))
splits = [s for s in os.environ.get("SPLITS", "train").split() if s.strip()]
suffixes = [s for s in os.environ.get("TARGET_SUFFIXES", "GT GT_blurred").split() if s.strip()]

if not gopro_root.exists():
    raise SystemExit(f"[fatal] GOPRO_ROOT not found: {gopro_root}")

total_built = 0
total_skipped_bad = 0

def is_readable_png(path: Path) -> bool:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        return True
    try:
        with Image.open(path) as p:
            p.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False

for split in splits:
    for suffix in suffixes:
        target = gopro_root / f"{split}_{suffix}"
        if not target.exists():
            print(f"[skip] missing directory: {target}")
            continue

        img_paths, keys = create_lmdb_module.prepare_keys_gopro(str(target))
        if not img_paths:
            print(f"[skip] no images found: {target}")
            continue

        filtered_paths = []
        filtered_keys = []
        bad_paths = []
        for rel_path, key in zip(img_paths, keys):
            abs_path = target / rel_path
            if is_readable_png(abs_path):
                filtered_paths.append(rel_path)
                filtered_keys.append(key)
            else:
                bad_paths.append(str(abs_path))

        lmdb_path = Path(str(target) + ".lmdb")
        if lmdb_path.exists():
            print(f"[clean] removing existing lmdb: {lmdb_path}")
            shutil.rmtree(lmdb_path)

        print(f"[build] {lmdb_path}")
        print(f"        source={target}")
        print(f"        images_ok={len(filtered_paths)} images_bad={len(bad_paths)}")

        maker = LmdbMaker(str(lmdb_path), compress_level=1)
        written = 0
        for rel_path, key in zip(filtered_paths, filtered_keys):
            abs_path = target / rel_path
            try:
                _, img_byte, img_shape = read_img_worker(str(abs_path), key, 1)
                maker.put(img_byte, key, img_shape)
                written += 1
            except Exception as exc:
                bad_paths.append(f"{abs_path} :: {exc}")
        maker.close()

        total_built += 1
        total_skipped_bad += len(bad_paths)

        print(f"[done] {lmdb_path} written={written} bad={len(bad_paths)}")
        if bad_paths:
            print("[bad-sample] up to first 20 entries:")
            for item in bad_paths[:20]:
                print(f"  - {item}")

print("============================================================")
print(f"[summary] lmdb_built={total_built}, total_bad_skipped={total_skipped_bad}")
print("============================================================")
PY

echo
echo "LMDB preparation finished."
