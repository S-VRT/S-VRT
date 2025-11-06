#!/usr/bin/env python3
"""
Quick Dataset Status Checker

This script quickly checks the status of your GoPro + Spike dataset
and tells you if data preparation is needed.

Usage:
    python scripts/data_preparation/check_dataset_status.py
"""

import sys
from pathlib import Path
from typing import Tuple

# Default paths
DEFAULT_GOPRO_ROOT = Path("/media/mallm/hd4t/modelrepostore/datasets/GOPRO_Large")
DEFAULT_SPIKE_ROOT = Path("/media/mallm/hd4t/modelrepostore/datasets/gopro_spike/GOPRO_Large_spike_seq")
KAIR_ROOT = Path(__file__).parent.parent.parent


def check_raw_data(gopro_root: Path, spike_root: Path) -> Tuple[bool, bool]:
    """Check if raw data exists."""
    gopro_raw = (gopro_root / "train").exists() and (gopro_root / "test").exists()
    spike_raw = (spike_root / "train").exists() and (spike_root / "test").exists()
    return gopro_raw, spike_raw


def check_prepared_data(gopro_root: Path) -> Tuple[bool, bool, bool]:
    """Check if prepared data exists."""
    train_gt = (gopro_root / "train_GT").exists()
    train_lq = (gopro_root / "train_GT_blurred").exists()
    has_lmdb = (gopro_root / "train_GT.lmdb").exists()
    return train_gt, train_lq, has_lmdb


def check_meta_info() -> bool:
    """Check if meta info files exist."""
    train_meta = (KAIR_ROOT / "data" / "meta_info" / "meta_info_GoPro_train_GT.txt").exists()
    return train_meta


def count_sequences(directory: Path) -> int:
    """Count subdirectories (sequences) in a directory."""
    if not directory.exists():
        return 0
    return len([d for d in directory.iterdir() if d.is_dir()])


def print_status():
    """Print dataset status."""
    print("=" * 70)
    print("GoPro + Spike Dataset Status Check")
    print("=" * 70)
    print()
    
    # Check raw data
    print("📦 Raw Dataset Status:")
    print("-" * 70)
    gopro_raw, spike_raw = check_raw_data(DEFAULT_GOPRO_ROOT, DEFAULT_SPIKE_ROOT)
    
    if gopro_raw:
        train_seqs = count_sequences(DEFAULT_GOPRO_ROOT / "train")
        test_seqs = count_sequences(DEFAULT_GOPRO_ROOT / "test")
        print(f"✓ GoPro raw data found: {train_seqs} train, {test_seqs} test sequences")
    else:
        print(f"✗ GoPro raw data NOT found at {DEFAULT_GOPRO_ROOT}")
    
    if spike_raw:
        train_seqs = count_sequences(DEFAULT_SPIKE_ROOT / "train")
        test_seqs = count_sequences(DEFAULT_SPIKE_ROOT / "test")
        print(f"✓ Spike raw data found: {train_seqs} train, {test_seqs} test sequences")
    else:
        print(f"✗ Spike raw data NOT found at {DEFAULT_SPIKE_ROOT}")
    
    print()
    
    # Check prepared data
    print("🎯 Prepared Dataset Status:")
    print("-" * 70)
    train_gt, train_lq, has_lmdb = check_prepared_data(DEFAULT_GOPRO_ROOT)
    
    if train_gt and train_lq:
        train_seqs = count_sequences(DEFAULT_GOPRO_ROOT / "train_GT")
        print(f"✓ Prepared training data found: {train_seqs} sequences")
        print(f"  - GT: {DEFAULT_GOPRO_ROOT}/train_GT/")
        print(f"  - LQ: {DEFAULT_GOPRO_ROOT}/train_GT_blurred/")
    else:
        print("✗ Prepared training data NOT found")
        print(f"  Missing: train_GT/ and/or train_GT_blurred/")
    
    if has_lmdb:
        print("✓ LMDB databases found (fast loading enabled)")
    else:
        print("○ LMDB databases not found (optional, for faster loading)")
    
    has_meta = check_meta_info()
    if has_meta:
        print("✓ Meta info files found")
    else:
        print("✗ Meta info files NOT found")
    
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    print("-" * 70)
    
    if not (gopro_raw and spike_raw):
        print("⚠️  Raw datasets are missing!")
        print("   Please download and extract both datasets:")
        print("   1. GoPro Large Dataset")
        print("   2. GoPro Spike Dataset")
        print()
    
    elif not (train_gt and train_lq and has_meta):
        print("📋 Data preparation needed!")
        print("   Run one of these commands:")
        print()
        print("   # Quick start with data preparation:")
        print("   bash launch_train.sh 1 --prepare-data")
        print()
        print("   # With LMDB for faster loading (recommended):")
        print("   bash launch_train.sh 1 --prepare-data --generate-lmdb")
        print()
        print("   # Or prepare data separately:")
        print("   python scripts/data_preparation/prepare_gopro_spike_dataset.py")
        print()
    
    elif not has_lmdb:
        print("✅ Data is prepared and ready for training!")
        print("   You can start training with:")
        print("   bash launch_train.sh 1")
        print()
        print("💡 Consider generating LMDB for faster data loading:")
        print("   bash launch_train.sh 1 --prepare-data --generate-lmdb")
        print()
    
    else:
        print("✅ Everything is ready! Data is prepared with LMDB.")
        print("   Start training with:")
        print("   bash launch_train.sh 1")
        print()
    
    print("=" * 70)


def main():
    print_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())

