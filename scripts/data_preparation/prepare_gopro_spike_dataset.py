#!/usr/bin/env python3
"""
GoPro + Spike Dataset Preparation Script
=========================================

This script prepares the GoPro video deblurring dataset along with its corresponding
Spike camera data for training. It handles:

1. Dataset structure validation
2. Creating organized train/test splits with GT, LQ (blurred), and Spike data
3. Generating meta info files for the training pipeline
4. Optional LMDB generation for faster data loading

Dataset Structure:
------------------
Input (after unzipping gopro_spike.zip):
  - gopro_spike/
    ├── GOPRO_Large/
    │   ├── train/
    │   │   ├── GOPR0374_11_00/
    │   │   │   ├── blur/         # Blurred frames
    │   │   │   ├── sharp/        # Sharp (GT) frames
    │   │   │   └── ...
    │   │   └── ...
    │   └── test/
    │       └── ...
    │
    └── GOPRO_Large_spike_seq/
        ├── train/
        │   ├── GOPR0374_11_00/
        │   │   └── spike/        # Spike data (.dat files)
        │   └── ...
        ├── test/
        └── config.yaml           # Spike camera config (250x400)

Output:
  - GOPRO_Large/
    ├── train_GT/             # Organized sharp frames
    ├── train_GT_blurred/     # Organized blurred frames  
    └── test_GT/ test_GT_blurred/
  
  - gopro_spike/GOPRO_Large_spike_seq/
    ├── train/                # Spike data (already organized)
    └── test/

Usage:
------
    # If zip extracted to /media/mallm/hd4t/modelrepostore/datasets/gopro_spike/
    python prepare_gopro_spike_dataset.py --dataset_root /media/mallm/hd4t/modelrepostore/datasets/gopro_spike
    
    # Or specify paths explicitly
    python prepare_gopro_spike_dataset.py --gopro_root /path/to/gopro_spike/GOPRO_Large \\
                                          --spike_root /path/to/gopro_spike/GOPRO_Large_spike_seq \\
                                          --generate_lmdb
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import yaml

# Add KAIR to path
SCRIPT_DIR = Path(__file__).parent.absolute()
KAIR_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(KAIR_ROOT))

# -----------------------------------------------------------------------------
# Centralized dataset path defaults (environment-driven only)
# -----------------------------------------------------------------------------
ENV_DATASET_ROOT = os.environ.get("SVRT_DATASET_ROOT")
ENV_GOPRO_ROOT = os.environ.get("SVRT_GOPRO_ROOT")
ENV_SPIKE_ROOT = os.environ.get("SVRT_SPIKE_ROOT")

DEFAULT_DATASET_ROOT = ENV_DATASET_ROOT
DEFAULT_GOPRO_ROOT = ENV_GOPRO_ROOT or (
    str(Path(ENV_DATASET_ROOT) / "GOPRO_Large") if ENV_DATASET_ROOT else None
)
DEFAULT_SPIKE_ROOT = ENV_SPIKE_ROOT or (
    str(Path(ENV_DATASET_ROOT) / "GOPRO_Large_spike_seq") if ENV_DATASET_ROOT else None
)

SPIKE_CONFIG_FILENAME = "config.yaml"
DEFAULT_SPIKE_CONFIG = {
    "spike_h": 250,
    "spike_w": 400,
    "is_labeled": False,
    "labeled_data_type": [0],
}


def validate_gopro_structure(gopro_root: Path, required_splits: List[str]) -> Tuple[bool, str]:
    """Validate the GoPro dataset has the expected structure."""
    if not gopro_root.exists():
        return False, f"GoPro root does not exist: {gopro_root}"
    
    available_splits = [split for split in required_splits if (gopro_root / split).exists()]
    if not available_splits:
        return False, (
            f"None of the requested splits {required_splits} exist under {gopro_root}. "
            "At least one split (e.g., 'train') is required."
        )
    
    for split in available_splits:
        split_dir = gopro_root / split
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return False, f"No sequences found in {split_dir}"
        
        # Check first subdir for blur/sharp folders
        first_subdir = subdirs[0]
        has_blur = (first_subdir / "blur").exists()
        has_sharp = (first_subdir / "sharp").exists()
        
        if not (has_blur and has_sharp):
            return False, f"Expected blur/sharp folders in {first_subdir}"
    
    return True, f"GoPro structure is valid for splits: {', '.join(available_splits)}"


def validate_spike_structure(spike_root: Path, required_splits: List[str]) -> Tuple[bool, str]:
    """Validate the Spike dataset has the expected structure."""
    if not spike_root.exists():
        return False, f"Spike root does not exist: {spike_root}"
    
    config_file = spike_root / "config.yaml"
    
    if not config_file.exists():
        return False, f"Missing config.yaml in {spike_root}"
    
    available_splits = [split for split in required_splits if (spike_root / split).exists()]
    if not available_splits:
        return False, (
            f"None of the requested splits {required_splits} exist under {spike_root}. "
            "At least one split (e.g., 'train') is required."
        )
    
    for split in available_splits:
        split_dir = spike_root / split
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return False, f"No sequences found in {split_dir}"
        
        # Check first subdir for spike folder
        first_subdir = subdirs[0]
        spike_dir = first_subdir / "spike"
        if not spike_dir.exists():
            return False, f"Expected spike folder in {first_subdir}"
        
        # Check for .dat files
        dat_files = list(spike_dir.glob("*.dat"))
        if not dat_files:
            return False, f"No .dat files found in {spike_dir}"
    
    return True, f"Spike structure is valid for splits: {', '.join(available_splits)}"


def load_spike_config(spike_root: Path) -> dict:
    """Load spike camera configuration."""
    config_file = spike_root / SPIKE_CONFIG_FILENAME
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_spike_config(spike_root: Path, config_values: dict, force: bool = False) -> Path:
    """Ensure spike config.yaml exists with the provided values."""
    config_file = spike_root / SPIKE_CONFIG_FILENAME

    if config_file.exists() and not force:
        print(f"  ✓ Spike config already present: {config_file}")
        return config_file

    if not spike_root.exists():
        raise FileNotFoundError(
            f"Spike root does not exist, cannot write config: {spike_root}"
        )

    with open(config_file, "w") as f:
        yaml.safe_dump(config_values, f, sort_keys=False)

    print(f"  ✓ Wrote spike config: {config_file}")
    return config_file


def organize_gopro_data(gopro_root: Path, split: str = "train", force: bool = False) -> Path:
    """
    Organize GoPro data from the raw structure to the organized structure.
    
    Args:
        gopro_root: Root directory of GOPRO_Large dataset
        split: 'train' or 'test'
        force: If True, regenerate even if output exists
    
    Returns:
        Path to the organized GT directory
    """
    raw_dir = gopro_root / split
    gt_out_dir = gopro_root / f"{split}_GT"
    lq_out_dir = gopro_root / f"{split}_GT_blurred"
    
    # Check if already organized
    if gt_out_dir.exists() and lq_out_dir.exists() and not force:
        print(f"  ✓ {split} data already organized, skipping")
        return gt_out_dir
    
    print(f"  Organizing {split} data...")
    gt_out_dir.mkdir(exist_ok=True)
    lq_out_dir.mkdir(exist_ok=True)
    
    # Get all sequence directories
    sequences = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    
    for seq in sequences:
        seq_name = seq.name
        blur_dir = seq / "blur"
        sharp_dir = seq / "sharp"
        
        if not blur_dir.exists() or not sharp_dir.exists():
            print(f"    Warning: Skipping {seq_name} (missing blur or sharp folder)")
            continue
        
        # Create output sequence directories
        gt_seq_out = gt_out_dir / seq_name
        lq_seq_out = lq_out_dir / seq_name
        gt_seq_out.mkdir(exist_ok=True)
        lq_seq_out.mkdir(exist_ok=True)
        
        # Copy sharp frames to GT
        sharp_frames = sorted(sharp_dir.glob("*.png"))
        for frame in sharp_frames:
            dst = gt_seq_out / frame.name
            if not dst.exists():
                shutil.copy2(frame, dst)
        
        # Copy blurred frames to LQ
        blur_frames = sorted(blur_dir.glob("*.png"))
        for frame in blur_frames:
            dst = lq_seq_out / frame.name
            if not dst.exists():
                shutil.copy2(frame, dst)
        
        print(f"    ✓ {seq_name}: {len(sharp_frames)} GT, {len(blur_frames)} LQ frames")
    
    print(f"  ✓ Organized {len(sequences)} sequences for {split}")
    return gt_out_dir


def verify_gopro_spike_alignment(gopro_root: Path, spike_root: Path, split: str = "train") -> bool:
    """Verify that GoPro and Spike datasets have matching sequences."""
    gopro_dir = gopro_root / split
    spike_dir = spike_root / split
    
    if not gopro_dir.exists() or not spike_dir.exists():
        print(f"  Skipping alignment for split '{split}' (missing directory).")
        return False
    
    gopro_seqs = set([d.name for d in gopro_dir.iterdir() if d.is_dir()])
    spike_seqs = set([d.name for d in spike_dir.iterdir() if d.is_dir()])
    
    missing_in_spike = gopro_seqs - spike_seqs
    missing_in_gopro = spike_seqs - gopro_seqs
    
    if missing_in_spike:
        print(f"  Warning: {len(missing_in_spike)} sequences in GoPro but not in Spike:")
        for seq in sorted(list(missing_in_spike)[:5]):
            print(f"    - {seq}")
        if len(missing_in_spike) > 5:
            print(f"    ... and {len(missing_in_spike) - 5} more")
    
    if missing_in_gopro:
        print(f"  Warning: {len(missing_in_gopro)} sequences in Spike but not in GoPro:")
        for seq in sorted(list(missing_in_gopro)[:5]):
            print(f"    - {seq}")
        if len(missing_in_gopro) > 5:
            print(f"    ... and {len(missing_in_gopro) - 5} more")
    
    common = gopro_seqs & spike_seqs
    print(f"  ✓ {len(common)} common sequences found")
    
    return len(common) > 0


def _extract_frame_indices(frames: List[Path]) -> Tuple[int, int]:
    """Return start frame index and frame count from a list of frame paths."""
    if not frames:
        return 0, 0

    indices = sorted(int(frame.stem) for frame in frames)
    start_frame = indices[0]
    # Check for gaps
    expected = list(range(start_frame, start_frame + len(indices)))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        extra = sorted(set(indices) - set(expected))
        print(
            "    Warning: Non-contiguous frame indices detected. "
            f"start={start_frame}, count={len(indices)}, "
            f"missing_samples={missing[:5]}, extra_samples={extra[:5]}"
        )
    return start_frame, len(indices)


def generate_meta_info(gopro_root: Path, split: str = "train") -> Path:
    """
    Generate meta info file for the dataset.
    
    Format: folder_name frame_count border start_frame
    Example: GOPR0372_07_00 100 0 0
    
    Where:
    - folder_name: sequence name
    - frame_count: number of frames
    - border: border size (always 0 for GoPro)
    - start_frame: starting frame number (always 0 for GoPro)
    """
    gt_dir = gopro_root / f"{split}_GT"
    meta_info_file = KAIR_ROOT / "data" / "meta_info" / f"meta_info_GoPro_{split}_GT.txt"
    
    # Create meta_info directory if it doesn't exist
    meta_info_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not gt_dir.exists():
        print(f"  Warning: GT directory missing for split '{split}' ({gt_dir}), skipping meta info.")
        return meta_info_file
    
    sequences = sorted([d for d in gt_dir.iterdir() if d.is_dir()])
    
    with open(meta_info_file, 'w') as f:
        for seq in sequences:
            frames = sorted(seq.glob("*.png"))
            lq_seq = (gopro_root / f"{split}_GT_blurred" / seq.name)
            lq_frames = sorted(lq_seq.glob("*.png")) if lq_seq.exists() else []

            start_frame, frame_count = _extract_frame_indices(frames)

            if frame_count == 0:
                print(f"    Warning: No frames found for {seq.name}, skipping.")
                continue

            if len(frames) != len(lq_frames):
                print(
                    f"    Warning: GT/LQ frame count mismatch for {seq.name}. "
                    f"GT={len(frames)}, LQ={len(lq_frames)}"
                )

            # Write: folder_name frame_count border start_frame
            f.write(f"{seq.name} {frame_count} 0 {start_frame}\n")
    
    print(f"  ✓ Generated meta info: {meta_info_file}")
    print(f"    {len(sequences)} sequences")
    
    return meta_info_file


def generate_lmdb(gopro_root: Path, split: str = "train"):
    """Generate LMDB files for faster data loading."""
    try:
        from scripts.data_preparation import create_lmdb as create_lmdb_module
    except ImportError as e:
        print(f"  Warning: Could not import create_lmdb: {e}")
        print("  You can generate it later using scripts/data_preparation/create_lmdb.py")
        return

    gt_dir = gopro_root / f"{split}_GT"
    lq_dir = gopro_root / f"{split}_GT_blurred"

    def _build_lmdb(target_dir: Path, label: str):
        if not target_dir.exists():
            print(f"  Warning: {label} directory {target_dir} not found. Skipping.")
            return

        lmdb_path = Path(str(target_dir) + ".lmdb")
        img_path_list, keys = create_lmdb_module.prepare_keys_gopro(str(target_dir))
        if not img_path_list:
            print(f"  Warning: No images found in {target_dir}. Skipping LMDB creation.")
            return

        print(f"  Generating LMDB for {split} {label}...")
        create_lmdb_module.make_lmdb_from_imgs(
            str(target_dir),
            str(lmdb_path),
            img_path_list,
            keys,
            multiprocessing_read=True
        )

    _build_lmdb(gt_dir, "GT")
    _build_lmdb(lq_dir, "LQ")

    print(f"  ✓ LMDB generation complete for {split}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GoPro + Spike dataset for VRT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--gopro_root",
        type=str,
        default=DEFAULT_GOPRO_ROOT,
        help="Path to GOPRO_Large dataset root "
             "(default driven by $SVRT_GOPRO_ROOT or $SVRT_DATASET_ROOT/GOPRO_Large)"
    )
    
    parser.add_argument(
        "--spike_root",
        type=str,
        default=DEFAULT_SPIKE_ROOT,
        help="Path to GoPro Spike dataset root "
             "(default driven by $SVRT_SPIKE_ROOT or $SVRT_DATASET_ROOT/GOPRO_Large_spike_seq)"
    )
    
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory where the combined dataset was extracted. "
             "When provided (or via $SVRT_DATASET_ROOT), gopro/spike roots auto-detect."
    )
    
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Which splits to process"
    )
    
    parser.add_argument(
        "--generate_lmdb",
        action="store_true",
        help="Generate LMDB files for faster loading"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if output exists"
    )
    
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip dataset validation checks"
    )
    
    args = parser.parse_args()
    
    # Auto-detect paths from dataset_root if provided
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        if dataset_root.exists():
            # Check if we have the expected structure
            auto_gopro = dataset_root / "GOPRO_Large"
            auto_spike = dataset_root / "GOPRO_Large_spike_seq"
            
            if auto_gopro.exists():
                args.gopro_root = str(auto_gopro)
                print(f"Auto-detected GoPro root: {auto_gopro}")
            
            if auto_spike.exists():
                args.spike_root = str(auto_spike)
                print(f"Auto-detected Spike root: {auto_spike}")
    
    if not args.gopro_root or not args.spike_root:
        print("✗ Missing dataset paths.")
        print("  Please provide --gopro_root/--spike_root or set the environment variables")
        print("  SVRT_DATASET_ROOT, SVRT_GOPRO_ROOT, SVRT_SPIKE_ROOT accordingly.")
        return 1
    
    gopro_root = Path(args.gopro_root)
    spike_root = Path(args.spike_root)
    
    print("=" * 70)
    print("GoPro + Spike Dataset Preparation")
    print("=" * 70)
    
    requested_splits = args.splits or ["train", "test"]
    available_splits = []
    missing_splits = []
    for split in requested_splits:
        if (gopro_root / split).exists():
            available_splits.append(split)
        else:
            missing_splits.append(split)
    
    if not available_splits:
        print(f"✗ None of the requested splits {requested_splits} exist under {gopro_root}.")
        print("  Please double-check the dataset path or provide --splits with valid values.")
        return 1
    
    if missing_splits:
        print(f"  Warning: Skipping missing splits: {', '.join(missing_splits)}")
    
    args.splits = available_splits
    
    print(f"GoPro root: {gopro_root}")
    print(f"Spike root: {spike_root}")
    print(f"Requested splits: {requested_splits}")
    print(f"Using splits: {args.splits}")
    print(f"Generate LMDB: {args.generate_lmdb}")
    print()
    
    # Step 0: Ensure spike config exists
    print("Step 0: Preparing spike config...")
    try:
        ensure_spike_config(spike_root, DEFAULT_SPIKE_CONFIG, force=args.force)
    except FileNotFoundError as exc:
        print(f"  ✗ {exc}")
        return 1
    print()

    # Step 1: Validate dataset structure
    if not args.skip_validation:
        print("Step 1: Validating dataset structure...")
        
        valid, msg = validate_gopro_structure(gopro_root, args.splits)
        if not valid:
            print(f"  ✗ GoPro validation failed: {msg}")
            return 1
        print(f"  ✓ {msg}")
        
        valid, msg = validate_spike_structure(spike_root, args.splits)
        if not valid:
            print(f"  ✗ Spike validation failed: {msg}")
            return 1
        print(f"  ✓ {msg}")
        
        # Load spike config
        spike_config = load_spike_config(spike_root)
        print(f"  ✓ Spike config: {spike_config['spike_h']}x{spike_config['spike_w']}")
        print()
    
    # Step 2: Organize GoPro data
    print("Step 2: Organizing GoPro data...")
    for split in args.splits:
        organize_gopro_data(gopro_root, split, force=args.force)
    print()
    
    # Step 3: Verify alignment between GoPro and Spike
    print("Step 3: Verifying GoPro-Spike alignment...")
    for split in args.splits:
        verify_gopro_spike_alignment(gopro_root, spike_root, split)
    print()
    
    # Step 4: Generate meta info files
    print("Step 4: Generating meta info files...")
    for split in args.splits:
        generate_meta_info(gopro_root, split)
    print()
    
    # Step 5: Generate LMDB (optional)
    if args.generate_lmdb:
        print("Step 5: Generating LMDB files...")
        for split in args.splits:
            generate_lmdb(gopro_root, split)
        print()
    
    print("=" * 70)
    print("Dataset preparation complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Verify the organized data structure:")
    print(f"     - {gopro_root}/train_GT/")
    print(f"     - {gopro_root}/train_GT_blurred/")
    print(f"     - {spike_root}/train/")
    print()
    print("  2. Update your training config to point to these paths")
    print()
    print("  3. Start training:")
    print("     bash launch_train.sh 1")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

