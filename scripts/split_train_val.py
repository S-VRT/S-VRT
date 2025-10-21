#!/usr/bin/env python3
"""
Split training data into train and validation sets.

Creates symbolic links to avoid duplicating data.
Typical split: 80% train, 20% val
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Split train data into train/val")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/processed/gopro_spike_unified",
        help="Root directory of processed data"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Ratio of validation data (default: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing val directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    
    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return 1
    
    # Get all train sequences
    sequences = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"📊 Found {len(sequences)} training sequences")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    random.shuffle(sequences)
    
    # Split
    num_val = max(1, int(len(sequences) * args.val_ratio))
    num_train = len(sequences) - num_val
    
    val_sequences = sequences[:num_val]
    train_sequences = sequences[num_val:]
    
    print(f"\n📈 Split:")
    print(f"  Train: {num_train} sequences ({100*(1-args.val_ratio):.0f}%)")
    print(f"  Val:   {num_val} sequences ({100*args.val_ratio:.0f}%)")
    
    # Create val directory
    if val_dir.exists():
        if not args.force:
            print(f"\n⚠️  Val directory already exists: {val_dir}")
            print("   Use --force to overwrite")
            return 1
        else:
            print(f"\n🗑️  Removing existing val directory...")
            shutil.rmtree(val_dir)
    
    val_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✅ Created val directory: {val_dir}")
    
    # Create symbolic links for validation sequences
    print(f"\n🔗 Creating symbolic links for validation sequences...")
    for seq in val_sequences:
        src = seq.absolute()
        dst = val_dir / seq.name
        dst.symlink_to(src)
        print(f"  ✓ {seq.name}")
    
    print(f"\n✅ Validation split created successfully!")
    print(f"\n📝 Validation sequences:")
    for seq in val_sequences:
        print(f"  - {seq.name}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Update config file to use VAL_SPLIT: val")
    print(f"   2. Run tests: python src/test.py --config configs/deblur/vrt_spike_baseline.yaml")
    
    return 0


if __name__ == "__main__":
    exit(main())





