#!/usr/bin/env python3
"""
Prepare GoPro+Spike dataset structure by creating symbolic links to blur/sharp images
and spike .dat files, without needing to precompute voxels.
"""
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare GoPro+Spike dataset structure")
    parser.add_argument(
        "--spike_root",
        type=str,
        default="data/raw/gopro_spike/GOPRO_Large_spike_seq",
        help="Root directory of GoPro spike data"
    )
    parser.add_argument(
        "--blur_root",
        type=str,
        default="data/raw/gopro_spike/GOPRO_Large",
        help="Root directory of GoPro blur images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/gopro_spike_unified",
        help="Output directory for unified dataset"
    )
    parser.add_argument(
        "--use_symlinks",
        action="store_true",
        help="Use symbolic links instead of copying files"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    spike_root = Path(args.spike_root)
    blur_root = Path(args.blur_root)
    output_root = Path(args.output)
    
    if not spike_root.exists():
        print(f"Error: Spike root not found: {spike_root}")
        return 1
    
    if not blur_root.exists():
        print(f"Error: Blur root not found: {blur_root}")
        return 1
    
    print(f"Preparing GoPro+Spike dataset structure...")
    print(f"  Spike data: {spike_root}")
    print(f"  Blur data: {blur_root}")
    print(f"  Output: {output_root}")
    print(f"  Use symlinks: {args.use_symlinks}")
    
    for split in ["train", "test"]:
        spike_split_dir = spike_root / split
        blur_split_dir = blur_root / split
        
        if not spike_split_dir.exists():
            print(f"Warning: Spike {split} directory not found, skipping")
            continue
        
        if not blur_split_dir.exists():
            print(f"Warning: Blur {split} directory not found, skipping")
            continue
        
        output_split_dir = output_root / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each sequence
        spike_sequences = sorted([d for d in spike_split_dir.iterdir() if d.is_dir()])
        
        print(f"\nProcessing {split} split ({len(spike_sequences)} sequences)...")
        
        for spike_seq_dir in tqdm(spike_sequences):
            seq_name = spike_seq_dir.name
            
            # Find corresponding blur directory
            # GoPro structure: train/GOPR0372_07_00 contains blur/ and sharp/
            blur_seq_dir = blur_split_dir / seq_name
            
            if not blur_seq_dir.exists():
                print(f"Warning: Blur sequence not found: {blur_seq_dir}")
                continue
            
            blur_img_dir = blur_seq_dir / "blur"
            sharp_img_dir = blur_seq_dir / "sharp"
            spike_data_dir = spike_seq_dir / "spike"
            
            if not blur_img_dir.exists() or not sharp_img_dir.exists():
                print(f"Warning: Missing blur/sharp in: {blur_seq_dir}")
                continue
            
            if not spike_data_dir.exists():
                print(f"Warning: Missing spike data in: {spike_seq_dir}")
                continue
            
            # Create output structure
            out_seq_dir = output_split_dir / seq_name
            out_seq_dir.mkdir(parents=True, exist_ok=True)
            
            out_blur_dir = out_seq_dir / "blur"
            out_sharp_dir = out_seq_dir / "sharp"
            out_spike_dir = out_seq_dir / "spike"
            
            # Create symlinks or copy directories
            if args.use_symlinks:
                # Remove existing links if any
                if out_blur_dir.exists() or out_blur_dir.is_symlink():
                    out_blur_dir.unlink(missing_ok=True)
                if out_sharp_dir.exists() or out_sharp_dir.is_symlink():
                    out_sharp_dir.unlink(missing_ok=True)
                if out_spike_dir.exists() or out_spike_dir.is_symlink():
                    out_spike_dir.unlink(missing_ok=True)
                
                # Create symlinks
                out_blur_dir.symlink_to(blur_img_dir.absolute())
                out_sharp_dir.symlink_to(sharp_img_dir.absolute())
                out_spike_dir.symlink_to(spike_data_dir.absolute())
            else:
                # Copy directories
                if not out_blur_dir.exists():
                    shutil.copytree(blur_img_dir, out_blur_dir)
                if not out_sharp_dir.exists():
                    shutil.copytree(sharp_img_dir, out_sharp_dir)
                if not out_spike_dir.exists():
                    shutil.copytree(spike_data_dir, out_spike_dir)
    
    print(f"\n✓ Dataset structure prepared at: {output_root}")
    print(f"\nYou can now train with:")
    print(f"  python src/train.py --config configs/deblur/vrt_spike_baseline.yaml")
    return 0


if __name__ == "__main__":
    exit(main())

