#!/usr/bin/env python3
"""
Generate meta_info file for GoPro dataset

Meta info format for GoPro (VRT specific):
folder_name num_frames start_frame end_frame

Example:
GOPR0372_07_00 100 0 99
GOPR0868_11_02 100 681 780
"""

import os
import sys
import argparse
from pathlib import Path


def get_frame_range(folder_path):
    """Get the frame range for a video folder"""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    if not files:
        return None, None, 0
    
    # Extract frame numbers from filenames
    first_frame = int(files[0].replace('.png', ''))
    last_frame = int(files[-1].replace('.png', ''))
    num_frames = len(files)
    
    return first_frame, last_frame, num_frames


def generate_meta_info(input_dir, output_file):
    """Generate meta info file for the dataset"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Get all subdirectories
    folders = sorted([d for d in os.listdir(input_dir) 
                     if os.path.isdir(os.path.join(input_dir, d))])
    
    if not folders:
        print(f"Error: No subdirectories found in {input_dir}")
        return
    
    print(f"Found {len(folders)} folders")
    
    # Generate meta info
    meta_info = []
    for folder in folders:
        folder_path = os.path.join(input_dir, folder)
        first_frame, last_frame, num_frames = get_frame_range(folder_path)
        
        if num_frames == 0:
            print(f"Warning: No PNG files found in {folder}")
            continue
        
        # Format: folder_name num_frames start_frame end_frame
        meta_info.append(f"{folder} {num_frames} {first_frame} {last_frame}\n")
        print(f"  {folder}: {num_frames} frames ({first_frame} to {last_frame})")
    
    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.writelines(meta_info)
    
    print(f"\nMeta info file generated: {output_file}")
    print(f"Total sequences: {len(meta_info)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate meta_info file for GoPro dataset')
    parser.add_argument('input_dir', help='Input directory containing video folders')
    parser.add_argument('output_file', help='Output meta_info file path')
    
    args = parser.parse_args()
    
    generate_meta_info(args.input_dir, args.output_file)

