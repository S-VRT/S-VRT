#!/usr/bin/env python3
"""
Test script for the unified preprocessing pipeline.

This script demonstrates how to use the preprocessing module for both
GoPro and X4K datasets, and verifies the output structure.
"""

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import get_preprocessor


def test_gopro_preprocessing():
    """Test GoPro preprocessing pipeline."""
    print("=" * 80)
    print("Testing GoPro Dataset Preprocessing")
    print("=" * 80)
    
    # Example configuration
    data_root = REPO_ROOT / "data" / "raw" / "gopro_spike"
    output_root = REPO_ROOT / "data" / "processed" / "gopro_spike_unified"
    config_path = REPO_ROOT / "configs" / "deblur" / "vrt_spike_baseline.yaml"
    
    print(f"\nConfiguration:")
    print(f"  Data Root: {data_root}")
    print(f"  Output Root: {output_root}")
    print(f"  Config: {config_path}")
    
    # Create preprocessor
    preprocessor = get_preprocessor(
        dataset_type="gopro",
        data_root=data_root,
        output_root=output_root,
        config_path=config_path if config_path.exists() else None,
        spike_frames=10,
        spike_height=396,
        spike_width=640,
        num_bins=32,
    )
    
    # Check if data is ready
    print(f"\nChecking if dataset is ready...")
    is_ready = preprocessor.check_ready()
    
    if is_ready:
        print(f"✓ Dataset is ready for training!")
        return True
    else:
        print(f"✗ Dataset is not ready.")
        print(f"\nTo prepare the dataset, run:")
        print(f"  python -m src.data.preprocessing --dataset gopro \\")
        print(f"      --data-root {data_root} \\")
        print(f"      --output-root {output_root} \\")
        print(f"      --config {config_path} \\")
        print(f"      --action prepare")
        return False


def test_x4k_preprocessing():
    """Test X4K preprocessing pipeline."""
    print("\n" + "=" * 80)
    print("Testing X4K Dataset Preprocessing")
    print("=" * 80)
    
    # Example configuration
    data_root = REPO_ROOT / "data" / "raw" / "x4k"
    output_root = REPO_ROOT / "data" / "processed" / "x4k_unified"
    config_path = REPO_ROOT / "configs" / "deblur" / "vrt_spike_baseline.yaml"
    
    print(f"\nConfiguration:")
    print(f"  Data Root: {data_root}")
    print(f"  Output Root: {output_root}")
    print(f"  Config: {config_path}")
    
    # Check if raw data exists
    if not data_root.exists():
        print(f"\n⚠ Raw data directory not found: {data_root}")
        print(f"  Skipping X4K test.")
        print(f"\nTo test X4K preprocessing, ensure raw data is available at:")
        print(f"  {data_root}")
        return None
    
    # Create preprocessor
    try:
        preprocessor = get_preprocessor(
            dataset_type="x4k",
            data_root=data_root,
            output_root=output_root,
            config_path=config_path if config_path.exists() else None,
            fps=1000,
            exposure_frames=33,
            num_bins=32,
        )
        
        # Check if data is ready
        print(f"\nChecking if dataset is ready...")
        is_ready = preprocessor.check_ready()
    except Exception as e:
        print(f"\n✗ Error creating preprocessor: {e}")
        return None
    
    if is_ready:
        print(f"✓ Dataset is ready for training!")
        return True
    else:
        print(f"✗ Dataset is not ready.")
        print(f"\nTo prepare the dataset, run:")
        print(f"  python -m src.data.preprocessing --dataset x4k \\")
        print(f"      --data-root {data_root} \\")
        print(f"      --output-root {output_root} \\")
        print(f"      --config {config_path} \\")
        print(f"      --action prepare")
        return False


def print_usage_examples():
    """Print common usage examples."""
    print("\n" + "=" * 80)
    print("Preprocessing Module Usage Examples")
    print("=" * 80)
    
    print("\n1. Check if GoPro dataset is ready:")
    print("   python -m src.data.preprocessing --dataset gopro \\")
    print("       --data-root /path/to/gopro --output-root data/processed/gopro_unified \\")
    print("       --action check")
    
    print("\n2. Prepare GoPro dataset (full pipeline):")
    print("   python -m src.data.preprocessing --dataset gopro \\")
    print("       --data-root /path/to/gopro --output-root data/processed/gopro_unified \\")
    print("       --config configs/deblur/vrt_spike_baseline.yaml \\")
    print("       --action prepare")
    
    print("\n3. Compute statistics only (no voxelization):")
    print("   python -m src.data.preprocessing --dataset gopro \\")
    print("       --data-root /path/to/gopro --output-root data/processed/gopro_unified \\")
    print("       --config configs/deblur/vrt_spike_baseline.yaml \\")
    print("       --action stats")
    
    print("\n4. Prepare X4K dataset with custom parameters:")
    print("   python -m src.data.preprocessing --dataset x4k \\")
    print("       --data-root /path/to/x4k --output-root data/processed/x4k_unified \\")
    print("       --config configs/deblur/vrt_spike_baseline.yaml \\")
    print("       --fps 1000 --exposure-frames 33 --num-bins 32 \\")
    print("       --action prepare")
    
    print("\n5. Force recompute existing preprocessed data:")
    print("   python -m src.data.preprocessing --dataset gopro \\")
    print("       --data-root /path/to/gopro --output-root data/processed/gopro_unified \\")
    print("       --action prepare --force")
    
    print("\n6. Process specific splits only:")
    print("   python -m src.data.preprocessing --dataset gopro \\")
    print("       --data-root /path/to/gopro --output-root data/processed/gopro_unified \\")
    print("       --splits train val \\")
    print("       --action prepare")
    
    print("\n" + "=" * 80)
    print("Auto-Preprocessing in Training")
    print("=" * 80)
    
    print("\nTo enable auto-preprocessing in training, set in your YAML config:")
    print("""
DATA:
  ROOT: data/processed/gopro_spike_unified
  PREPROCESSING:
    AUTO_PREPARE: true        # Enable auto-preprocessing
    DATASET_TYPE: "gopro"     # or "x4k"
    FORCE_RECOMPUTE: false    # Set to true to force recompute
    
    VOXEL:
      NUM_BINS: 32
      APPLY_LOG1P: true
      CACHE_DIRNAME: "spike_vox"
    
    GOPRO:  # GoPro-specific settings
      SPIKE_TEMPORAL_FRAMES: 10
      SPIKE_HEIGHT: 396
      SPIKE_WIDTH: 640
    
    # Or for X4K:
    X4K:
      FPS: 1000
      EXPOSURE_FRAMES: 33
""")
    
    print("\n" + "=" * 80)
    print("Expected Directory Structure After Preprocessing")
    print("=" * 80)
    
    print("""
For GoPro:
  data/processed/gopro_spike_unified/
  ├── train/
  │   ├── blurry/         # Blurry RGB images
  │   ├── sharp/          # Sharp RGB images
  │   └── spike_vox/      # Voxelized spike data (.npy)
  └── val/
      ├── blurry/
      ├── sharp/
      └── spike_vox/

For X4K:
  data/processed/x4k_unified/
  ├── train/
  │   ├── blurry/         # Synthesized blurry images
  │   ├── sharp/          # Original sharp frames
  │   └── spike_vox/      # Voxelized spike data (.npy)
  └── val/
      ├── blurry/
      ├── sharp/
      └── spike_vox/
""")


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("Unified Preprocessing Pipeline Test")
    print("=" * 80)
    
    # Test both datasets
    gopro_ready = test_gopro_preprocessing()
    x4k_ready = test_x4k_preprocessing()
    
    # Print usage examples
    print_usage_examples()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"  GoPro Dataset: {'✓ Ready' if gopro_ready else ('✗ Not Ready' if gopro_ready is False else '⚠ Skipped')}")
    print(f"  X4K Dataset:   {'✓ Ready' if x4k_ready else ('✗ Not Ready' if x4k_ready is False else '⚠ Skipped')}")
    print("=" * 80)
    
    # Return 0 if at least one dataset was tested successfully
    tested = [x for x in [gopro_ready, x4k_ready] if x is not None]
    return 0 if (tested and any(tested)) else 1


if __name__ == "__main__":
    sys.exit(main())

