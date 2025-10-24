# Dataset Preprocessing Guide

This guide covers the unified preprocessing pipeline for spike-based deblurring datasets.

## Table of Contents

1. [Overview](#overview)
2. [Supported Datasets](#supported-datasets)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Command-Line Interface](#command-line-interface)
6. [Python API](#python-api)
7. [Auto-Preprocessing in Training](#auto-preprocessing-in-training)
8. [Configuration](#configuration)
9. [Directory Structure](#directory-structure)
10. [Troubleshooting](#troubleshooting)

## Overview

The preprocessing pipeline handles:
- **Data organization**: Restructuring raw datasets into a unified format
- **Spike voxelization**: Converting spike streams into dense temporal representations
- **Statistics computation**: Computing dataset mean/std for normalization
- **Configuration updates**: Automatically updating YAML configs with computed statistics
- **Validation**: Checking data integrity and completeness

## Supported Datasets

### GoPro-Spike Dataset
- **Source**: GoPro dataset with spike camera captures
- **Raw format**: 
  - Blurry/sharp RGB images (`.png`)
  - Spike data in `.dat` files (binary format)
- **Preprocessing**:
  - Organizes images into `blurry/` and `sharp/` directories
  - Reads and voxelizes spike `.dat` files
  - Computes normalization statistics

### X4K1000FPS Dataset
- **Source**: High-speed video dataset
- **Raw format**:
  - High-FPS sharp frames
  - Spike streams
- **Preprocessing**:
  - Synthesizes blurry images from sharp sequences
  - Aligns spike data with RGB frames
  - Voxelizes aligned spike data
  - Computes normalization statistics

## Installation

The preprocessing module is part of the main project. Ensure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Check if dataset is ready

```bash
python -m src.data.preprocessing \
    --dataset gopro \
    --data-root /path/to/raw/gopro \
    --output-root data/processed/gopro_unified \
    --action check
```

### Prepare dataset (full pipeline)

```bash
python -m src.data.preprocessing \
    --dataset gopro \
    --data-root /path/to/raw/gopro \
    --output-root data/processed/gopro_unified \
    --config configs/deblur/vrt_spike_baseline.yaml \
    --action prepare
```

### Compute statistics only

```bash
python -m src.data.preprocessing \
    --dataset gopro \
    --data-root /path/to/raw/gopro \
    --output-root data/processed/gopro_unified \
    --config configs/deblur/vrt_spike_baseline.yaml \
    --action stats
```

## Command-Line Interface

### Basic Usage

```bash
python -m src.data.preprocessing [OPTIONS]
```

### Required Arguments

- `--dataset {gopro,x4k}`: Dataset type to preprocess
- `--data-root PATH`: Path to raw dataset root directory
- `--output-root PATH`: Path to output directory for preprocessed data

### Optional Arguments

- `--action {prepare,check,stats}`: Action to perform (default: `prepare`)
  - `prepare`: Full preprocessing pipeline
  - `check`: Validate dataset readiness
  - `stats`: Compute statistics only
- `--config PATH`: Path to YAML config file to update with statistics
- `--splits SPLIT [SPLIT ...]`: Splits to process (e.g., `train val`)
- `--force`: Force overwrite existing preprocessed data
- `--num-bins INT`: Number of temporal bins for voxelization (default: 32)

### GoPro-Specific Parameters

- `--spike-frames INT`: Number of temporal frames in spike `.dat` files (default: 10)
- `--spike-height INT`: Height of spike data (default: 396)
- `--spike-width INT`: Width of spike data (default: 640)

### X4K-Specific Parameters

- `--fps INT`: Source FPS for spike/blurry alignment (default: 1000)
- `--exposure-frames INT`: Exposure frames for blurry synthesis (default: 33)

### Examples

#### GoPro Dataset

```bash
# Full preprocessing with custom parameters
python -m src.data.preprocessing \
    --dataset gopro \
    --data-root /mnt/data/gopro_spike \
    --output-root data/processed/gopro_unified \
    --config configs/deblur/vrt_spike_baseline.yaml \
    --spike-frames 10 \
    --spike-height 396 \
    --spike-width 640 \
    --num-bins 32 \
    --action prepare

# Process only training split
python -m src.data.preprocessing \
    --dataset gopro \
    --data-root /mnt/data/gopro_spike \
    --output-root data/processed/gopro_unified \
    --splits train \
    --action prepare

# Force recompute all data
python -m src.data.preprocessing \
    --dataset gopro \
    --data-root /mnt/data/gopro_spike \
    --output-root data/processed/gopro_unified \
    --action prepare \
    --force
```

#### X4K Dataset

```bash
# Full preprocessing
python -m src.data.preprocessing \
    --dataset x4k \
    --data-root /mnt/data/x4k1000fps \
    --output-root data/processed/x4k_unified \
    --config configs/deblur/vrt_spike_baseline.yaml \
    --fps 1000 \
    --exposure-frames 33 \
    --num-bins 32 \
    --action prepare

# Check readiness
python -m src.data.preprocessing \
    --dataset x4k \
    --data-root /mnt/data/x4k1000fps \
    --output-root data/processed/x4k_unified \
    --action check
```

## Python API

### Using the Factory Function

```python
from pathlib import Path
from src.data.preprocessing import get_preprocessor

# Create preprocessor
preprocessor = get_preprocessor(
    dataset_type="gopro",
    data_root=Path("/path/to/raw/gopro"),
    output_root=Path("data/processed/gopro_unified"),
    config_path=Path("configs/deblur/vrt_spike_baseline.yaml"),
    spike_frames=10,
    spike_height=396,
    spike_width=640,
    num_bins=32,
)

# Check if ready
is_ready = preprocessor.check_ready()

# Prepare data
if not is_ready:
    preprocessor.prepare(force=False)

# Compute statistics
stats = preprocessor.compute_stats()
print(f"Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")

# Update config
preprocessor.update_config(stats)
```

### Direct Class Usage

```python
from pathlib import Path
from src.data.preprocessing.datasets import GoProPreprocessor

# Create preprocessor directly
preprocessor = GoProPreprocessor(
    data_root=Path("/path/to/raw/gopro"),
    output_root=Path("data/processed/gopro_unified"),
    config_path=Path("configs/deblur/vrt_spike_baseline.yaml"),
    spike_frames=10,
    spike_height=396,
    spike_width=640,
    num_bins=32,
)

# Use the same API as above
preprocessor.prepare()
```

## Auto-Preprocessing in Training

You can enable automatic preprocessing checks and execution in the training script by configuring your YAML file.

### Configuration

Add the following to your YAML config (e.g., `configs/deblur/vrt_spike_baseline.yaml`):

```yaml
DATA:
  ROOT: data/processed/gopro_spike_unified
  
  PREPROCESSING:
    AUTO_PREPARE: true          # Enable auto-preprocessing
    DATASET_TYPE: "gopro"       # Dataset type: "gopro" or "x4k"
    FORCE_RECOMPUTE: false      # Force recompute existing data
    
    # Voxelization settings
    VOXEL:
      NUM_BINS: 32              # Number of temporal bins
      APPLY_LOG1P: true         # Apply log1p normalization
      CACHE_DIRNAME: "spike_vox"  # Cache directory name
    
    # GoPro-specific settings
    GOPRO:
      SPIKE_TEMPORAL_FRAMES: 10
      SPIKE_HEIGHT: 396
      SPIKE_WIDTH: 640
    
    # X4K-specific settings (alternative to GOPRO)
    X4K:
      FPS: 1000
      EXPOSURE_FRAMES: 33
```

### Behavior

When `AUTO_PREPARE: true`:
1. Training script checks if data is preprocessed before loading datasets
2. If not ready or `FORCE_RECOMPUTE: true`, runs full preprocessing pipeline
3. Only the main process (rank 0) performs preprocessing
4. Other processes wait via `dist.barrier()`
5. Training continues with preprocessed data

### Raw Data Location

By default, the training script assumes raw data is located at:
```
{output_root}/../raw/{dataset_type}/
```

For example, if `DATA.ROOT` is `data/processed/gopro_spike_unified`, raw data should be at:
```
data/raw/gopro/
```

## Configuration

### YAML Config Structure

The preprocessing module reads/updates the following sections:

```yaml
DATA:
  NUM_VOXEL_BINS: 32           # Updated by preprocessing
  VOXEL_CACHE_DIRNAME: spike_vox
  NORM:
    MEAN: 0.123456             # Computed by preprocessing
    STD: 0.234567              # Computed by preprocessing
  
  PREPROCESSING:
    AUTO_PREPARE: false
    DATASET_TYPE: "gopro"
    FORCE_RECOMPUTE: false
    
    VOXEL:
      NUM_BINS: 32
      APPLY_LOG1P: true
      CACHE_DIRNAME: "spike_vox"
    
    GOPRO:
      SPIKE_TEMPORAL_FRAMES: 10
      SPIKE_HEIGHT: 396
      SPIKE_WIDTH: 640
    
    X4K:
      FPS: 1000
      EXPOSURE_FRAMES: 33
```

## Directory Structure

### Expected Raw Dataset Structure

#### GoPro

```
/path/to/raw/gopro/
├── train/
│   ├── GOPR0372_07_00/
│   │   ├── blur/
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   ├── sharp/
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   └── spike/
│   │       ├── 000001.dat
│   │       └── ...
│   └── ...
└── val/
    └── ...
```

#### X4K

```
/path/to/raw/x4k/
├── train/
│   ├── scene_001/
│   │   ├── sharp/
│   │   │   ├── 00000.png
│   │   │   └── ...
│   │   └── spike/
│   │       ├── 00000.npy
│   │       └── ...
│   └── ...
└── val/
    └── ...
```

### Output Structure After Preprocessing

```
data/processed/{dataset}_unified/
├── train/
│   ├── blurry/
│   │   ├── scene_000_frame_000.png
│   │   └── ...
│   ├── sharp/
│   │   ├── scene_000_frame_000.png
│   │   └── ...
│   └── spike_vox/
│       ├── scene_000_frame_000.npy
│       └── ...
├── val/
│   ├── blurry/
│   ├── sharp/
│   └── spike_vox/
└── stats.json  # Dataset statistics
```

## Troubleshooting

### Common Issues

#### 1. "Data root not found"

**Problem**: Raw dataset directory doesn't exist.

**Solution**: 
```bash
# Verify path exists
ls -l /path/to/raw/dataset

# Use absolute paths
python -m src.data.preprocessing \
    --data-root /absolute/path/to/raw/dataset \
    --output-root /absolute/path/to/output
```

#### 2. "No spike files found"

**Problem**: Spike directory structure doesn't match expected format.

**Solution**: 
- Verify spike files are in the correct location
- Check file extensions (`.dat` for GoPro, `.npy` for X4K)
- Use `--spike-dir` parameter if spike files are in a different subdirectory

#### 3. Memory Issues During Voxelization

**Problem**: Out of memory when processing large datasets.

**Solution**:
```python
# Process splits separately
python -m src.data.preprocessing \
    --dataset gopro \
    --data-root /path/to/raw \
    --output-root /path/to/output \
    --splits train \
    --action prepare

python -m src.data.preprocessing \
    --dataset gopro \
    --data-root /path/to/raw \
    --output-root /path/to/output \
    --splits val \
    --action prepare
```

#### 4. Statistics Not Updating in Config

**Problem**: YAML config not being updated with computed statistics.

**Solution**:
- Ensure `--config` path is correct
- Check file permissions (needs write access)
- Verify YAML file has correct structure
- Use `--action stats` to compute and update without full preprocessing

#### 5. Training Script Can't Find Preprocessed Data

**Problem**: Training fails to load preprocessed data.

**Solution**:
- Verify `DATA.ROOT` in config matches `--output-root` from preprocessing
- Check that preprocessing completed successfully (no errors in logs)
- Use `--action check` to validate data:
  ```bash
  python -m src.data.preprocessing \
      --dataset gopro \
      --data-root /path/to/raw \
      --output-root data/processed/gopro_unified \
      --action check
  ```

### Debug Mode

For detailed debugging, modify the preprocessor to print verbose output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.data.preprocessing import get_preprocessor

preprocessor = get_preprocessor(...)
preprocessor.prepare()
```

### Getting Help

For more help:
1. Check the test script: `scripts/test_preprocessing.py`
2. Review example configurations in `configs/deblur/`
3. Examine the source code in `src/data/preprocessing/`
4. Open an issue on GitHub with:
   - Command used
   - Full error message
   - Dataset structure
   - System information

## Performance Tips

1. **Use SSD storage**: Significantly faster for I/O-heavy voxelization
2. **Sufficient RAM**: Have at least 32GB for large datasets
3. **Parallel processing**: The pipeline uses multiprocessing where applicable
4. **Incremental preprocessing**: Process splits separately if memory-constrained
5. **Cache reuse**: Don't use `--force` unless data needs to be regenerated

## Advanced Usage

### Custom Voxelization Parameters

```bash
python -m src.data.preprocessing \
    --dataset gopro \
    --data-root /path/to/raw \
    --output-root /path/to/output \
    --num-bins 64 \  # Increase temporal resolution
    --action prepare
```

### Processing Only Statistics

Useful when voxelization is already done:

```bash
python -m src.data.preprocessing \
    --dataset gopro \
    --data-root /path/to/raw \
    --output-root /path/to/output \
    --config configs/deblur/vrt_spike_baseline.yaml \
    --action stats
```

### Batch Processing Multiple Datasets

```bash
#!/bin/bash

DATASETS=("gopro" "x4k")
RAW_ROOT="/mnt/data/raw"
OUTPUT_ROOT="data/processed"

for dataset in "${DATASETS[@]}"; do
    python -m src.data.preprocessing \
        --dataset "$dataset" \
        --data-root "$RAW_ROOT/$dataset" \
        --output-root "$OUTPUT_ROOT/${dataset}_unified" \
        --config configs/deblur/vrt_spike_baseline.yaml \
        --action prepare
done
```



