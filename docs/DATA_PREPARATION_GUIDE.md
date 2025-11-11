# GoPro + Spike Dataset Preparation Guide

This guide explains how to prepare the GoPro video deblurring dataset with Spike camera data for VRT training.

## Quick Start

### Simple Training (Data already prepared)

If your data is already organized in `train_GT/`, `train_GT_blurred/` folders:

```bash
bash launch_train.sh 1
```

### Training with Automatic Data Preparation

If you just unzipped the raw datasets:

```bash
# Prepare data and start training
bash launch_train.sh 1 --prepare-data

# With LMDB generation (recommended for faster loading)
bash launch_train.sh 1 --prepare-data --generate-lmdb
```

## Dataset Setup

### 1. Download and Extract Datasets

You need two datasets:

1. **GoPro Large Dataset**: RGB video frames (blur + sharp)
   - Extract to: `/media/mallm/hd4t/modelrepostore/datasets/GOPRO_Large/`
   - Should contain `train/` and `test/` folders

2. **GoPro Spike Dataset**: Corresponding spike camera data
   - Extract to: `/media/mallm/hd4t/modelrepostore/datasets/gopro_spike/GOPRO_Large_spike_seq/`
   - Should contain `train/`, `test/` folders and `config.yaml`

### 2. Raw Dataset Structure (After Extraction)

```
GOPRO_Large/
├── train/                     # 22 sequences
│   ├── GOPR0374_11_00/
│   │   ├── blur/              # Blurred frames (100-150 frames per sequence)
│   │   └── sharp/             # Sharp GT frames
│   └── ...
└── test/                      # 11 sequences
    └── ...

gopro_spike/GOPRO_Large_spike_seq/
├── train/                     # 22 sequences (matching GoPro)
│   ├── GOPR0374_11_00/
│   │   └── spike/             # Spike data (.dat files, 250x400)
│   └── ...
├── test/                      # 11 sequences
└── config.yaml                # spike_h: 250, spike_w: 400
```

### 3. Run Data Preparation

#### Option A: Integrated with Training (Recommended)

```bash
# Basic preparation + training
bash launch_train.sh 1 --prepare-data

# With LMDB for faster data loading
bash launch_train.sh 1 --prepare-data --generate-lmdb

# Multi-GPU training
bash launch_train.sh 4 --prepare-data --generate-lmdb
```

#### Option B: Standalone Preparation

```bash
# Prepare train + test splits
python scripts/data_preparation/prepare_gopro_spike_dataset.py

# With LMDB generation
python scripts/data_preparation/prepare_gopro_spike_dataset.py --generate_lmdb

# Custom paths
python scripts/data_preparation/prepare_gopro_spike_dataset.py \
    --gopro_root /path/to/GOPRO_Large \
    --spike_root /path/to/gopro_spike \
    --generate_lmdb
```

### 4. Verify Prepared Data

After preparation, you should have:

```
GOPRO_Large/
├── train_GT/                  # Organized sharp frames (22 sequences)
├── train_GT_blurred/          # Organized blurred frames (22 sequences)
├── train_GT.lmdb/             # (Optional) LMDB database
├── train_GT_blurred.lmdb/     # (Optional) LMDB database
├── test_GT/
├── test_GT_blurred/
└── ... (original train/test folders remain)

gopro_spike/GOPRO_Large_spike_seq/
├── train/                     # Spike data (unchanged)
└── test/

KAIR/data/meta_info/
├── meta_info_GoPro_train_GT.txt
└── meta_info_GoPro_test_GT.txt
```

## What the Preparation Script Does

1. **Validates** dataset structure (both GoPro and Spike)
2. **Organizes** frames from `blur/sharp` folders into `train_GT/train_GT_blurred`
3. **Verifies** GoPro and Spike datasets have matching sequences
4. **Generates** meta info files listing all sequences and frame counts
5. **Creates** LMDB databases (optional, for faster data loading)

## Configuration

The training configuration is in:
```
options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike_local.json
```

Key dataset settings:
```json
{
  "dataroot_gt": "/media/mallm/hd4t/modelrepostore/datasets/GOPRO_Large/train_GT",
  "dataroot_lq": "/media/mallm/hd4t/modelrepostore/datasets/GOPRO_Large/train_GT_blurred",
  "dataroot_spike": "/media/mallm/hd4t/modelrepostore/datasets/gopro_spike/GOPRO_Large_spike_seq/train",
  "spike_h": 250,
  "spike_w": 400,
  "spike_channels": 1,
  "spike_flipud": true
}
```

## Training Commands

### Single GPU

```bash
# Without data preparation (data already ready)
bash launch_train.sh 1

# With data preparation
bash launch_train.sh 1 --prepare-data --generate-lmdb
```

### Multi-GPU

```bash
# 4 GPUs
bash launch_train.sh 4 --prepare-data --generate-lmdb

# 8 GPUs with custom config
bash launch_train.sh 8 options/vrt/custom.json
```

### Custom Configuration

```bash
bash launch_train.sh 1 options/vrt/my_config.json --prepare-data
```

## Troubleshooting

### Data Preparation Issues

**Issue**: "GoPro validation failed"
- **Solution**: Ensure GOPRO_Large is extracted with `train/` and `test/` folders
- Each sequence folder should contain `blur/` and `sharp/` subfolders

**Issue**: "Spike validation failed"  
- **Solution**: Ensure gopro_spike is extracted with `train/`, `test/`, and `config.yaml`
- Each sequence folder should contain a `spike/` subfolder with `.dat` files

**Issue**: "No common sequences found"
- **Solution**: Sequence names must match between GoPro and Spike datasets
- Verify both datasets are from the same source and version

**Issue**: `meta_info_GoPro_test_GT.txt` (or train) is empty / reports `0 sequences`
- **Solution**: The preparation script skips any sequence folder that has no PNG frames. This usually means `*_GT/` and `*_GT_blurred/` already existed but were empty or partially copied.
- Delete the stale folders or rerun with `--force` to rebuild them:
  ```bash
  python scripts/data_preparation/prepare_gopro_spike_dataset.py --splits train test --force
  ```
- After forcing regeneration, re-run the script (or `launch_train.sh ... --prepare-data`) and re-check the meta info file.

### Training Issues

**Issue**: "FileNotFoundError" during training
- **Solution**: Run data preparation first: `bash launch_train.sh 1 --prepare-data`

**Issue**: Slow data loading during training
- **Solution**: Use LMDB: `bash launch_train.sh 1 --prepare-data --generate-lmdb`

**Issue**: "spike file not found"
- **Solution**: Check `dataroot_spike` path in config matches your spike data location
- Ensure path does NOT include extra `/trainsets/` directory

## Advanced Options

### Force Re-preparation

If you need to regenerate data:

```bash
bash launch_train.sh 1 --prepare-data --force-prepare
```

### Prepare Only Test Data

```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py --splits test
```

### Skip Validation

If you're sure your data is correct:

```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py --skip_validation
```

## Dataset Statistics

- **Training**: 22 sequences, ~2,103 frames total
- **Testing**: 11 sequences, ~1,111 frames total
- **RGB Resolution**: 720×1280 (PNG format)
- **Spike Resolution**: 250×400 (binary .dat format)
- **Spike Channels**: 1 (binary events)

## File Organization

```
KAIR/
├── launch_train.sh                          # Main training launcher
├── scripts/
│   └── data_preparation/
│       ├── prepare_gopro_spike_dataset.py   # Data preparation script
│       ├── create_lmdb_for_vrt.py           # LMDB generation
│       └── README.md                        # Detailed documentation
├── options/
│   └── vrt/
│       └── 006_train_vrt_videodeblurring_gopro_rgbspike_local.json
└── data/
    └── meta_info/
        ├── meta_info_GoPro_train_GT.txt
        └── meta_info_GoPro_test_GT.txt
```

## Support

For more details, see:
- `scripts/data_preparation/README.md` - Detailed preparation documentation
- `options/vrt/*.json` - Training configurations
- `main_train_vrt.py` - Training entry point

## Summary

**Simplest workflow:**
1. Extract both datasets to default locations
2. Run: `bash launch_train.sh 1 --prepare-data --generate-lmdb`
3. Training starts automatically after data preparation completes

**Already have prepared data:**
1. Run: `bash launch_train.sh 1`
2. Training starts immediately

