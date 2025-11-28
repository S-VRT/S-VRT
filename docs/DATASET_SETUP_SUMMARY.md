# Dataset Setup Summary

## ✅ Completed Tasks

All data preparation infrastructure has been successfully set up for GoPro + Spike training.

## 📁 Created Files

### 1. Data Preparation Script
**Location**: `scripts/data_preparation/prepare_gopro_spike_dataset.py`

A comprehensive script that:
- Validates GoPro and Spike dataset structure
- Organizes frames from raw `blur/sharp` folders into `train_GT/train_GT_blurred`
- Verifies GoPro-Spike alignment (matching sequences)
- Generates meta info files
- Optionally creates LMDB databases for faster data loading

### 2. Enhanced Training Launcher
**Location**: `launch_train.sh`

Updated with integrated data preparation:
- `--prepare-data`: Automatically prepare datasets before training
- `--generate-lmdb`: Enable LMDB generation
- `--force-prepare`: Force re-preparation
- `--help`: Show usage information

### 3. Dataset Status Checker
**Location**: `scripts/data_preparation/check_dataset_status.py`

Quick diagnostic tool that shows:
- Raw dataset availability
- Prepared data status
- LMDB presence
- Recommendations for next steps

### 4. Documentation
- **`DATA_PREPARATION_GUIDE.md`**: Complete guide for dataset preparation
- **`scripts/data_preparation/README.md`**: Detailed technical documentation

## 🎯 Current Dataset Status

```
✓ GoPro raw data: 22 train + 11 test sequences
✓ Spike raw data: 22 train + 11 test sequences  
✓ Prepared training data: 22 sequences (GT + LQ)
✓ Meta info files generated
○ LMDB: Not generated yet (optional)
```

## 🚀 Quick Start Commands

### Check Dataset Status
```bash
python scripts/data_preparation/check_dataset_status.py
```

### Prepare Data Only
```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py
```

### Prepare Data + Generate LMDB
```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py --generate_lmdb
```

### Prepare Data + Start Training (All-in-One)
```bash
# Single GPU
bash launch_train.sh 1 --prepare-data

# With LMDB (recommended)
bash launch_train.sh 1 --prepare-data --generate-lmdb

# Multi-GPU
bash launch_train.sh 4 --prepare-data --generate-lmdb
```

### Start Training (Data Already Prepared)
```bash
bash launch_train.sh 1
```

## 📊 Dataset Information

### GoPro Large Dataset
- **Training**: 22 sequences, ~2,103 frames
- **Testing**: 11 sequences, ~1,111 frames
- **Resolution**: 720×1280 (PNG)
- **Location**: `/media/mallm/hd4t/modelrepostore/datasets/GOPRO_Large/`

### Spike Dataset
- **Training**: 22 sequences (matching GoPro)
- **Testing**: 11 sequences (matching GoPro)
- **Resolution**: 250×400 (binary .dat)
- **Location**: `/media/mallm/hd4t/modelrepostore/datasets/gopro_spike/GOPRO_Large_spike_seq/`

## 📂 Directory Structure

### Before Preparation (Raw)
```
GOPRO_Large/
├── train/
│   └── GOPR0374_11_00/
│       ├── blur/
│       └── sharp/
└── test/

gopro_spike/GOPRO_Large_spike_seq/
├── train/
│   └── GOPR0374_11_00/
│       └── spike/
└── test/
```

### After Preparation
```
GOPRO_Large/
├── train_GT/                  # ✓ Created
├── train_GT_blurred/          # ✓ Created
├── train_GT.lmdb/             # ○ Optional (use --generate-lmdb)
├── train_GT_blurred.lmdb/     # ○ Optional
├── test_GT/
├── test_GT_blurred/
└── ... (raw data preserved)

KAIR/data/meta_info/
├── meta_info_GoPro_train_GT.txt  # ✓ Created
└── meta_info_GoPro_test_GT.txt
```

## ⚙️ Configuration

**Training Config**: `options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike_local.json`

Key settings (already configured):
```json
{
  "dataroot_gt": "/media/mallm/hd4t/modelrepostore/datasets/GOPRO_Large/train_GT",
  "dataroot_lq": "/media/mallm/hd4t/modelrepostore/datasets/GOPRO_Large/train_GT_blurred",
  "dataroot_spike": "/media/mallm/hd4t/modelrepostore/datasets/gopro_spike/GOPRO_Large_spike_seq/train",
  "spike_h": 250,
  "spike_w": 400,
  "spike_channels": 4,
  "spike_flipud": true
}
```

## 🔧 Advanced Usage

### Force Re-preparation
```bash
bash launch_train.sh 1 --prepare-data --force-prepare
```

### Prepare Only Test Split
```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py --splits test
```

### Custom Paths
```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py \
    --gopro_root /custom/path/GOPRO_Large \
    --spike_root /custom/path/gopro_spike \
    --generate_lmdb
```

### Multi-GPU Training
```bash
# 4 GPUs
bash launch_train.sh 4 --prepare-data --generate-lmdb

# 8 GPUs with custom config
bash launch_train.sh 8 options/vrt/custom.json
```

## 🐛 Common Issues & Solutions

### Issue: "FileNotFoundError: spike file not found"
**Solution**: Path in config had extra `trainsets/` directory. This has been fixed.

### Issue: Slow data loading
**Solution**: Generate LMDB databases:
```bash
bash launch_train.sh 1 --prepare-data --generate-lmdb
```

### Issue: Need to re-prepare data
**Solution**: Use `--force-prepare`:
```bash
bash launch_train.sh 1 --prepare-data --force-prepare
```

### Issue: `meta_info_GoPro_test_GT.txt` is empty
**Solution**: The organized `test_GT/` folders exist but contain no PNG frames (often due to a previous interrupted preparation). Delete the empty folders or rerun:
```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py --splits train test --force
```
The script prints `Warning: No frames found ...` for any skipped sequence.

## 📚 Documentation References

- **Main Guide**: `DATA_PREPARATION_GUIDE.md`
- **Technical Details**: `scripts/data_preparation/README.md`
- **Training Script**: `launch_train.sh --help`
- **Preparation Script**: `scripts/data_preparation/prepare_gopro_spike_dataset.py --help`

## ✨ Features

1. **Automatic Validation**: Checks dataset structure before processing
2. **Smart Skip**: Won't re-process if data already exists (unless `--force`)
3. **Alignment Check**: Verifies GoPro and Spike sequences match
4. **Meta Info Generation**: Creates required metadata files
5. **Optional LMDB**: Fast data loading for training
6. **Integrated Workflow**: Seamless data prep + training in one command
7. **Status Checking**: Quick diagnostic tool to check dataset readiness

## 🎉 Next Steps

Your dataset is now ready! You can:

1. **Check Status**: `python scripts/data_preparation/check_dataset_status.py`
2. **Start Training**: `bash launch_train.sh 1`
3. **Or Prepare + Train**: `bash launch_train.sh 1 --prepare-data --generate-lmdb`

---

**Date**: November 6, 2025  
**Status**: ✅ All systems ready for training

