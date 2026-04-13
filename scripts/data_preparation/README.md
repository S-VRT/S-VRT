# GoPro + Spike Dataset Preparation

This directory contains scripts for preparing the GoPro video deblurring dataset along with its corresponding Spike camera data for VRT training.

## Quick Start

### Option 1: Using the integrated launch script (Recommended)

```bash
# Prepare data and start training in one command
bash launch_train.sh 1 --prepare-data

# Prepare data with LMDB generation (faster loading during training)
bash launch_train.sh 1 --prepare-data --generate-lmdb

# Force re-preparation (if you've made changes to the raw data)
bash launch_train.sh 1 --prepare-data --force-prepare
```

### Option 2: Manual data preparation

```bash
# Basic preparation (train + test splits)
python scripts/data_preparation/prepare_gopro_spike_dataset.py

# With LMDB generation
python scripts/data_preparation/prepare_gopro_spike_dataset.py --generate_lmdb

# Custom paths
python scripts/data_preparation/prepare_gopro_spike_dataset.py \
    --gopro_root /path/to/GOPRO_Large \
    --spike_root /path/to/gopro_spike/GOPRO_Large_spike_seq \
    --generate_lmdb
```

## Dataset Structure

### Input (Raw, after unzipping)

```
GOPRO_Large/
├── train/
│   ├── GOPR0374_11_00/
│   │   ├── blur/              # Blurred input frames
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   └── ...
│   │   ├── sharp/             # Sharp ground truth frames
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   └── ...
│   │   └── frames 11 offset 0.txt
│   ├── GOPR0374_11_01/
│   └── ... (22 sequences)
└── test/
    └── ... (11 sequences)

gopro_spike/GOPRO_Large_spike_seq/
├── train/
│   ├── GOPR0374_11_00/
│   │   └── spike/             # Spike camera data
│   │       ├── 000001.dat
│   │       ├── 000002.dat
│   │       └── ...
│   ├── GOPR0374_11_01/
│   └── ... (22 sequences)
├── test/
│   └── ... (11 sequences)
└── config.yaml                # Spike camera config (250x400)
```

### Output (After preparation)

```
GOPRO_Large/
├── train_GT/                  # Organized sharp frames
│   ├── GOPR0374_11_00/
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   └── ...
│   └── ...
├── train_GT_blurred/          # Organized blurred frames
│   ├── GOPR0374_11_00/
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   └── ...
│   └── ...
├── train_GT.lmdb/             # (Optional) LMDB for faster loading
├── train_GT_blurred.lmdb/     # (Optional) LMDB for faster loading
├── test_GT/
├── test_GT_blurred/
├── test_GT.lmdb/              # (Optional)
└── test_GT_blurred.lmdb/      # (Optional)

gopro_spike/GOPRO_Large_spike_seq/
├── train/                     # Spike data (unchanged)
│   ├── GOPR0374_11_00/
│   │   └── spike/
│   │       ├── 000001.dat
│   │       └── ...
│   └── ...
└── test/
    └── ...
```

### Meta Info Files

The script also generates meta info files in `data/meta_info/`:

- `meta_info_GoPro_train_GT.txt`: Training sequences and frame counts
- `meta_info_GoPro_test_GT.txt`: Test sequences and frame counts

Format (`folder frame_count border start_frame`):
```
GOPR0374_11_00 150 0 1
GOPR0374_11_01 80 0 203
# ...
```

## What the preparation script does

1. **Validates dataset structure**: Checks that both GoPro and Spike datasets are properly extracted
2. **Organizes GoPro data**: Copies frames from `blur/sharp` folders into organized `train_GT` and `train_GT_blurred` directories
3. **Verifies alignment**: Ensures GoPro and Spike datasets have matching sequences
4. **Generates meta info**: Creates text files listing all sequences and frame counts
5. **Generates LMDB** (optional): Creates LMDB databases for faster data loading during training

## Script Options

```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py --help
```

### Key Options:

- `--gopro_root`: Path to GOPRO_Large dataset (default: `/media/mallm/hd4t/modelrepostore/datasets/GOPRO_Large`)
- `--spike_root`: Path to Spike dataset (default: `/media/mallm/hd4t/modelrepostore/datasets/gopro_spike/GOPRO_Large_spike_seq`)
- `--splits`: Which splits to process (default: `train test`)
- `--generate_lmdb`: Generate LMDB files for faster loading
- `--force`: Force regeneration even if output exists
- `--skip_validation`: Skip dataset validation checks

## Troubleshooting

### Issue: "GoPro validation failed"

Make sure you've extracted the GOPRO_Large dataset and it contains `train/` and `test/` directories with subdirectories containing `blur/` and `sharp/` folders.

### Issue: "Spike validation failed"

Make sure you've extracted the gopro_spike dataset and it contains `train/` and `test/` directories with subdirectories containing `spike/` folders with `.dat` files.

### Issue: "No common sequences found"

The GoPro and Spike datasets must have matching sequence names. Check that both datasets are properly extracted and correspond to each other.

### Issue: meta_info file is empty

Existing `train_GT/`, `train_GT_blurred/`, `test_GT/`, or `test_GT_blurred/` directories might be empty if a previous run was interrupted. Remove the empty directories or force regeneration:
```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py --splits train test --force
```
The script will log `Warning: No frames found ...` for any skipped sequence and omit it from the meta info file.

### Issue: LMDB generation fails

Make sure you have enough disk space and the LMDB Python package is installed:

```bash
pip install lmdb
```

## Dataset Information

### GoPro Large Dataset

- **Training**: 22 sequences, ~2,103 frames
- **Test**: 11 sequences, ~1,111 frames
- **Resolution**: 720×1280
- **Format**: PNG

### Spike Dataset

- **Training**: 22 sequences (matching GoPro)
- **Test**: 11 sequences (matching GoPro)
- **Resolution**: 250×400 (spike camera)
- **Format**: Binary .dat files
- **Channels**: 1 (binary spike events)

## Next Steps

After data preparation:

1. Verify the organized data structure exists
2. Check that meta info files are generated in `data/meta_info/`
3. Update training config if using custom paths
4. Start training:
   ```bash
   bash launch_train.sh 1
   ```

## Advanced Usage

### Preparing only test data

```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py --splits test
```

### Custom paths with LMDB

```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py \
    --gopro_root /custom/path/GOPRO_Large \
    --spike_root /custom/path/gopro_spike \
    --generate_lmdb
```

### Force regeneration

If you've modified the raw data or want to regenerate everything:

```bash
python scripts/data_preparation/prepare_gopro_spike_dataset.py --force --generate_lmdb
```

## Related Files

- `create_lmdb_for_vrt.py`: Low-level LMDB generation (called by prepare script)
- `../../launch_train.sh`: Integrated training launcher with data preparation
- `../../options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike_local.json`: Training configuration


## SCFlow strict encoding25 preparation

Use the dedicated script to generate strict 25-slice flow artifacts next to each clip:

```bash
python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root /path/to/GOPRO_Large_spike_seq/train \
  --dt 10
```

Optional flags:

- `--meta-info-file data/meta_info/meta_info_GoPro_train_GT.txt` to use explicit clip start frame mapping.
- `--dry-run` to report what would be generated.
- `--overwrite` to regenerate existing `.npy` files.
