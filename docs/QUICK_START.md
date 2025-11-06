# Quick Start Guide

## For First-Time Setup

If you just extracted the raw datasets:

```bash
# All-in-one: Prepare data + generate LMDB + start training
bash launch_train.sh 1 --prepare-data --generate-lmdb
```

## For Subsequent Runs

If data is already prepared:

```bash
# Just start training
bash launch_train.sh 1
```

## Check Dataset Status

```bash
python scripts/data_preparation/check_dataset_status.py
```

## Multi-GPU Training

```bash
# 4 GPUs
bash launch_train.sh 4

# 8 GPUs with data preparation
bash launch_train.sh 8 --prepare-data --generate-lmdb
```

## Need Help?

```bash
bash launch_train.sh --help
```

## Documentation

- **Full Guide**: `DATA_PREPARATION_GUIDE.md`
- **Summary**: `DATASET_SETUP_SUMMARY.md`
- **Technical Details**: `scripts/data_preparation/README.md`

---

**That's it!** The simplest command to get started is:

```bash
bash launch_train.sh 1 --prepare-data --generate-lmdb
```

This will:
1. ✓ Check your dataset structure
2. ✓ Organize GoPro frames (GT + LQ)
3. ✓ Verify Spike data alignment
4. ✓ Generate meta info files
5. ✓ Create LMDB databases
6. ✓ Start training automatically

