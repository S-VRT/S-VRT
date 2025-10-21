#!/usr/bin/env python3
"""
Integration test for the complete training data pipeline.

Tests the full data loading pipeline including:
- Dataset creation
- DataLoader with multiple workers
- Batch collation
- Data augmentation (cropping)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader
from src.data.datasets.spike_deblur_dataset import SpikeDeblurDataset
from src.data.collate_fns import safe_spike_deblur_collate


def test_training_dataloader():
    """Test the complete training data loading pipeline."""
    
    print("\n" + "="*70)
    print("Integration Test: Training Data Pipeline")
    print("="*70)
    
    # Load config
    config_path = REPO_ROOT / "configs/deblur/vrt_spike_baseline.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    data_root = REPO_ROOT / "data/processed/gopro_spike_unified"
    
    print(f"\n1. Creating training dataset...")
    try:
        train_set = SpikeDeblurDataset(
            root=str(data_root),
            split="train",
            clip_length=5,
            crop_size=256,  # Training uses random crops
            spike_dir="spike",
            num_voxel_bins=5,
            use_precomputed_voxels=False,
        )
        print(f"   ✓ Training dataset created: {len(train_set)} samples")
    except Exception as e:
        print(f"   ✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n2. Creating data loader with multi-processing...")
    try:
        train_loader = DataLoader(
            train_set,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=safe_spike_deblur_collate,
            drop_last=True,
        )
        print(f"   ✓ Data loader created")
        print(f"   - Batch size: 4")
        print(f"   - Num workers: 4")
        print(f"   - Expected batches: {len(train_loader)}")
    except Exception as e:
        print(f"   ✗ Data loader creation failed: {e}")
        return False
    
    print(f"\n3. Testing batch loading...")
    num_test_batches = min(3, len(train_loader))
    
    for i, batch in enumerate(train_loader):
        if i >= num_test_batches:
            break
        
        print(f"\n   Batch {i+1}:")
        print(f"   - blur shape: {batch['blur'].shape}")
        print(f"   - sharp shape: {batch['sharp'].shape}")
        print(f"   - spike_vox shape: {batch['spike_vox'].shape}")
        print(f"   - sequences: {batch['meta']['seq'][:2]}...")  # Show first 2
        
        try:
            # Validate batch shapes
            B, T, C, H, W = batch['blur'].shape
            assert T == 5, f"Expected clip_length=5, got {T}"
            assert C == 3, f"Expected RGB channels=3, got {C}"
            assert H == 256, f"Expected crop height=256, got {H}"
            assert W == 256, f"Expected crop width=256, got {W}"
            assert B <= 4, f"Batch size should be <= 4, got {B}"
            
            # Validate spike voxel dimensions
            B_s, T_s, bins, H_s, W_s = batch['spike_vox'].shape
            assert B_s == B, "Spike batch size mismatch"
            assert T_s == T, "Spike temporal length mismatch"
            assert bins == 5, f"Expected 5 voxel bins, got {bins}"
            
            # Validate value ranges
            assert batch['blur'].min() >= 0 and batch['blur'].max() <= 1, \
                "Blur values out of [0, 1] range"
            assert batch['sharp'].min() >= 0 and batch['sharp'].max() <= 1, \
                "Sharp values out of [0, 1] range"
            
            # Check voxel statistics
            vox_nonzero = (batch['spike_vox'] > 0).sum().item()
            vox_total = batch['spike_vox'].numel()
            vox_percent = 100 * vox_nonzero / vox_total
            
            print(f"   - Voxel resolution: {H_s}×{W_s}")
            print(f"   - Voxel sparsity: {vox_percent:.2f}% non-zero")
            print(f"   ✓ Batch {i+1} validated successfully")
            
        except AssertionError as e:
            print(f"   ✗ Validation failed: {e}")
            return False
        except Exception as e:
            print(f"   ✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*70}")
    print(f"✓ All integration tests passed!")
    print(f"{'='*70}")
    print(f"\nPipeline Summary:")
    print(f"  - Total samples: {len(train_set)}")
    print(f"  - Batches per epoch: {len(train_loader)}")
    print(f"  - Voxel generation: Real-time (no precomputation)")
    print(f"  - Multi-processing: 4 workers")
    print(f"  - Data augmentation: Random 256×256 crops")
    
    print(f"\n{'='*70}")
    print(f"Ready to train with:")
    print(f"  python src/train.py --config configs/deblur/vrt_spike_baseline.yaml")
    print(f"{'='*70}")
    
    return True


def main():
    """Run the test."""
    try:
        success = test_training_dataloader()
        return 0 if success else 1
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

