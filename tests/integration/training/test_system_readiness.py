#!/usr/bin/env python3
"""
Integration test for complete system readiness.

Validates all components needed for training:
- Python packages
- Data availability
- Configuration files
- Dataset loading
- Model imports
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_package_imports():
    """Test if all required packages are installed."""
    print("\n" + "="*70)
    print("1. Testing Package Imports")
    print("="*70)
    
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
    ]
    
    all_ok = True
    for pkg, name in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def test_data_availability():
    """Test if data is properly set up."""
    print("\n" + "="*70)
    print("2. Testing Data Availability")
    print("="*70)
    
    data_root = REPO_ROOT / "data/processed/gopro_spike_unified"
    
    if not data_root.exists():
        print(f"  ✗ Data root not found: {data_root}")
        return False
    
    print(f"  ✓ Data root exists: {data_root}")
    
    # Check train split
    train_dir = data_root / "train"
    if not train_dir.exists():
        print(f"  ✗ Train directory not found")
        return False
    
    sequences = list(train_dir.iterdir())
    print(f"  ✓ Train sequences: {len(sequences)}")
    
    # Check first sequence structure
    if len(sequences) > 0:
        seq = sequences[0]
        has_blur = (seq / "blur").exists()
        has_sharp = (seq / "sharp").exists()
        has_spike = (seq / "spike").exists()
        
        print(f"  ✓ Example sequence: {seq.name}")
        print(f"    - blur/: {'✓' if has_blur else '✗'}")
        print(f"    - sharp/: {'✓' if has_sharp else '✗'}")
        print(f"    - spike/: {'✓' if has_spike else '✗'}")
        
        return has_blur and has_sharp and has_spike
    
    return True


def test_configuration():
    """Test if config file is properly set up."""
    print("\n" + "="*70)
    print("3. Testing Configuration")
    print("="*70)
    
    config_path = REPO_ROOT / "configs/deblur/vrt_spike_baseline.yaml"
    
    if not config_path.exists():
        print(f"  ✗ Config not found: {config_path}")
        return False
    
    print(f"  ✓ Config file exists")
    
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        print(f"  ✓ Config is valid YAML")
        print(f"  ✓ Data root: {cfg['DATA']['ROOT']}")
        print(f"  ✓ Batch size: {cfg['TRAIN']['BATCH_SIZE']}")
        print(f"  ✓ Clip length: {cfg['DATA']['CLIP_LEN']}")
        print(f"  ✓ Real-time mode: {not cfg['DATA']['USE_PRECOMPUTED_VOXELS']}")
        print(f"  ✓ Voxel bins: {cfg['DATA']['NUM_VOXEL_BINS']}")
        
        return True
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False


def test_dataset_loading():
    """Test if dataset can be loaded."""
    print("\n" + "="*70)
    print("4. Testing Dataset Loading")
    print("="*70)
    
    try:
        import yaml
        from src.data.datasets.spike_deblur_dataset import SpikeDeblurDataset
        
        config_path = REPO_ROOT / "configs/deblur/vrt_spike_baseline.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        data_root = REPO_ROOT / "data/processed/gopro_spike_unified"
        
        dataset = SpikeDeblurDataset(
            root=str(data_root),
            split="train",
            clip_length=5,
            crop_size=256,
            spike_dir="spike",
            num_voxel_bins=5,
            use_precomputed_voxels=False,
        )
        
        print(f"  ✓ Dataset created")
        print(f"  ✓ Total samples: {len(dataset)}")
        
        # Test loading one sample
        sample = dataset[0]
        print(f"  ✓ Sample loaded successfully")
        print(f"    - blur: {sample['blur'].shape}")
        print(f"    - sharp: {sample['sharp'].shape}")
        print(f"    - voxel: {sample['spike_vox'].shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Dataset error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_imports():
    """Test if model can be imported."""
    print("\n" + "="*70)
    print("5. Testing Model Imports")
    print("="*70)
    
    try:
        import torch
        
        # Add VRT to path
        VRT_ROOT = REPO_ROOT / "third_party" / "VRT"
        if str(VRT_ROOT) not in sys.path:
            sys.path.insert(0, str(VRT_ROOT))
        
        from models.network_vrt import VRT
        from src.models.integrate_vrt import VRTWithSpike
        
        print(f"  ✓ Model imports successful")
        print(f"  ✓ VRT module available")
        print(f"  ✓ VRTWithSpike integration available")
        
        print(f"\n  Note: Forward pass test skipped (requires specific input sizes)")
        print(f"        Model will be validated during training")
        
        return True
    except Exception as e:
        print(f"  ✗ Model import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all system readiness tests."""
    print("\n" + "="*70)
    print("System Readiness Test for VRT+Spike Training")
    print("="*70)
    
    tests = [
        ("Package Imports", test_package_imports),
        ("Data Availability", test_data_availability),
        ("Configuration", test_configuration),
        ("Dataset Loading", test_dataset_loading),
        ("Model Imports", test_model_imports),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✓ ALL TESTS PASSED - SYSTEM READY!")
        print("\nYou can start training:")
        print("  python src/train.py --config configs/deblur/vrt_spike_baseline.yaml")
        print("="*70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before training.")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

