# Tests for VRT+Spike Video Deblurring

This directory contains comprehensive tests for the VRT+Spike video deblurring project, organized following SOTA project standards (PyTorch, MMDetection, BasicSR).

## Directory Structure

```
tests/
├── conftest.py                              # Pytest configuration and fixtures
├── __init__.py                              # Package initialization
│
├── unit/                                    # Unit tests (individual components)
│   ├── data/                               # Data loading & processing
│   │   └── test_spike_voxel_realtime.py   # Real-time voxel generation
│   ├── models/                             # Model architectures
│   └── losses/                             # Loss functions
│
├── integration/                             # Integration tests (workflows)
│   ├── pipeline/                           # Data pipeline tests
│   │   └── test_training_dataloader.py    # Complete training pipeline
│   └── training/                           # Training workflow tests
│       └── test_system_readiness.py       # System readiness check
│
└── benchmark/                               # Performance benchmarks
    └── bench_data_loading.py               # Data loading performance
```

## Test Categories

### 🔬 Unit Tests (`unit/`)

Tests for individual components in isolation.

**Data Tests** (`unit/data/`)
- `test_spike_voxel_realtime.py`: Tests real-time spike voxel generation
  - Validates voxel conversion from spike streams
  - Checks value ranges and dimensions
  - Tests multiple samples

**Model Tests** (`unit/models/`)
- TODO: Add model architecture tests

**Loss Tests** (`unit/losses/`)
- TODO: Add loss function tests

### 🔗 Integration Tests (`integration/`)

Tests for complete workflows and component interactions.

**Pipeline Tests** (`integration/pipeline/`)
- `test_training_dataloader.py`: Tests complete training data pipeline
  - Dataset creation
  - Multi-worker data loading
  - Batch collation
  - Data augmentation (cropping)
  - Shape and value validation

**Training Tests** (`integration/training/`)
- `test_system_readiness.py`: Comprehensive system readiness check
  - Package imports
  - Data availability
  - Configuration validation
  - Dataset loading
  - Model imports

### ⚡ Benchmarks (`benchmark/`)

Performance benchmarks for optimization.

- `bench_data_loading.py`: Data loading performance benchmark
  - Real-time voxel generation speed
  - Throughput measurement
  - Latency analysis

## Running Tests

### Run All Tests
```bash
cd /home/mallm/henry/Deblur
pytest tests/ -v
```

### Run Specific Test Categories

**Unit Tests**
```bash
pytest tests/unit/ -v
```

**Integration Tests**
```bash
pytest tests/integration/ -v
```

**Benchmarks**
```bash
pytest tests/benchmark/ -v
```

### Run Specific Test Files

**System Readiness Check** (recommended before training)
```bash
python tests/integration/training/test_system_readiness.py
```

**Real-time Voxel Generation**
```bash
python tests/unit/data/test_spike_voxel_realtime.py
```

**Training Pipeline**
```bash
python tests/integration/pipeline/test_training_dataloader.py
```

**Data Loading Benchmark**
```bash
python tests/benchmark/bench_data_loading.py
```

## Test Naming Conventions

Following pytest and SOTA project conventions:

- **Test files**: `test_*.py` or `*_test.py`
- **Test functions**: `test_*`
- **Test classes**: `Test*`
- **Benchmarks**: `bench_*.py`

## Fixtures

Common fixtures are defined in `conftest.py`:

- `repo_root`: Project root directory
- `config_path`: Path to default config file
- `data_root`: Path to processed data
- `default_dataset_config`: Default dataset configuration
- `default_loader_config`: Default data loader configuration

## Testing Workflow

### Before Training

1. **System Readiness Check**
   ```bash
   python tests/integration/training/test_system_readiness.py
   ```
   Validates all prerequisites are met.

2. **Data Pipeline Test**
   ```bash
   python tests/integration/pipeline/test_training_dataloader.py
   ```
   Ensures data loading works correctly.

### During Development

1. **Unit Tests** - Test individual components
2. **Integration Tests** - Test component interactions
3. **Benchmarks** - Measure and optimize performance

### CI/CD Integration

For continuous integration:

```bash
# Quick tests (unit + integration)
pytest tests/unit/ tests/integration/ -v --tb=short

# Full test suite (including benchmarks)
pytest tests/ -v --tb=short

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

## Performance Benchmarks

### Data Loading Benchmark

Measures throughput and latency:

```bash
python tests/benchmark/bench_data_loading.py
```

**Expected Results:**
- Real-time voxel generation: ~50-100 ms/batch (4 samples)
- Throughput: ~40-80 samples/sec with 4 workers

## Adding New Tests

### Unit Test Template

```python
#!/usr/bin/env python3
"""Test description."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

def test_feature():
    """Test a specific feature."""
    # Arrange
    # Act
    # Assert
    pass

if __name__ == "__main__":
    # Allow running as standalone script
    test_feature()
```

### Integration Test Template

```python
#!/usr/bin/env python3
"""Integration test description."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

def test_workflow():
    """Test a complete workflow."""
    print("\n" + "="*70)
    print("Integration Test: Workflow Name")
    print("="*70)
    
    # Test steps with clear output
    # Return True/False or raise assertions
    
if __name__ == "__main__":
    success = test_workflow()
    sys.exit(0 if success else 1)
```

## Best Practices

1. **Clear Test Names**: Use descriptive names that explain what is being tested
2. **Standalone Scripts**: Tests should be runnable as standalone Python scripts
3. **Clear Output**: Print informative messages during test execution
4. **Proper Cleanup**: Clean up any temporary files or resources
5. **Fast Unit Tests**: Unit tests should run quickly (< 1 second each)
6. **Isolated Tests**: Tests should not depend on each other
7. **Fixtures**: Use fixtures for common setup/teardown

## Troubleshooting

### Test Fails: Module Not Found

Ensure project root is in Python path:
```python
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
```

### Test Fails: Data Not Found

Check data is prepared:
```bash
python scripts/prepare_gopro_spike_structure.py \
    --gopro_root /path/to/GOPRO_Large \
    --spike_gopro_root /path/to/Spike-GOPRO \
    --output_root ./data/processed/gopro_spike_aligned
```

### Test Fails: VRT Import Error

VRT is added to path in `conftest.py`, but ensure it's available:
```bash
ls -la third_party/VRT/
```

## Contributing

When adding new features:

1. Add corresponding unit tests
2. Add integration tests if the feature interacts with other components
3. Update this README if adding new test categories
4. Follow existing naming and structure conventions

## References

This test structure follows conventions from:
- [PyTorch](https://github.com/pytorch/pytorch/tree/master/test)
- [MMDetection](https://github.com/open-mmlab/mmdetection/tree/master/tests)
- [BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master/tests)
- [Detectron2](https://github.com/facebookresearch/detectron2/tree/main/tests)

