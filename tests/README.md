# VRT/S-VRT Test Suite

This directory contains the test suite for the VRT (Video Restoration Transformer) and S-VRT (Spike-enhanced VRT) projects.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── models/                  # Model-specific tests
│   ├── test_vrt_integration.py    # VRT integration tests
│   └── ...                       # Other model tests
├── smoke/                   # Quick smoke tests
└── README.md               # This file
```

## Running Tests

### Using the test runner script

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --models          # Model tests
python tests/run_tests.py --smoke           # Smoke tests
python tests/run_tests.py --integration     # Integration tests

# Run specific test files
python tests/run_tests.py tests/models/test_vrt_integration.py

# Run with coverage
python tests/run_tests.py --coverage

# Run without output capture (for debugging)
python tests/run_tests.py --no-capture
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/models/test_vrt_integration.py
pytest tests/models/test_vrt_integration.py::TestVRTIntegration::test_vrt_forward_pass

# Run with coverage
pytest --cov=. --cov-report=html
```

## Test Categories

### Model Tests (`tests/models/`)
- **VRT Integration Tests**: Test VRT model instantiation, forward passes, and optical flow integration
- **Optical Flow Tests**: Test SeaRAFT and SpyNet optical flow modules
- **Compatibility Tests**: Ensure model calling conventions match between training and testing

### Smoke Tests (`tests/smoke/`)
- Quick sanity checks for basic functionality
- Fast-running tests suitable for CI/CD pipelines

## Key Test Cases

### VRT Integration Tests
1. **Import Test**: Verify VRT can be imported successfully
2. **Instantiation Test**: Test VRT can be created with standard configurations
3. **Forward Pass Test**: Test VRT forward pass with dummy data
4. **Optical Flow Integration**: Test SpyNet integration with VRT's reshape logic
5. **Model Calling Convention**: Verify correct model calling patterns

### Known Issues
- SeaRAFT integration is currently skipped due to format mismatch with VRT's expected multi-scale flow format
- TODO: Align SeaRAFT output format with SpyNet's format for VRT compatibility

## Adding New Tests

1. Create test files in appropriate directories (`tests/models/`, `tests/smoke/`, etc.)
2. Use pytest fixtures from `conftest.py` for common setup
3. Follow naming convention: `test_*.py` for test files, `test_*` for test functions
4. Add appropriate pytest markers for categorization

## CI/CD Integration

Tests can be run in CI/CD pipelines using:
```bash
python tests/run_tests.py --smoke  # Fast smoke tests for PR validation
python tests/run_tests.py          # Full test suite for nightly builds
```
