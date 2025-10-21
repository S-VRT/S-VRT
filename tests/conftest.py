"""
Pytest configuration and shared fixtures for tests.
"""
import sys
from pathlib import Path
import pytest

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Add VRT to path
VRT_ROOT = REPO_ROOT / "third_party" / "VRT"
if VRT_ROOT.exists() and str(VRT_ROOT) not in sys.path:
    sys.path.insert(0, str(VRT_ROOT))


@pytest.fixture(scope="session")
def repo_root():
    """Project repository root directory."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def config_path(repo_root):
    """Path to default configuration file."""
    return repo_root / "configs" / "deblur" / "vrt_spike_baseline.yaml"


@pytest.fixture(scope="session")
def data_root(repo_root):
    """Path to processed data directory."""
    return repo_root / "data" / "processed" / "gopro_spike_unified"


@pytest.fixture
def default_dataset_config():
    """Default configuration for dataset creation."""
    return {
        "split": "train",
        "clip_length": 5,
        "crop_size": 256,
        "spike_dir": "spike",
        "num_voxel_bins": 5,
        "use_precomputed_voxels": False,
    }


@pytest.fixture
def default_loader_config():
    """Default configuration for data loader."""
    return {
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": True,
    }

