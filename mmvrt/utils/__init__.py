from .file_client import FileClient, imfrombytes
from .io import scandir
from .spike_loader import SpikeStreamSimple, voxelize_spikes_tfp
import importlib


def import_from_string(path: str):
    """Import a class or callable from a full import path string."""
    module_path, _, attr = path.rpartition('.')
    if not module_path:
        raise ImportError(f"Invalid import path: {path}")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f"Module '{module_path}' has no attribute '{attr}'") from e

__all__ = ['FileClient', 'imfrombytes', 'scandir', 'SpikeStreamSimple', 'voxelize_spikes_tfp']

"""General utilities (task-agnostic, pure functions/thin wrappers)."""

