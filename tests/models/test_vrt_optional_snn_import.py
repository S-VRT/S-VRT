import importlib
import sys


def test_vrt_package_imports_without_optional_snntorch():
    # Ensure package import does not hard-require optional SNN dependency.
    # Clear cached modules AND registry entry so re-import is clean.
    from models.registry import MODEL_REGISTRY
    MODEL_REGISTRY.pop("VRT", None)
    for key in list(sys.modules):
        if key == "models.architectures.vrt" or key.startswith("models.architectures.vrt."):
            sys.modules.pop(key, None)
    module = importlib.import_module("models.architectures.vrt")
    assert hasattr(module, "VRT")
