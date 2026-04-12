import importlib
import sys


def test_vrt_package_imports_without_optional_snntorch():
    # Ensure package import does not hard-require optional SNN dependency.
    sys.modules.pop("models.architectures.vrt", None)
    module = importlib.import_module("models.architectures.vrt")
    assert hasattr(module, "VRT")
