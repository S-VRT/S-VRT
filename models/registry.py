from typing import Callable, Dict, Any

MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str):
    """Decorator to register a model builder/class under a name."""
    def _register(cls_or_fn: Callable[..., Any]):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return _register


def get_model(name: str):
    """Retrieve a registered model builder/class by name."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"Model '{name}' is not registered.") from e


def list_models():
    return list(MODEL_REGISTRY.keys())


__all__ = ['register_model', 'get_model', 'list_models', 'MODEL_REGISTRY']


