from typing import Callable, Dict, Any

MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str):
    """Decorator to register a model builder/class under a name.

    Re-registering the same name is allowed when the module is reloaded
    (e.g. during testing with importlib.import_module after sys.modules.pop).
    Registering a *different* callable under an already-registered name raises.
    """
    def _register(cls_or_fn: Callable[..., Any]):
        existing = MODEL_REGISTRY.get(name)
        if existing is not None and existing.__qualname__ != cls_or_fn.__qualname__:
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


