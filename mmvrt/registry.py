"""Central registry definitions for mmvrt.

We prefer MMEngine registries with an explicit default scope so that configs
can use ``type=...`` without prefixing ``mmvrt.``.  When MMEngine is not
available we fall back to a lightweight registry with a compatible API so
that unit tests and minimal environments still work.
"""

from typing import Any, Callable, Dict, Optional

try:  # pragma: no cover - executed only when mmengine is installed
    from mmengine.registry import DefaultScope, Registry

    # Ensure default scope is established before any registration happens.
    DefaultScope.get_instance('mmvrt', scope_name='mmvrt')

    MODELS = Registry('model', default_scope='mmvrt')
    DATASETS = Registry('dataset', default_scope='mmvrt')
    TRANSFORMS = Registry('transform', default_scope='mmvrt')
    METRICS = Registry('metric', default_scope='mmvrt')
    HOOKS = Registry('hook', default_scope='mmvrt')

    _USE_MMENGINE = True
except Exception:
    class Registry:
        """Simple registry compatible with MMEngine-style API (fallback)."""

        def __init__(self, name: str):
            self._name = name
            self._store: Dict[str, Callable[..., Any]] = {}

        def register_module(self, name: Optional[str] = None, force: bool = False):
            """Decorator to register a callable under an optional name."""

            def _decorator(func: Callable[..., Any]):
                key = name or func.__name__
                if key in self._store and not force:
                    raise KeyError(f"{key} already registered in {self._name}")
                self._store[key] = func
                return func

            return _decorator

        # alias to keep parity with MMEngine's ``@MODELS.register_module()``
        register = register_module

        def get(self, name: str) -> Callable[..., Any]:
            if name not in self._store:
                raise KeyError(f"{name} is not registered in {self._name}")
            return self._store[name]

        def build(self, cfg: Dict[str, Any], *args, **kwargs) -> Any:
            if not isinstance(cfg, dict):
                raise TypeError(f"cfg must be a dict, got {type(cfg)}")
            if 'type' not in cfg:
                raise KeyError(f"cfg must contain 'type' key, got {cfg}")
            obj_type = self.get(cfg['type'])
            cfg = {k: v for k, v in cfg.items() if k != 'type'}
            return obj_type(*args, **kwargs, **cfg)

        def __contains__(self, name: str) -> bool:
            return name in self._store

        def __len__(self) -> int:
            return len(self._store)

    MODELS = Registry("models")
    DATASETS = Registry("datasets")
    TRANSFORMS = Registry("transforms")
    METRICS = Registry("metrics")
    HOOKS = Registry("hooks")

    _USE_MMENGINE = False

__all__ = ['MODELS', 'DATASETS', 'TRANSFORMS', 'METRICS', 'HOOKS', '_USE_MMENGINE']

