from typing import Dict, Any

from .registry import get_model


def build_model(cfg: Dict[str, Any]):
    """Build a model from a configuration dictionary.

    Expected config formats:
      - {'name': 'VRT', 'params': {...}}
      - {'model': {'name': 'VRT', 'params': {...}}}
    """
    if cfg is None:
        raise ValueError("cfg must be a dict containing model name and params")

    # support nested 'model' key
    if isinstance(cfg, dict) and 'model' in cfg:
        cfg = cfg['model']

    if isinstance(cfg, dict):
        name = cfg.get('name')
        params = cfg.get('params', {})
    else:
        name = getattr(cfg, 'name', None)
        params = getattr(cfg, 'params', {})

    if name is None:
        raise ValueError("No model name provided in cfg")

    builder = get_model(name)
    if builder is None:
        raise KeyError(f"Model {name} is not registered.")

    return builder(**params)


__all__ = ['build_model']


