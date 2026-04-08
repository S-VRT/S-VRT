from typing import Any, Protocol


class FusionOperator(Protocol):
    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        ...


def validate_mode(mode: str) -> str:
    normalized = str(mode).lower().strip()
    if normalized not in ('replace', 'residual'):
        raise ValueError(f"Unsupported fusion mode: {normalized}")
    return normalized


__all__ = ['FusionOperator', 'validate_mode']
