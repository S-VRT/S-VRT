from typing import Any, Protocol


class FusionOperator(Protocol):
    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        ...


def validate_mode(mode: str) -> str:
    if mode not in ('replace', 'residual'):
        raise ValueError(f"Unknown fusion placement: {mode}")
    return mode


__all__ = ['FusionOperator', 'validate_mode']
