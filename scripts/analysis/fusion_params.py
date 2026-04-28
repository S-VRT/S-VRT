from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FusionParameterCountError(RuntimeError):
    """Raised when a model has no enabled fusion module to count."""


def _unique_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    seen: set[int] = set()
    for param in module.parameters():
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        yield param


def _ensure_optional_fusion_submodules_available(module: nn.Module) -> None:
    for name, submodule in module.named_modules():
        if "mamba" in submodule.__dict__ and getattr(submodule, "mamba") is None:
            display_name = name or submodule.__class__.__name__
            raise FusionParameterCountError(
                "Cannot count fusion parameters because optional fusion submodule "
                f"{display_name!r} is unavailable. Install its runtime dependency first."
            )


def count_fusion_parameters(net: nn.Module) -> int:
    if not bool(getattr(net, "fusion_enabled", False)):
        raise FusionParameterCountError("Fusion is not enabled on this model.")

    fusion_module = getattr(net, "fusion_adapter", None)
    if fusion_module is None:
        raise FusionParameterCountError("Model has no fusion_adapter to count.")
    if not isinstance(fusion_module, nn.Module):
        raise FusionParameterCountError(
            f"fusion_adapter must be an nn.Module, got {type(fusion_module).__name__}."
        )

    _ensure_optional_fusion_submodules_available(fusion_module)
    return sum(param.numel() for param in _unique_parameters(fusion_module))


def format_parameter_count(count: int) -> str:
    return f"{count:,} ({count / 1_000_000:.3f} M)"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate the complete parameter count of the configured fusion module."
    )
    parser.add_argument("opt", help="Path to an S-VRT option JSON file.")
    return parser


def build_model_from_option(opt_path: str) -> nn.Module:
    from models.select_network import define_G
    from utils import utils_option as option

    opt = option.parse(opt_path, is_train=False)
    return define_G(opt)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        net = build_model_from_option(args.opt)
        count = count_fusion_parameters(net)
    except FusionParameterCountError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Fusion parameters: {format_parameter_count(count)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
