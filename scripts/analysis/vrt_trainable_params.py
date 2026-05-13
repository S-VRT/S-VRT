from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class ParameterSummary:
    label: str
    total: int
    trainable: int
    lora_total: int
    lora_trainable: int
    fusion_trainable: int
    vrt_backbone_trainable: int
    trainable_tensors: int
    lora_modules: int


def _numel(parameters: Iterable[nn.Parameter]) -> int:
    return sum(param.numel() for param in parameters)


def _is_lora_parameter(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def _is_fusion_parameter(name: str) -> bool:
    return "fusion_adapter" in name or "fusion_operator" in name


def _count_lora_modules(net: nn.Module) -> int:
    return sum(1 for module in net.modules() if module.__class__.__name__ == "LoRALinear")


def summarize_parameters(label: str, net: nn.Module) -> ParameterSummary:
    named_params = list(net.named_parameters())
    trainable_named_params = [(name, param) for name, param in named_params if param.requires_grad]
    lora_named_params = [(name, param) for name, param in named_params if _is_lora_parameter(name)]
    trainable_lora = [(name, param) for name, param in trainable_named_params if _is_lora_parameter(name)]
    trainable_fusion = [(name, param) for name, param in trainable_named_params if _is_fusion_parameter(name)]

    return ParameterSummary(
        label=label,
        total=_numel(param for _, param in named_params),
        trainable=_numel(param for _, param in trainable_named_params),
        lora_total=_numel(param for _, param in lora_named_params),
        lora_trainable=_numel(param for _, param in trainable_lora),
        fusion_trainable=_numel(param for _, param in trainable_fusion),
        vrt_backbone_trainable=_numel(
            param
            for name, param in trainable_named_params
            if not _is_lora_parameter(name) and not _is_fusion_parameter(name)
        ),
        trainable_tensors=len(trainable_named_params),
        lora_modules=_count_lora_modules(net),
    )


def build_model(
    opt_path: str,
    *,
    use_lora: bool,
    freeze_backbone_enabled: bool,
    phase2_lora_mode: bool,
    emulate_phase2: bool = False,
) -> nn.Module:
    from models.lora import inject_lora
    from models.model_plain import freeze_backbone, freeze_known_unused_parameters
    from models.select_network import define_G
    from utils import utils_option as option

    opt = option.parse(opt_path, is_train=True)
    opt["rank"] = 0
    train_opt = opt.setdefault("train", {})
    train_opt["use_lora"] = use_lora
    train_opt["freeze_backbone"] = freeze_backbone_enabled
    train_opt["phase2_lora_mode"] = phase2_lora_mode

    net = define_G(opt)

    if use_lora:
        inject_lora(
            net,
            train_opt.get("lora_target_modules", ["qkv", "proj"]),
            rank=int(train_opt.get("lora_rank", 8)),
            alpha=float(train_opt.get("lora_alpha", 16)),
        )

    if freeze_backbone_enabled:
        freeze_backbone(net)
        if use_lora and phase2_lora_mode:
            for name, param in net.named_parameters():
                if _is_lora_parameter(name):
                    param.requires_grad_(False)
        freeze_known_unused_parameters(net)

    if emulate_phase2:
        fix_keys = train_opt.get("fix_keys", [])
        for name, param in net.named_parameters():
            if any(key in name for key in fix_keys):
                param.requires_grad_(True)
            if use_lora and _is_lora_parameter(name):
                param.requires_grad_(True)
        freeze_known_unused_parameters(net)

    return net


def format_count(value: int) -> str:
    return f"{value:,} ({value / 1_000_000:.3f} M)"


def print_summary(summary: ParameterSummary) -> None:
    print(summary.label)
    print(f"  total: {format_count(summary.total)}")
    print(f"  trainable: {format_count(summary.trainable)}")
    print(f"  vrt_backbone_trainable_excluding_lora_and_fusion: {format_count(summary.vrt_backbone_trainable)}")
    print(f"  lora_total: {format_count(summary.lora_total)}")
    print(f"  lora_trainable: {format_count(summary.lora_trainable)}")
    print(f"  fusion_trainable: {format_count(summary.fusion_trainable)}")
    print(f"  trainable_tensors: {summary.trainable_tensors}")
    print(f"  lora_modules: {summary.lora_modules}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report VRT trainable parameter counts for non-LoRA and LoRA training modes."
    )
    parser.add_argument("opt", help="Path to an S-VRT option JSON file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenarios = [
        (
            "non_lora_full_train",
            dict(use_lora=False, freeze_backbone_enabled=False, phase2_lora_mode=False),
        ),
        (
            "non_lora_with_config_freeze_backbone",
            dict(use_lora=False, freeze_backbone_enabled=True, phase2_lora_mode=False),
        ),
        (
            "lora_phase1_inserted_but_frozen_until_fix_iter",
            dict(use_lora=True, freeze_backbone_enabled=True, phase2_lora_mode=True),
        ),
        (
            "lora_phase2_fix_keys_plus_lora",
            dict(
                use_lora=True,
                freeze_backbone_enabled=True,
                phase2_lora_mode=True,
                emulate_phase2=True,
            ),
        ),
        (
            "lora_stage_c_lora_trainable_immediately",
            dict(use_lora=True, freeze_backbone_enabled=True, phase2_lora_mode=False),
        ),
    ]

    for index, (label, kwargs) in enumerate(scenarios):
        if index:
            print()
        net = build_model(args.opt, **kwargs)
        print_summary(summarize_parameters(label, net))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
