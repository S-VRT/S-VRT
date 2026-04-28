import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from scripts.analysis.fusion_params import (
    FusionParameterCountError,
    build_arg_parser,
    count_fusion_parameters,
    format_parameter_count,
)


class _DummyNet(nn.Module):
    def __init__(self, fusion_enabled=True, fusion_adapter=None):
        super().__init__()
        self.fusion_enabled = fusion_enabled
        if fusion_adapter is not None:
            self.fusion_adapter = fusion_adapter


def test_count_fusion_parameters_counts_complete_adapter_tree_once():
    shared = nn.Parameter(torch.ones(5))
    fusion_adapter = nn.Module()
    fusion_adapter.early = nn.Linear(3, 4)
    fusion_adapter.aux = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3), nn.ReLU())
    fusion_adapter.register_parameter("shared_a", shared)
    fusion_adapter.register_parameter("shared_b", shared)
    fusion_adapter.frozen = nn.Linear(2, 2)
    for param in fusion_adapter.frozen.parameters():
        param.requires_grad_(False)
    net = _DummyNet(fusion_adapter=fusion_adapter)

    expected = (
        fusion_adapter.early.weight.numel()
        + fusion_adapter.early.bias.numel()
        + fusion_adapter.aux[0].weight.numel()
        + fusion_adapter.aux[0].bias.numel()
        + shared.numel()
        + fusion_adapter.frozen.weight.numel()
        + fusion_adapter.frozen.bias.numel()
    )

    assert count_fusion_parameters(net) == expected


def test_count_fusion_parameters_rejects_disabled_fusion():
    net = _DummyNet(fusion_enabled=False, fusion_adapter=nn.Linear(1, 1))

    with pytest.raises(FusionParameterCountError, match="Fusion is not enabled"):
        count_fusion_parameters(net)


def test_count_fusion_parameters_rejects_missing_adapter():
    net = _DummyNet()

    with pytest.raises(FusionParameterCountError, match="fusion_adapter"):
        count_fusion_parameters(net)


def test_count_fusion_parameters_rejects_unavailable_optional_submodules():
    fusion_adapter = nn.Module()
    fusion_adapter.mamba_block = nn.Module()
    fusion_adapter.mamba_block.mamba = None
    net = _DummyNet(fusion_adapter=fusion_adapter)

    with pytest.raises(FusionParameterCountError, match="optional fusion submodule"):
        count_fusion_parameters(net)


def test_format_parameter_count_uses_integer_and_million_scale():
    assert format_parameter_count(1_234_567) == "1,234,567 (1.235 M)"


def test_arg_parser_accepts_option_path():
    args = build_arg_parser().parse_args(["options/gopro_rgbspike_server.json"])

    assert args.opt == "options/gopro_rgbspike_server.json"


def test_cli_direct_script_execution_can_import_project_modules():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/fusion_params.py",
            "options/does-not-exist.json",
        ],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "No module named 'models'" not in result.stderr
