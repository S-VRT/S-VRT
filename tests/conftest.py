#!/usr/bin/env python3
"""
Pytest configuration and fixtures for VRT project tests.
"""
import json
import os
import sys
from pathlib import Path

import pytest
import torch

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_vrt_config():
    """Standard VRT configuration for testing."""
    return {
        "upscale": 1,
        "in_chans": 11,
        "img_size": [6, 160, 160],
        "window_size": [6, 8, 8],
        "depths": [8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
        "indep_reconsts": [9, 10],
        "embed_dims": [96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
        "num_heads": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        "pa_frames": 2,
        "deformable_groups": 16,
        "nonblind_denoising": False,
        "use_checkpoint_attn": True,
        "use_checkpoint_ffn": True,
        "no_checkpoint_attn_blocks": [2, 3, 4],
        "no_checkpoint_ffn_blocks": [1, 2, 3, 4, 5, 9],
    }


@pytest.fixture
def sample_batch(device):
    """Sample batch data for testing."""
    batch_size, frames, channels, height, width = 1, 6, 11, 160, 160
    return torch.randn(batch_size, frames, channels, height, width, device=device)


def _strip_line_comments(json_text):
    """Strip // comments while preserving // inside JSON strings."""
    out = []
    in_string = False
    escaped = False
    i = 0
    n = len(json_text)

    while i < n:
        ch = json_text[i]

        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < n and json_text[i + 1] == "/":
            while i < n and json_text[i] != "\n":
                i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _load_json_with_line_comments(path_obj):
    raw_text = path_obj.read_text(encoding="utf-8")
    return json.loads(_strip_line_comments(raw_text))


@pytest.fixture(scope="session")
def server_option_path():
    """Path to server-side option file used by e2e tests."""
    return Path(project_root) / "options" / "gopro_rgbspike_server.json"


@pytest.fixture(scope="session")
def server_option(server_option_path):
    """Load server option JSON with support for line comments."""
    if not server_option_path.exists():
        pytest.skip(f"server option not found: {server_option_path}")
    return _load_json_with_line_comments(server_option_path)


def _dataset_paths_from_opt(opt_dict):
    train = opt_dict.get("datasets", {}).get("train", {})
    test = opt_dict.get("datasets", {}).get("test", {})
    return {
        "train_gt": train.get("dataroot_gt"),
        "train_lq": train.get("dataroot_lq"),
        "train_spike": train.get("dataroot_spike"),
        "test_gt": test.get("dataroot_gt"),
        "test_lq": test.get("dataroot_lq"),
        "test_spike": test.get("dataroot_spike"),
    }


def require_paths_or_skip(paths, reason_prefix="dataset path missing"):
    missing = [p for p in paths if not p or not Path(p).exists()]
    if missing:
        pytest.skip(f"{reason_prefix}: {missing}")


@pytest.fixture(scope="session")
def require_paths_or_skip_fn():
    """Expose path-check helper to tests without importing conftest directly."""
    return require_paths_or_skip
