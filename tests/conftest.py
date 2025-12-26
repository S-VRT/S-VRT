#!/usr/bin/env python3
"""
Pytest configuration and fixtures for VRT project tests.
"""
import pytest
import torch
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_vrt_config():
    """Standard VRT configuration for testing."""
    return {
        'upscale': 1,
        'in_chans': 11,
        'img_size': [6, 160, 160],
        'window_size': [6, 8, 8],
        'depths': [8,8,8,8,8,8,8, 4,4, 4,4],
        'indep_reconsts': [9,10],
        'embed_dims': [96,96,96,96,96,96,96, 120,120, 120,120],
        'num_heads': [6,6,6,6,6,6,6, 6,6, 6,6],
        'pa_frames': 2,
        'deformable_groups': 16,
        'nonblind_denoising': False,
        'use_checkpoint_attn': True,
        'use_checkpoint_ffn': True,
        'no_checkpoint_attn_blocks': [2,3,4],
        'no_checkpoint_ffn_blocks': [1,2,3,4,5,9]
    }


@pytest.fixture
def sample_batch(device):
    """Sample batch data for testing."""
    batch_size, frames, channels, height, width = 1, 6, 11, 160, 160
    return torch.randn(batch_size, frames, channels, height, width, device=device)
