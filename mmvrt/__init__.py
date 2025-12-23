"""MMVRT: Video Restoration Transformer library following OpenMMLab conventions.

This package follows MMDet architecture principles:
- Registry system with default_scope='mmvrt'
- Component-based model design (Restorer + Backbone + Head + Loss)
- Dataset + Pipeline + DataPreprocessor separation
- MMEngine Runner.from_cfg for training/testing
"""

from .registry import MODELS, DATASETS, TRANSFORMS, METRICS, HOOKS
from .version import __version__

# Import key components for convenience
from .models.restorers import BaseRestorer, VRTDeblurRestorer
from .models.backbones import VRTBackbone
from .models.data_preprocessors import RGBSpikeDataPreprocessor
from .models.losses import CharbonnierLoss, SSIMLoss
from .structures import RestorationDataSample
from .evaluation.metrics import PSNR, SSIM

__all__ = [
    'MODELS', 'DATASETS', 'TRANSFORMS', 'METRICS', 'HOOKS', '__version__',
    'BaseRestorer', 'VRTDeblurRestorer', 'VRTBackbone',
    'RGBSpikeDataPreprocessor', 'RestorationDataSample',
    'CharbonnierLoss', 'SSIMLoss',
    'PSNR', 'SSIM',
]

