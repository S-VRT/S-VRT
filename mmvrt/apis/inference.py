"""Simple high-level inference helper for mmvrt.

This provides a minimal entry used by users who want a single-call inference
function that builds model from config and runs prediction on a video clip.
It's intentionally small — for production use callers should implement
their own wrappers.
"""
from pathlib import Path
from typing import Any, Dict, Optional, List
import sys

try:
    from mmengine.config import Config
    from mmengine.runner import Runner
    _USE_MMENGINE = True
except Exception:
    _USE_MMENGINE = False

from mmvrt.registry import MODELS


def inference_video(cfg_path: str, checkpoint: Optional[str] = None, device: str = 'cpu') -> List[Dict[str, Any]]:
    """Load config, build runner/model and run inference on dataset defined in cfg.

    Returns a list of prediction DataSamples or dicts.
    """
    if _USE_MMENGINE:
        cfg = Config.fromfile(cfg_path)
        runner = Runner.from_cfg(cfg)
        if checkpoint:
            runner.load_checkpoint(checkpoint)
        # Run test/infer via runner; assume cfg.test_dataloader and evaluator configured
        results = runner.test()
        return results
    else:
        raise RuntimeError("MMEngine is required for inference_video helper")




