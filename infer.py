#!/usr/bin/env python
"""Inference entry with optional tiling."""

import sys
from pathlib import Path
import argparse
import os

ROOT = Path(__file__).resolve().parent
try:
    from mmvrt.config.parser import build_argparser, load_config  # type: ignore
    from mmvrt.inference.predictor import Predictor  # type: ignore
    from mmvrt.tools.prepare_data import prepare  # type: ignore
    from mmvrt.core.runtime import set_cuda_visible_devices, apply_tile_overrides  # type: ignore
except Exception:
    sys.path.insert(0, str(ROOT / "src"))
    from config.parser import build_argparser, load_config  # noqa: E402
    from inference.predictor import Predictor  # noqa: E402
    from tools.prepare_data import prepare  # noqa: E402
    from core.runtime import set_cuda_visible_devices, apply_tile_overrides  # noqa: E402
# Legacy fallback: if Predictor is not available, use legacy test entrypoint.
try:
    if "Predictor" not in globals() or Predictor is None:  # type: ignore[name-defined]
        raise ImportError
except Exception:
    try:
        import main_test_vrt as legacy_test  # type: ignore
    except Exception:
        legacy_test = None


def main():
    parser = build_argparser()
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids, sets CUDA_VISIBLE_DEVICES")
    parser.add_argument("--prepare-data", action="store_true", help="Run data preparation before infer")
    parser.add_argument("--tile", type=int, nargs=3, default=None, metavar=("T", "H", "W"), help="Tile size for testing")
    parser.add_argument("--tile-overlap", type=int, nargs=3, default=None, metavar=("T", "H", "W"), help="Tile overlap")
    args = parser.parse_args()

    if args.gpus:
        set_cuda_visible_devices(args.gpus)

    cfg = load_config(args.config, args.override)

    if args.tile or args.tile_overlap:
        cfg = apply_tile_overrides(cfg, args.tile, args.tile_overlap)

    if args.prepare_data:
        prepare(cfg)

    # Prefer the new Predictor if present; otherwise hand off to legacy test.
    if "Predictor" in globals() and Predictor is not None:  # type: ignore[name-defined]
        predictor = Predictor(cfg)
        predictor.run()
    else:
        if legacy_test is not None:
            legacy_test.main()
        else:
            raise RuntimeError("No compatible Predictor found and legacy inference entry unavailable.")


if __name__ == "__main__":
    main()


