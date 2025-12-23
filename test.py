#!/usr/bin/env python
"""Validation/Test entry."""

import sys
from pathlib import Path
import argparse
import os

ROOT = Path(__file__).resolve().parent
try:
    from mmvrt.config.parser import build_argparser, load_config  # type: ignore
    from mmvrt.engine.runner import Runner  # type: ignore
    from mmvrt.tools.prepare_data import prepare  # type: ignore
    from mmvrt.core.runtime import set_cuda_visible_devices, apply_tile_overrides  # type: ignore
except Exception:
    sys.path.insert(0, str(ROOT / "src"))
    from config.parser import build_argparser, load_config  # noqa: E402
    from engine.runner import Runner  # noqa: E402
    from tools.prepare_data import prepare  # noqa: E402
    from core.runtime import set_cuda_visible_devices, apply_tile_overrides  # noqa: E402
# If modern Runner not available, prepare a legacy fallback.
try:
    if "Runner" not in globals() or Runner is None:  # type: ignore[name-defined]
        raise ImportError
except Exception:
    try:
        import main_test_vrt as legacy_test  # type: ignore
    except Exception:
        legacy_test = None


def main():
    parser = build_argparser()
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids, sets CUDA_VISIBLE_DEVICES")
    parser.add_argument("--prepare-data", action="store_true", help="Run data preparation before test")
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

    workdir = Path(cfg.get("experiment", {}).get("workdir", "./runs/default"))
    # Use MMEngine-style Runner if available, otherwise fall back to legacy test.
    if "Runner" in globals() and Runner is not None:  # type: ignore[name-defined]
        runner = Runner(cfg, workdir)
        runner.build()
        runner.validate(epoch=0)
    else:
        if legacy_test is not None:
            # The legacy test script parses its own args; hand off control to it.
            legacy_test.main()
        else:
            raise RuntimeError("No compatible Runner found and legacy test entry unavailable.")


if __name__ == "__main__":
    main()


