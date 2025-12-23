from __future__ import annotations

"""
New S-VRT training entry (src-only).

This script replaces the legacy main_train_vrt.py and is intended
to be the only public training entry going forward.
"""

import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent
try:
    # Prefer package imports when installed as `mmvrt`
    from mmvrt.config.parser import build_argparser, load_config  # type: ignore
    from mmvrt.engine.runner import Runner  # type: ignore
    from mmvrt.tools.prepare_data import prepare  # type: ignore
    from mmvrt.core.runtime import set_cuda_visible_devices  # type: ignore
except Exception:
    # Fallback to repository `src/` layout for development/checkouts.
    sys.path.insert(0, str(ROOT / "src"))
    from config.parser import build_argparser, load_config  # noqa: E402
    from engine.runner import Runner  # noqa: E402
    from tools.prepare_data import prepare  # noqa: E402
    from core.runtime import set_cuda_visible_devices  # noqa: E402
# If the new mmvrt Runner isn't available (partial refactor), fall back to legacy training entry.
try:
    if "Runner" not in globals() or Runner is None:  # type: ignore[name-defined]
        raise ImportError
except Exception:
    try:
        import main_train_vrt as legacy_train  # type: ignore
    except Exception:
        legacy_train = None


def build_train_parser() -> argparse.ArgumentParser:
    parser = build_argparser()
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids, sets CUDA_VISIBLE_DEVICES (single-node only).",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Run data preparation before training using src/tools/prepare_data.py.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="Path to checkpoint created by Runner.save_checkpoint to resume from.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (AMP) during training.",
    )
    return parser


def main():
    parser = build_train_parser()
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)

    # ------------------------------------------------------------------ #
    # GPU selection semantics
    # ------------------------------------------------------------------ #
    # Prefer explicit CLI ``--gpus`` (for quick experiments). When it is
    # not provided, fall back to the configuration's ``runtime.gpu_ids``
    # so that the new YAML entrypoint mirrors the legacy JSON behaviour
    # where GPU visibility was driven from the config file.
    if args.gpus:
        set_cuda_visible_devices(args.gpus)
    else:
        runtime_cfg = cfg.get("runtime", {}) or {}
        gpu_ids = runtime_cfg.get("gpu_ids")
        if isinstance(gpu_ids, (list, tuple)) and gpu_ids:
            gpus_str = ",".join(str(g) for g in gpu_ids)
            set_cuda_visible_devices(gpus_str)

    # Wire AMP/resume flags into cfg so Runner can see them if needed.
    runtime_cfg = cfg.setdefault("runtime", {})
    runtime_cfg.setdefault("amp", bool(args.amp))
    if args.resume:
        runtime_cfg["resume"] = args.resume

    if args.prepare_data:
        prepare(cfg)

    workdir = Path(cfg.get("experiment", {}).get("workdir", "./runs/default"))
    # If a modern Runner is available, use it. Otherwise fall back to legacy main.
    if "Runner" in globals() and Runner is not None:  # type: ignore[name-defined]
        runner = Runner(cfg, workdir)
        runner.build()

        # Basic resume: load model/optimizer/scheduler/EMA if checkpoint provided.
        ckpt_path = runtime_cfg.get("resume")
        if ckpt_path:
            ckpt_path = Path(ckpt_path)
            if ckpt_path.is_file():
                import torch

                state = torch.load(ckpt_path, map_location="cpu")
                if "model" in state and runner.model is not None:
                    runner.model.load_state_dict(state["model"], strict=True)
                if "optimizer" in state and runner.optimizer is not None and state["optimizer"] is not None:
                    runner.optimizer.load_state_dict(state["optimizer"])
                if "scheduler" in state and runner.lr_scheduler is not None and state["scheduler"] is not None:
                    runner.lr_scheduler.load_state_dict(state["scheduler"])
                if "model_ema" in state and runner.ema is not None:
                    runner.ema.model.load_state_dict(state["model_ema"], strict=False)

        runner.train()
    else:
        # Legacy fallback: delegate to the legacy monolithic training script which
        # parses its own CLI. This keeps the entrypoint runnable during refactor.
        if legacy_train is not None:
            legacy_train.main()
        else:
            raise RuntimeError("No compatible Runner found and legacy training entry unavailable.")



if __name__ == "__main__":
    main()


