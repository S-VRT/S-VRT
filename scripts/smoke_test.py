#!/usr/bin/env python
"""
Lightweight smoke tests:
- Config parsing
- Dataset/Dataloader one-step
- Model forward (no backward)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config.parser import load_config
from core.builder import build_dataset, build_dataloader, build_model
import data.datasets  # noqa: F401
import models.vrt  # noqa: F401
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--check-forward", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    print("Config OK.")

    if args.check_data or args.check_forward:
        train_cfg = cfg.get("data", {}).get("train", {})
        ds = build_dataset(train_cfg, split="train")
        dl = build_dataloader(ds, train_cfg.get("dataloader", {}), distributed=False)
        batch = next(iter(dl))
        print("Dataloader OK: keys", list(batch.keys()))

    if args.check_forward:
        model = build_model(cfg.get("model", {}))
        model.eval()
        if "lq" not in batch and "L" in batch:
            batch["lq"] = batch["L"]
        with torch.no_grad():
            out = model(batch)
        if isinstance(out, dict):
            print("Forward OK, keys:", list(out.keys()))
        else:
            print("Forward OK, tensor output")


if __name__ == "__main__":
    main()

