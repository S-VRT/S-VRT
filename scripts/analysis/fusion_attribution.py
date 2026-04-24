from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.fusion_attr.io import load_samples_file, strip_json_comments, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline S-VRT fusion attribution toolkit")
    parser.add_argument("--opt", required=True, help="Path to S-VRT option JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--samples", required=True, help="Path to fusion_samples.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--baseline-opt", default=None, help="Optional baseline option JSON")
    parser.add_argument("--baseline-checkpoint", default=None, help="Optional baseline checkpoint")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--cam-method", default="gradcam", choices=["gradcam", "hirescam", "fallback"])
    parser.add_argument("--target", default="masked_charbonnier", choices=["masked_charbonnier"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--center-frame-only", action="store_true")
    parser.add_argument("--save-raw", action="store_true", default=True)
    parser.add_argument("--save-panel", action="store_true", default=True)
    parser.add_argument("--perturb-spike", default="zero", choices=["zero", "shuffle", "noise", "temporal-drop"])
    parser.add_argument("--mask-source", default="manual", choices=["manual", "motion", "error-topk"])
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write manifest without loading model")
    return parser


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_run_manifest(args: argparse.Namespace, samples_count: int) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    opt_text = _read_text(args.opt)
    (out_root / "config_snapshot.json").write_text(strip_json_comments(opt_text), encoding="utf-8")
    write_json(
        out_root / "run_manifest.json",
        {
            "opt": args.opt,
            "checkpoint": args.checkpoint,
            "samples": args.samples,
            "num_samples": samples_count,
            "baseline_opt": args.baseline_opt,
            "baseline_checkpoint": args.baseline_checkpoint,
            "device": args.device,
            "cam_method": args.cam_method,
            "target": args.target,
            "perturb_spike": args.perturb_spike,
            "mask_source": args.mask_source,
            "dry_run": bool(args.dry_run),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    samples = load_samples_file(args.samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    write_run_manifest(args, len(samples))
    if args.dry_run:
        print("Fusion attribution dry run complete.")
        return 0
    raise RuntimeError("Model-backed attribution execution is added in the next task.")


if __name__ == "__main__":
    raise SystemExit(main())
