from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from scripts.analysis.fusion_attr.io import (
    build_sample_output_dir,
    load_samples_file,
    save_gray_map_png,
    save_rgb_tensor_png,
    strip_json_comments,
    write_json,
)
from scripts.analysis.fusion_attr.maps import (
    compute_error_map,
    compute_fusion_delta,
    gradient_activation_cam,
    integrated_gradients_map,
    normalize_map,
)
from scripts.analysis.fusion_attr.pca import pca_feature_heatmap, pca_variance_ratio
from scripts.analysis.fusion_attr.panels import make_six_column_panel
from scripts.analysis.fusion_attr.probes import FusionProbe, find_fusion_adapter, reduce_operator_explanations
from scripts.analysis.fusion_attr.targets import build_box_mask, masked_charbonnier_target


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
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-panel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--perturb-spike", default="zero", choices=["zero", "shuffle", "noise", "temporal-drop"])
    parser.add_argument("--mask-source", default="manual", choices=["manual", "motion", "error-topk"])
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write manifest without loading model")
    parser.add_argument("--save-ig", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--save-pca", action=argparse.BooleanOptionalAction, default=False)
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


def select_center_frame_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 5:
        return tensor[:, tensor.shape[1] // 2]
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Expected 4D or 5D tensor, got {tuple(tensor.shape)}")


def _load_json_config(path: str) -> dict:
    return json.loads(strip_json_comments(Path(path).read_text(encoding="utf-8")))


def _prepare_eval_opt(opt: dict) -> dict:
    cfg = dict(opt)
    cfg["is_train"] = False
    cfg["dist"] = False
    cfg["rank"] = 0
    cfg.setdefault("path", {})
    return cfg


def _load_checkpoint_if_available(model, checkpoint: str) -> None:
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    bare = model.get_bare_model(model.netG) if hasattr(model, "get_bare_model") else model.netG
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "params" in state:
        state = state["params"]
    result = bare.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"[checkpoint] missing keys: {result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
    if result.unexpected_keys:
        print(f"[checkpoint] unexpected keys: {result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")


def _build_test_loader(opt: dict) -> DataLoader:
    datasets = opt.get("datasets", {})
    test_opt = dict(datasets.get("test") or datasets.get("val") or {})
    if not test_opt:
        raise ValueError("Option file must contain datasets.test or datasets.val for attribution")
    test_opt["phase"] = "test"
    dataset = define_Dataset(test_opt)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def _sample_matches(batch: dict, sample_clip: str) -> bool:
    folder = batch.get("folder")
    if isinstance(folder, (list, tuple)):
        folder = folder[0]
    if isinstance(folder, str) and sample_clip in folder:
        return True
    lq_path = batch.get("L_path") or batch.get("lq_path")
    if isinstance(lq_path, (list, tuple)) and lq_path:
        return sample_clip in str(lq_path[0])
    return False


def _find_batch_for_sample(loader: DataLoader, sample) -> dict:
    for batch in loader:
        if _sample_matches(batch, sample.clip):
            return batch
    raise ValueError(f"Could not find sample clip in dataset: {sample.clip}")


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
    opt = _prepare_eval_opt(_load_json_config(args.opt))
    model = define_Model(opt)
    _load_checkpoint_if_available(model, args.checkpoint)
    model.netG.eval()
    loader = _build_test_loader(opt)

    for sample in samples:
        batch = _find_batch_for_sample(loader, sample)
        lq = batch["L"].to(next(model.netG.parameters()).device)
        gt = batch["H"].to(lq.device)
        adapter = find_fusion_adapter(model.netG)
        probe = FusionProbe(adapter)
        probe.attach()
        model.netG.zero_grad(set_to_none=True)
        output = model.netG(lq)
        if isinstance(output, (tuple, list)):
            output = output[0]
        probe.close()
        if probe.record is None:
            raise RuntimeError("Fusion probe did not capture a forward pass")

        center_output = select_center_frame_tensor(output)
        center_gt = select_center_frame_tensor(gt)
        mask = build_box_mask(sample, center_output.shape[-2], center_output.shape[-1], center_output.device)
        activation = probe.record.output
        if activation.requires_grad:
            activation.retain_grad()
        target = masked_charbonnier_target(center_output, center_gt, mask)
        cam = gradient_activation_cam(activation, target)

        sample_dir = build_sample_output_dir(args.out, sample)
        inputs_dir = sample_dir / "inputs"
        outputs_dir = sample_dir / "outputs"
        maps_dir = sample_dir / "maps"
        save_rgb_tensor_png(inputs_dir / "blurry_rgb.png", select_center_frame_tensor(lq[:, :, :3]))
        save_rgb_tensor_png(inputs_dir / "spike_cue.png", select_center_frame_tensor(lq[:, :, 3:]))
        save_rgb_tensor_png(inputs_dir / "gt.png", center_gt)
        save_rgb_tensor_png(outputs_dir / "restored.png", center_output)
        error_full = compute_error_map(center_output, center_gt)
        save_gray_map_png(maps_dir / "error_full.png", normalize_map(error_full))
        save_gray_map_png(maps_dir / "cam.png", normalize_map(cam))
        np.save(str(maps_dir / "cam_raw.npy"), cam.detach().cpu().numpy())

        if probe.record.inputs:
            reference = probe.record.inputs[0]
            if reference.shape == probe.record.output.shape:
                fusion_delta = compute_fusion_delta(probe.record.output, reference)
                np.save(str(maps_dir / "fusion_delta.npy"), fusion_delta.cpu().numpy())
                save_gray_map_png(maps_dir / "fusion_delta.png", normalize_map(fusion_delta))

        operator = getattr(adapter, "operator", None)
        fusion_specific_path = maps_dir / "cam.png"
        if operator is not None and hasattr(operator, "explain"):
            reduced = reduce_operator_explanations(operator.explain())
            for name, value in reduced.items():
                np.save(str(maps_dir / f"{name}.npy"), value.detach().cpu().numpy())
                save_gray_map_png(maps_dir / f"{name}.png", normalize_map(value))
            if "effective_update" in reduced:
                fusion_specific_path = maps_dir / "effective_update.png"

        metadata: dict = {
            "sample_id": f"{sample.clip}_{sample.frame}",
            "frame_index": sample.frame_index,
            "mask_type": sample.mask_type,
            "mask_xyxy": list(sample.xyxy),
            "mask_label": sample.mask_label,
            "target": args.target,
            "cam_method": args.cam_method,
            "checkpoint": args.checkpoint,
            "opt": args.opt,
            "probe_module": probe.record.module_name,
        }

        if args.save_ig:
            rgb_input = select_center_frame_tensor(lq[:, :, :3]).detach()
            spike_input = select_center_frame_tensor(lq[:, :, 3:]).detach()

            def rgb_model(rgb_only):
                fused = torch.cat([rgb_only, spike_input], dim=1)
                out = model.netG(fused.unsqueeze(1))
                return select_center_frame_tensor(out if not isinstance(out, (tuple, list)) else out[0])

            def spike_model(spike_only):
                fused = torch.cat([rgb_input, spike_only], dim=1)
                out = model.netG(fused.unsqueeze(1))
                return select_center_frame_tensor(out if not isinstance(out, (tuple, list)) else out[0])

            def target_fn(restored):
                return masked_charbonnier_target(restored, center_gt, mask)

            ig_rgb = integrated_gradients_map(
                rgb_model,
                rgb_input,
                torch.zeros_like(rgb_input),
                target_fn,
                steps=args.ig_steps,
            )
            ig_spike = integrated_gradients_map(
                spike_model,
                spike_input,
                torch.zeros_like(spike_input),
                target_fn,
                steps=args.ig_steps,
            )
            save_gray_map_png(maps_dir / "ig_rgb.png", normalize_map(ig_rgb))
            save_gray_map_png(maps_dir / "ig_spike.png", normalize_map(ig_spike))

        if args.save_pca and operator is not None and hasattr(operator, "explain"):
            raw = operator.explain()
            for name in ("effective_update", "delta", "token_energy"):
                if name in raw:
                    pca_map = pca_feature_heatmap(raw[name])
                    np.save(str(maps_dir / f"{name}_pca.npy"), pca_map.detach().cpu().numpy())
                    save_gray_map_png(maps_dir / f"{name}_pca.png", normalize_map(pca_map))
                    variance = pca_variance_ratio(raw[name])[:3].detach().cpu().tolist()
                    metadata.setdefault("pca_variance_ratio", {})[name] = variance

        write_json(
            sample_dir / "metadata.json",
            metadata,
        )

        panel_images = {
            "Blurry RGB": cv2.imread(str(inputs_dir / "blurry_rgb.png"), cv2.IMREAD_COLOR),
            "Spike cue": cv2.imread(str(inputs_dir / "spike_cue.png"), cv2.IMREAD_COLOR),
            "Restored": cv2.imread(str(outputs_dir / "restored.png"), cv2.IMREAD_COLOR),
            "Error reduction": cv2.imread(str(maps_dir / "error_full.png"), cv2.IMREAD_COLOR),
            "Attribution heatmap": cv2.imread(str(maps_dir / "cam.png"), cv2.IMREAD_COLOR),
            "Fusion-specific map": cv2.imread(str(fusion_specific_path), cv2.IMREAD_COLOR),
        }
        make_six_column_panel(sample_dir / "panel.png", panel_images)
    print(f"Fusion attribution complete: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
