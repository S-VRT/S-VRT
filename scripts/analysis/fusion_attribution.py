from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
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
from models.model_plain import ModelPlain
from models.select_model import define_Model
from scripts.analysis.fusion_attr.io import (
    build_sample_output_dir,
    load_samples_file,
    save_gray_map_png,
    save_rgb_tensor_png,
    strip_json_comments,
    write_json,
)
from scripts.analysis.fusion_attr.cam import build_cam_metadata, build_cam_scope_targets, compute_cam_map, select_cam_target
from scripts.analysis.fusion_attr.maps import (
    compute_error_map,
    compute_fusion_delta,
    integrated_gradients_map,
    normalize_map,
)
from scripts.analysis.fusion_attr.pca import pca_feature_heatmap, pca_variance_ratio
from scripts.analysis.fusion_attr.panels import make_six_column_panel
from scripts.analysis.fusion_attr.probes import FusionProbe, find_fusion_adapter, reduce_operator_explanations
from scripts.analysis.fusion_attr.stitching import TileBox, crop_box_to_tile, mask_intersects_tile, stitch_weighted_tiles


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
    parser.add_argument("--analysis-num-frames", type=int, default=12)
    parser.add_argument("--analysis-crop-size", type=int, default=256)
    parser.add_argument("--analysis-tile-stride", type=int, default=None)
    parser.add_argument("--cam-scopes", nargs="+", default=["fullframe", "roi"], choices=["fullframe", "roi"])
    parser.add_argument("--stitch-weight", default="hann", choices=["hann"])
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


def _inject_lora_for_checkpoint_if_needed(model, opt: dict, checkpoint: str) -> None:
    train_opt = opt.get("train", {}) if isinstance(opt.get("train", {}), dict) else {}
    if not train_opt.get("use_lora", False):
        return
    if not ModelPlain._checkpoint_contains_lora(checkpoint):
        return
    if not hasattr(model, "_inject_lora_adapters"):
        raise RuntimeError("LoRA checkpoint detected, but model does not support LoRA injection.")
    bare = model.get_bare_model(model.netG) if hasattr(model, "get_bare_model") else model.netG
    model._inject_lora_adapters(train_opt, bare)
    model.netG = bare


def _build_test_loader(opt: dict) -> DataLoader:
    datasets = opt.get("datasets", {})
    test_opt = dict(datasets.get("test") or datasets.get("val") or {})
    if not test_opt:
        raise ValueError("Option file must contain datasets.test or datasets.val for attribution")
    test_opt["phase"] = "test"
    dataset = define_Dataset(test_opt)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def _build_test_dataset(opt: dict):
    datasets = opt.get("datasets", {})
    test_opt = dict(datasets.get("test") or datasets.get("val") or {})
    if not test_opt:
        raise ValueError("Option file must contain datasets.test or datasets.val for attribution")
    test_opt["phase"] = "test"
    return define_Dataset(test_opt)


def _build_folder_index(dataset) -> dict[str, int]:
    folders = getattr(dataset, "folders", None)
    if folders is None:
        raise ValueError("Dataset does not expose folders; cannot index samples directly.")
    return {str(folder): idx for idx, folder in enumerate(folders)}


def _ensure_batched_sample(batch: dict) -> dict:
    out = dict(batch)
    for key in ("L", "H", "L_flow_spike"):
        value = out.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            out[key] = value.unsqueeze(0)
    return out


def _get_batch_for_sample(dataset, folder_index: dict[str, int], sample) -> dict:
    if sample.clip not in folder_index:
        raise ValueError(f"Could not find sample clip in dataset: {sample.clip}")
    return _ensure_batched_sample(dataset[folder_index[sample.clip]])


def _as_first_string(value) -> str | None:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return None


def _path_matches(path: str | None, sample_clip: str, sample_frame: str | None = None) -> bool:
    if path is None or sample_clip not in path:
        return False
    if sample_frame is None:
        return True
    return sample_frame in path


def _sample_matches(batch: dict, sample_clip: str, sample_frame: str | None = None) -> bool:
    folder = _as_first_string(batch.get("folder"))
    if _path_matches(folder, sample_clip):
        return True
    lq_path = _as_first_string(batch.get("L_path") or batch.get("lq_path"))
    if _path_matches(lq_path, sample_clip, sample_frame):
        return True
    key = _as_first_string(batch.get("key"))
    if _path_matches(key, sample_clip, sample_frame):
        return True
    return False


def _find_batch_for_sample(loader: DataLoader, sample) -> dict:
    for batch in loader:
        if _sample_matches(batch, sample.clip, sample.frame):
            return batch
    raise ValueError(f"Could not find sample clip in dataset: {sample.clip}")


def _flatten_path_strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_path_strings(item))
        return result
    return [str(value)]


def _find_frame_index(batch: dict, sample) -> int:
    for idx, path in enumerate(_flatten_path_strings(batch.get("lq_path") or batch.get("L_path"))):
        if sample.frame in path:
            return idx
    return max(int(sample.frame_index) - 1, 0)


def _temporal_window_start(tensor: torch.Tensor, center_idx: int, num_frames: int) -> int:
    if tensor.ndim != 5 or num_frames <= 0 or tensor.shape[1] <= num_frames:
        return 0
    start = max(0, min(center_idx - num_frames // 2, tensor.shape[1] - num_frames))
    return start


def _slice_temporal_window(tensor: torch.Tensor, start: int, num_frames: int) -> torch.Tensor:
    if tensor.ndim != 5 or num_frames <= 0 or tensor.shape[1] <= num_frames:
        return tensor
    return tensor[:, start : start + num_frames]


def _crop_spatial_patch(tensor: torch.Tensor, xyxy: tuple[int, int, int, int], crop_size: int) -> tuple[torch.Tensor, int, int]:
    if tensor.ndim != 5 or crop_size <= 0:
        return tensor, 0, 0
    height, width = tensor.shape[-2:]
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    x1, y1, x2, y2 = xyxy
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    left = max(0, min(center_x - crop_w // 2, width - crop_w))
    top = max(0, min(center_y - crop_h // 2, height - crop_h))
    return tensor[..., top : top + crop_h, left : left + crop_w], left, top


def _prepare_analysis_batch(batch: dict, sample, num_frames: int, crop_size: int) -> tuple[dict, object]:
    center_idx = _find_frame_index(batch, sample)
    temporal_start = _temporal_window_start(batch["L"], center_idx, num_frames)
    prepared = dict(batch)
    for key in ("L", "H"):
        if key in prepared and isinstance(prepared[key], torch.Tensor):
            prepared[key] = _slice_temporal_window(prepared[key], temporal_start, num_frames)

    left = top = 0
    for key in ("L", "H"):
        if key in prepared and isinstance(prepared[key], torch.Tensor):
            prepared[key], left, top = _crop_spatial_patch(prepared[key], sample.xyxy, crop_size)

    x1, y1, x2, y2 = sample.xyxy
    height, width = prepared["L"].shape[-2:]
    shifted_xyxy = (
        max(0, min(width, x1 - left)),
        max(0, min(height, y1 - top)),
        max(0, min(width, x2 - left)),
        max(0, min(height, y2 - top)),
    )
    prepared["_analysis_temporal_start"] = temporal_start
    prepared["_analysis_crop_left"] = left
    prepared["_analysis_crop_top"] = top
    prepared["_analysis_full_h"] = batch["L"].shape[-2]
    prepared["_analysis_full_w"] = batch["L"].shape[-1]
    return prepared, replace(sample, xyxy=shifted_xyxy)


def _prepare_tile_batch(
    batch: dict,
    temporal_start: int,
    num_frames: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> dict:
    prepared = dict(batch)
    for key in ("L", "H"):
        if key in prepared and isinstance(prepared[key], torch.Tensor):
            prepared[key] = prepared[key][:, temporal_start : temporal_start + num_frames, :, top:bottom, left:right]
    prepared["_analysis_temporal_start"] = temporal_start
    prepared["_analysis_crop_left"] = left
    prepared["_analysis_crop_top"] = top
    prepared["_analysis_full_h"] = batch["L"].shape[-2]
    prepared["_analysis_full_w"] = batch["L"].shape[-1]
    return prepared


def _build_flow_spike_for_batch(model, batch: dict, lq: torch.Tensor) -> torch.Tensor | None:
    if "L_flow_spike" in batch and isinstance(batch["L_flow_spike"], torch.Tensor):
        flow = batch["L_flow_spike"]
        start = int(batch.get("_analysis_temporal_start", 0))
        if flow.ndim == 5 and flow.shape[1] > lq.shape[1]:
            flow = flow[:, start : start + lq.shape[1]]
        return flow.to(lq.device)

    raw_meta = batch.get("L_flow_spike_meta")
    if raw_meta is None or not hasattr(model, "_normalize_flow_spike_meta") or not hasattr(model, "_load_lazy_flow_patch"):
        return None
    meta = model._normalize_flow_spike_meta(raw_meta)
    return model._load_lazy_flow_patch(
        meta,
        int(batch.get("_analysis_temporal_start", 0)),
        lq.shape[1],
        int(batch.get("_analysis_crop_top", 0)),
        int(batch.get("_analysis_crop_left", 0)),
        lq.shape[-2],
        lq.shape[-1],
        int(batch.get("_analysis_full_h", lq.shape[-2])),
        int(batch.get("_analysis_full_w", lq.shape[-1])),
    ).to(lq.device)


def _spatial_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if tile_size >= length:
        return [0]
    starts = list(range(0, max(length - tile_size, 0), stride))
    starts.append(length - tile_size)
    return sorted(set(starts))


def _spatial_tiles(height: int, width: int, tile_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    tiles = []
    for top in _spatial_starts(height, tile_size, stride):
        for left in _spatial_starts(width, tile_size, stride):
            tiles.append((top, left, min(top + tile_size, height), min(left + tile_size, width)))
    return tiles


def resolve_tile_stride(tile_size: int, requested_stride: int | None) -> int:
    return max(1, tile_size // 2) if requested_stride is None else int(requested_stride)


def resolve_cam_default_scope(scopes_exported: list[str]) -> str:
    if "fullframe" in scopes_exported:
        return "fullframe"
    if scopes_exported:
        return scopes_exported[0]
    return "fullframe"


def _accumulate_tile(accum: torch.Tensor, weight: torch.Tensor, tile: torch.Tensor, top: int, left: int) -> None:
    bottom = top + tile.shape[-2]
    right = left + tile.shape[-1]
    accum[..., top:bottom, left:right] += tile.detach().to(accum.device)
    weight[..., top:bottom, left:right] += 1


def _normalize_accumulated(accum: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return accum / weight.clamp_min(1)


def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    data = tensor.detach().float().cpu()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.shape[0] > 3:
        data = data[:3]
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
    rgb = (data.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return rgb[:, :, ::-1]


def _save_color_map(path: Path, values: torch.Tensor, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = normalize_map(values).detach().float().cpu().numpy()
    gray = (data * 255.0).round().astype(np.uint8)
    color = cv2.applyColorMap(gray, colormap)
    if not cv2.imwrite(str(path), color):
        raise RuntimeError(f"cv2.imwrite failed for {path}")
    return color


def _save_overlay(path: Path, base_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    if heat_bgr.shape[:2] != base_bgr.shape[:2]:
        heat_bgr = cv2.resize(heat_bgr, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(base_bgr, 1.0 - alpha, heat_bgr, alpha, 0)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"cv2.imwrite failed for {path}")
    return overlay


def _rank_info() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _shard_samples_for_rank(samples: list, rank: int, world_size: int) -> list:
    if world_size <= 1:
        return samples
    return [sample for idx, sample in enumerate(samples) if idx % world_size == rank]


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        return f"cuda:{int(os.environ['LOCAL_RANK'])}"
    return requested


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    samples = load_samples_file(args.samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    rank, world_size = _rank_info()
    args.device = _resolve_device(args.device)
    samples = _shard_samples_for_rank(samples, rank, world_size)
    write_run_manifest(args, len(samples))
    if args.dry_run:
        print("Fusion attribution dry run complete.")
        return 0
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))
    opt = _prepare_eval_opt(_load_json_config(args.opt))
    opt["gpu_ids"] = [torch.device(args.device).index or 0] if args.device.startswith("cuda") else []
    model = define_Model(opt)
    _inject_lora_for_checkpoint_if_needed(model, opt, args.checkpoint)
    _load_checkpoint_if_available(model, args.checkpoint)
    model.netG.to(torch.device(args.device))
    model.netG.eval()
    dataset = _build_test_dataset(opt)
    folder_index = _build_folder_index(dataset)

    for sample in samples:
        batch = _get_batch_for_sample(dataset, folder_index, sample)
        full_lq = batch["L"]
        full_gt = batch["H"]
        center_idx = _find_frame_index(batch, sample)
        temporal_start = _temporal_window_start(full_lq, center_idx, args.analysis_num_frames)
        num_frames = min(args.analysis_num_frames, full_lq.shape[1])
        height, width = full_lq.shape[-2:]
        tile_size = int(args.analysis_crop_size)
        tile_stride = resolve_tile_stride(tile_size, args.analysis_tile_stride)
        tiles = _spatial_tiles(height, width, tile_size, tile_stride)
        device = next(model.netG.parameters()).device

        adapter = find_fusion_adapter(model.netG)
        restored_accum = torch.zeros(1, 3, height, width, dtype=torch.float32)
        error_accum = torch.zeros(1, 1, height, width, dtype=torch.float32)
        fusion_accum = torch.zeros(1, 1, height, width, dtype=torch.float32)
        weight = torch.zeros(1, 1, height, width, dtype=torch.float32)
        fusion_weight = torch.zeros_like(weight)
        requested_scopes = set(args.cam_scopes)
        cam_tiles: dict[str, list[tuple[torch.Tensor, TileBox]]] = {scope: [] for scope in args.cam_scopes}
        selection_meta = None

        for top, left, bottom, right in tiles:
            tile_box = TileBox(top=top, left=left, bottom=bottom, right=right)
            tile_batch = _prepare_tile_batch(batch, temporal_start, num_frames, top, left, bottom, right)
            lq = tile_batch["L"].to(device)
            gt = tile_batch["H"].to(device)
            flow_spike = _build_flow_spike_for_batch(model, tile_batch, lq)
            probe = FusionProbe(adapter)
            probe.attach()
            model.netG.zero_grad(set_to_none=True)
            output = model.netG(lq, flow_spike=flow_spike) if flow_spike is not None else model.netG(lq)
            if isinstance(output, (tuple, list)):
                output = output[0]
            probe.close()
            if probe.record is None:
                raise RuntimeError("Fusion probe did not capture a forward pass")

            center_output = select_center_frame_tensor(output)
            center_gt = select_center_frame_tensor(gt)
            structured = probe.record.structured_output or {
                "fused_main": probe.record.output,
                "backbone_view": probe.record.output,
                "meta": {},
            }
            selection = select_cam_target(structured)
            if selection_meta is None:
                selection_meta = selection
            local_roi_xyxy = crop_box_to_tile(sample.xyxy, tile_box)
            scope_targets = build_cam_scope_targets(output=center_output, gt=center_gt, roi_xyxy=local_roi_xyxy)

            for scope_name, scope_target in scope_targets.items():
                if scope_name not in requested_scopes:
                    continue
                if scope_name == "roi" and not mask_intersects_tile(sample.xyxy, tile_box):
                    continue
                cam_tile = compute_cam_map(
                    activation=selection.activation,
                    target=scope_target,
                    method=args.cam_method,
                    time_index=selection.time_index,
                ).detach().cpu().unsqueeze(0).unsqueeze(0)
                cam_tiles.setdefault(scope_name, []).append((cam_tile, tile_box))

            error_tile = compute_error_map(center_output, center_gt)

            _accumulate_tile(restored_accum, weight, center_output.detach().cpu(), top, left)
            _accumulate_tile(error_accum, torch.zeros_like(weight), error_tile.detach().cpu().unsqueeze(0).unsqueeze(0), top, left)

            operator = getattr(adapter, "operator", None)
            if operator is not None and hasattr(operator, "explain"):
                reduced = reduce_operator_explanations(operator.explain())
                if "effective_update" in reduced:
                    _accumulate_tile(
                        fusion_accum,
                        fusion_weight,
                        reduced["effective_update"].detach().cpu().unsqueeze(0).unsqueeze(0),
                        top,
                        left,
                    )

            del lq, gt, flow_spike, output, center_output, center_gt, selection, scope_targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        restored_full = _normalize_accumulated(restored_accum, weight)
        error_full_map = _normalize_accumulated(error_accum, weight)
        fusion_full = _normalize_accumulated(fusion_accum, fusion_weight) if fusion_weight.max() > 0 else None
        stitched_cams = {
            scope: stitch_weighted_tiles(
                canvas_shape=(1, 1, height, width),
                tiles=scope_tiles,
            )
            for scope, scope_tiles in cam_tiles.items()
            if scope_tiles
        }

        sample_dir = build_sample_output_dir(args.out, sample)
        inputs_dir = sample_dir / "inputs"
        outputs_dir = sample_dir / "outputs"
        maps_dir = sample_dir / "maps"
        overlays_dir = sample_dir / "overlays"
        full_lq_center = full_lq[:, center_idx : center_idx + 1]
        full_gt_center = full_gt[:, center_idx : center_idx + 1]
        save_rgb_tensor_png(inputs_dir / "blurry_rgb.png", full_lq_center[:, :, :3])
        save_rgb_tensor_png(inputs_dir / "spike_cue.png", full_lq_center[:, :, 3:])
        save_rgb_tensor_png(inputs_dir / "gt.png", full_gt_center)
        save_rgb_tensor_png(outputs_dir / "restored.png", restored_full)
        save_gray_map_png(maps_dir / "error_full.png", normalize_map(error_full_map[0, 0]))

        blurry_bgr = _tensor_to_bgr(full_lq_center[:, :, :3])
        restored_bgr = _tensor_to_bgr(restored_full)
        error_color = _save_color_map(maps_dir / "error_color.png", error_full_map[0, 0])
        cam_overlay = None
        cam_color = None
        for scope_name, cam_map in stitched_cams.items():
            stem = f"cam_{scope_name}"
            np.save(str(maps_dir / f"{stem}_raw.npy"), cam_map[0, 0].numpy())
            save_gray_map_png(maps_dir / f"{stem}_gray.png", normalize_map(cam_map[0, 0]))
            scope_color = _save_color_map(maps_dir / f"{stem}_color.png", cam_map[0, 0])
            _save_overlay(overlays_dir / f"{stem}_on_blurry.png", blurry_bgr, scope_color)
            _save_overlay(overlays_dir / f"{stem}_on_restored.png", restored_bgr, scope_color)
            if scope_name == "fullframe":
                cam_color = scope_color
                cam_overlay = _save_overlay(overlays_dir / "cam_on_blurry.png", blurry_bgr, scope_color)
        if cam_color is None and stitched_cams:
            first_scope = next(iter(stitched_cams))
            first_map = stitched_cams[first_scope]
            cam_color = _save_color_map(maps_dir / "cam_color.png", first_map[0, 0])
            cam_overlay = _save_overlay(overlays_dir / "cam_on_blurry.png", blurry_bgr, cam_color)
        if cam_overlay is None:
            cam_overlay = np.zeros_like(blurry_bgr)
        if cam_color is None:
            cam_color = np.zeros_like(blurry_bgr)
        error_overlay = _save_overlay(overlays_dir / "error_on_restored.png", restored_bgr, error_color)
        fusion_specific_image = cam_color
        if fusion_full is not None:
            np.save(str(maps_dir / "effective_update.npy"), fusion_full[0, 0].numpy())
            save_gray_map_png(maps_dir / "effective_update_gray.png", normalize_map(fusion_full[0, 0]))
            fusion_color = _save_color_map(maps_dir / "effective_update.png", fusion_full[0, 0])
            fusion_specific_image = _save_overlay(overlays_dir / "effective_update_on_blurry.png", blurry_bgr, fusion_color)

        metadata: dict = {
            "sample_id": f"{sample.clip}_{sample.frame}",
            "frame_index": sample.frame_index,
            "mask_type": sample.mask_type,
            "mask_xyxy": list(sample.xyxy),
            "mask_label": sample.mask_label,
            "target": "tiled_charbonnier",
            "cam_method": args.cam_method,
            "checkpoint": args.checkpoint,
            "opt": args.opt,
            "analysis_num_frames": num_frames,
            "analysis_crop_size": tile_size,
            "analysis_tile_stride": tile_stride,
            "tile_count": len(tiles),
        }
        if selection_meta is None:
            raise RuntimeError("No CAM target selection was captured")
        metadata.update(
            build_cam_metadata(
                requested_method=args.cam_method,
                effective_method=args.cam_method,
                scopes_exported=sorted(stitched_cams.keys()),
                default_scope=resolve_cam_default_scope(sorted(stitched_cams.keys())),
                selection=selection_meta,
                analysis_crop_size=tile_size,
                analysis_tile_stride=tile_stride,
                stitch_weight=args.stitch_weight,
                roi_xyxy=sample.xyxy,
            )
        )
        metadata["cam_target_module"] = probe.record.module_name

        if args.save_ig:
            metadata["ig_skipped"] = "Integrated gradients is disabled for tiled full-frame stitching."

        if args.save_pca and operator is not None and hasattr(operator, "explain"):
            metadata["pca_skipped"] = "PCA maps are disabled for tiled full-frame stitching."

        write_json(
            sample_dir / "metadata.json",
            metadata,
        )

        panel_images = {
            "Blurry RGB": cv2.imread(str(inputs_dir / "blurry_rgb.png"), cv2.IMREAD_COLOR),
            "Spike cue": cv2.imread(str(inputs_dir / "spike_cue.png"), cv2.IMREAD_COLOR),
            "Restored": cv2.imread(str(outputs_dir / "restored.png"), cv2.IMREAD_COLOR),
            "Error reduction": error_overlay,
            "Attribution heatmap": cam_overlay,
            "Fusion-specific map": fusion_specific_image,
        }
        make_six_column_panel(sample_dir / "panel.png", panel_images)
    print(f"Fusion attribution complete: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
