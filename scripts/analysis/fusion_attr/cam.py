from __future__ import annotations

from dataclasses import dataclass

import torch

from .maps import gradcam_from_activation, hirescam_from_activation
from .targets import masked_charbonnier_target


@dataclass(frozen=True)
class CamTargetSelection:
    activation: torch.Tensor
    tensor_name: str
    time_index: int | None
    frame_contract: str
    main_from_exec_rule: str | None
    spike_bins: int


def fullframe_charbonnier_target(output: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if output.shape != gt.shape:
        raise ValueError(f"output and gt shapes differ: {tuple(output.shape)} vs {tuple(gt.shape)}")
    diff = output - gt
    return -torch.sqrt(diff * diff + eps).mean()


def build_cam_scope_targets(
    output: torch.Tensor,
    gt: torch.Tensor,
    roi_xyxy: tuple[int, int, int, int] | None,
) -> dict[str, torch.Tensor]:
    targets = {"fullframe": fullframe_charbonnier_target(output, gt)}
    if roi_xyxy is None:
        return targets

    x1, y1, x2, y2 = roi_xyxy
    mask = torch.zeros(1, 1, output.shape[-2], output.shape[-1], device=output.device, dtype=output.dtype)
    mask[:, :, y1:y2, x1:x2] = 1.0
    targets["roi"] = masked_charbonnier_target(output, gt, mask)
    return targets


def select_cam_target(record: dict) -> CamTargetSelection:
    fused_main = record["fused_main"]
    backbone_view = record["backbone_view"]
    meta = dict(record.get("meta") or {})
    frame_contract = str(meta.get("frame_contract", "collapsed"))
    main_from_exec_rule = meta.get("main_from_exec_rule")
    spike_bins = int(meta.get("spike_bins", 1) or 1)

    if frame_contract == "expanded" and main_from_exec_rule == "center_subframe":
        frame_idx = fused_main.shape[1] // 2
        time_index = frame_idx * spike_bins + (spike_bins // 2)
        if backbone_view.ndim == 5 and time_index >= backbone_view.shape[1]:
            raise ValueError(
                f"CAM time_index {time_index} exceeds backbone_view time dimension {backbone_view.shape[1]}"
            )
        return CamTargetSelection(
            activation=backbone_view,
            tensor_name="backbone_view",
            time_index=time_index,
            frame_contract=frame_contract,
            main_from_exec_rule=main_from_exec_rule,
            spike_bins=spike_bins,
        )

    return CamTargetSelection(
        activation=fused_main,
        tensor_name="fused_main",
        time_index=fused_main.shape[1] // 2 if fused_main.ndim == 5 else None,
        frame_contract=frame_contract,
        main_from_exec_rule=main_from_exec_rule,
        spike_bins=spike_bins,
    )


def compute_cam_map(
    activation: torch.Tensor,
    target: torch.Tensor,
    method: str,
    time_index: int | None,
) -> torch.Tensor:
    normalized = str(method).strip().lower()
    if normalized in {"gradcam", "fallback"}:
        return gradcam_from_activation(activation, target, time_index=time_index)
    if normalized == "hirescam":
        return hirescam_from_activation(activation, target, time_index=time_index)
    raise ValueError(f"Unsupported cam method: {method}")


def build_cam_metadata(
    *,
    requested_method: str,
    effective_method: str,
    default_scope: str,
    scopes_exported: list[str],
    selection: CamTargetSelection,
    analysis_crop_size: int,
    analysis_tile_stride: int,
    stitch_weight: str,
    roi_xyxy: tuple[int, int, int, int],
) -> dict:
    return {
        "cam_default_scope": default_scope,
        "cam_scopes_exported": scopes_exported,
        "cam_method_requested": requested_method,
        "cam_method_effective": effective_method,
        "cam_target_tensor": selection.tensor_name,
        "frame_contract": selection.frame_contract,
        "main_from_exec_rule": selection.main_from_exec_rule,
        "spike_bins": selection.spike_bins,
        "cam_time_index": selection.time_index,
        "analysis_crop_size": analysis_crop_size,
        "analysis_tile_stride": analysis_tile_stride,
        "analysis_tile_overlap": max(0, analysis_crop_size - analysis_tile_stride),
        "stitch_weight": stitch_weight,
        "roi_mask_xyxy": list(roi_xyxy),
    }
