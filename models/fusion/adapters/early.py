from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


class SpikeUpsample(nn.Module):
    def __init__(self, spike_chans: int):
        super().__init__()
        self.spike_chans = spike_chans
        self.refine = nn.Sequential(
            nn.Conv2d(spike_chans, spike_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spike_chans, spike_chans, kernel_size=3, padding=1),
        )

    def forward(self, spike: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if spike.dim() != 4:
            raise ValueError("spike must be 4D tensor [B_flat, S, H, W]")

        _, spike_chans, _, _ = spike.shape
        if spike_chans != self.spike_chans:
            raise ValueError(f"Expected spike channels={self.spike_chans}, got {spike_chans}")

        upsampled = F.interpolate(
            spike,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(upsampled)


class EarlyFusionAdapter(nn.Module):
    def __init__(
        self,
        operator: nn.Module,
        mode: str = "replace",
        inject_stages: Optional[list] = None,
        spike_chans: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.operator = operator
        self.mode = mode
        self.inject_stages = inject_stages if inject_stages is not None else []
        self.spike_chans = spike_chans
        self.spike_upsample = SpikeUpsample(spike_chans) if spike_chans is not None else None
        self.frame_contract = str(getattr(operator, "frame_contract", "expanded")).strip().lower()
        self.expects_structured_early = bool(
            getattr(operator, "expects_structured_early", False)
        ) or self.frame_contract == "collapsed"
        self.kwargs = kwargs

    def _build_meta(
        self,
        frame_contract: str,
        spike_bins: int,
        main_steps: int,
        exec_steps: int,
        aux_steps: int | None,
        main_from_exec_rule: str | None,
    ) -> dict[str, Any]:
        return {
            "operator_name": self.operator.__class__.__name__,
            "frame_contract": frame_contract,
            "spike_bins": spike_bins,
            "main_steps": main_steps,
            "exec_steps": exec_steps,
            "aux_steps": aux_steps,
            "main_from_exec_rule": main_from_exec_rule,
        }

    @staticmethod
    def _reduce_expanded_exec_to_main(exec_view: torch.Tensor, spike_bins: int) -> torch.Tensor:
        return exec_view[:, spike_bins // 2 :: spike_bins, :, :, :]

    def forward(self, rgb: torch.Tensor, spike: torch.Tensor) -> dict[str, Any]:
        if rgb.dim() != 5:
            raise ValueError("rgb must be 5D tensor [B, N, C, H, W]")
        if spike.dim() != 5:
            raise ValueError("spike must be 5D tensor [B, N, S, H, W]")

        bsz, steps, rgb_chans, height, width = rgb.shape
        spike_bsz, spike_steps, spike_steps_per_frame, spike_height, spike_width = spike.shape

        if (bsz, steps) != (spike_bsz, spike_steps):
            raise ValueError("rgb and spike must share batch size and steps")

        if (spike_height, spike_width) != (height, width):
            if self.spike_upsample is None:
                raise ValueError(
                    "Cannot upsample spike features to match rgb spatial dimensions without spike_chans."
                )
            spike_flat = spike.reshape(bsz * steps, spike_steps_per_frame, spike_height, spike_width)
            spike_flat = self.spike_upsample(spike_flat, target_h=height, target_w=width)
            spike = spike_flat.reshape(bsz, steps, spike_steps_per_frame, height, width)

        frame_contract = str(getattr(self.operator, "frame_contract", self.frame_contract)).strip().lower()
        if frame_contract == "collapsed":
            backbone_view = self.operator(rgb, spike)
            if backbone_view.dim() != 5 or backbone_view.size(1) != steps:
                raise ValueError(
                    "Collapsed early fusion operators must return [B, N, C, H, W] "
                    f"with N={steps}, got {tuple(backbone_view.shape)}."
                )
            return {
                "fused_main": backbone_view,
                "backbone_view": backbone_view,
                "aux_view": None,
                "meta": self._build_meta(
                    frame_contract=frame_contract,
                    spike_bins=spike_steps_per_frame,
                    main_steps=steps,
                    exec_steps=backbone_view.size(1),
                    aux_steps=None,
                    main_from_exec_rule=None,
                ),
            }
        if frame_contract != "expanded":
            raise ValueError(f"Unsupported frame_contract={frame_contract!r}.")

        rgb_rep = rgb.unsqueeze(2).expand(
            bsz, steps, spike_steps_per_frame, rgb_chans, height, width
        )
        rgb_rep = rgb_rep.reshape(bsz, steps * spike_steps_per_frame, rgb_chans, height, width)
        spk = spike.reshape(bsz, steps * spike_steps_per_frame, 1, height, width)
        backbone_view = self.operator(rgb_rep, spk)
        expected_exec_steps = steps * spike_steps_per_frame
        if backbone_view.dim() != 5 or backbone_view.size(1) != expected_exec_steps:
            raise ValueError(
                "Expanded early fusion operators must return [B, N*S, C, H, W] "
                f"with N*S={expected_exec_steps}, got {tuple(backbone_view.shape)}."
            )
        fused_main = self._reduce_expanded_exec_to_main(backbone_view, spike_steps_per_frame)
        return {
            "fused_main": fused_main,
            "backbone_view": backbone_view,
            "aux_view": backbone_view,
            "meta": self._build_meta(
                frame_contract=frame_contract,
                spike_bins=spike_steps_per_frame,
                main_steps=steps,
                exec_steps=backbone_view.size(1),
                aux_steps=backbone_view.size(1),
                main_from_exec_rule="center_subframe",
            ),
        }


__all__ = ["SpikeUpsample", "EarlyFusionAdapter"]
