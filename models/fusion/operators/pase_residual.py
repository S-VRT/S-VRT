from typing import Dict

import torch
from torch import nn

from models.spk_encoder import PixelAdaptiveSpikeEncoder


class PaseResidualFusionOperator(nn.Module):
    expects_structured_early = True
    frame_contract = "collapsed"

    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, operator_params: Dict):
        super().__init__()
        if rgb_chans != 3:
            raise ValueError("PaseResidualFusionOperator requires rgb_chans=3.")
        if out_chans != 3:
            raise ValueError("PaseResidualFusionOperator requires out_chans=3.")
        if spike_chans <= 0:
            raise ValueError("PaseResidualFusionOperator requires spike_chans>0.")
        self.spike_chans = spike_chans

        pase_kernel_size = int(operator_params.get("kernel_size", 3))
        pase_hidden_chans = int(operator_params.get("hidden_chans", 32))
        pase_normalize_kernel = bool(operator_params.get("normalize_kernel", True))
        feature_chans = int(operator_params.get("feature_chans", operator_params.get("hidden_chans", 48)))
        alpha_init = float(operator_params.get("alpha_init", 0.05))
        gate_bias_init = float(operator_params.get("gate_bias_init", operator_params.get("init_gate_bias", -2.0)))
        enable_diagnostics = bool(operator_params.get("enable_diagnostics", False))

        self.enable_diagnostics = enable_diagnostics
        self.rgb_context_encoder = nn.Sequential(
            nn.Conv2d(3, feature_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pase = PixelAdaptiveSpikeEncoder(
            in_chans=spike_chans,
            out_chans=feature_chans,
            kernel_size=pase_kernel_size,
            hidden_chans=pase_hidden_chans,
            normalize_kernel=pase_normalize_kernel,
        )
        self.fusion_body = nn.Sequential(
            nn.Conv2d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fusion_writeback_head = nn.ModuleDict(
            {
                "delta": nn.Conv2d(feature_chans, 3, kernel_size=1),
                "gate": nn.Conv2d(feature_chans, 3, kernel_size=1),
            }
        )
        self.alpha = nn.Parameter(torch.full((1, 3, 1, 1), alpha_init))

        self._warmup_stage = "full"
        self._last_diagnostics: dict = {"warmup_stage": "full"}

        nn.init.normal_(self.fusion_writeback_head["delta"].weight, std=1e-3)
        nn.init.zeros_(self.fusion_writeback_head["delta"].bias)
        nn.init.normal_(self.fusion_writeback_head["gate"].weight, std=1e-3)
        nn.init.constant_(self.fusion_writeback_head["gate"].bias, gate_bias_init)

    def set_warmup_stage(self, stage) -> None:
        normalized = "full" if stage in {None, "", "full"} else str(stage).strip().lower()
        if normalized not in {"full", "writeback_only", "token_mixer"}:
            raise ValueError(f"Unsupported PASE residual warmup stage: {stage!r}")
        self._warmup_stage = normalized

        feature_trainable = normalized != "writeback_only"
        for module in (self.rgb_context_encoder, self.pase, self.fusion_body):
            for param in module.parameters():
                param.requires_grad_(feature_trainable)
        for param in self.fusion_writeback_head.parameters():
            param.requires_grad_(True)
        self.alpha.requires_grad_(True)

    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != 5:
            raise ValueError("pase_residual early fusion expects rgb with shape [B, T, 3, H, W].")
        if spike_feat.dim() != 5:
            raise ValueError("pase_residual early fusion expects spike with shape [B, T, S, H, W].")

        bsz, steps, rgb_chans, height, width = rgb_feat.shape
        spike_bsz, spike_steps, spike_chans, spike_height, spike_width = spike_feat.shape
        if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_height, spike_width):
            raise ValueError("rgb and spike must share batch, time, height, and width dimensions")
        if rgb_chans != 3:
            raise ValueError(f"Expected rgb channels=3, got {rgb_chans}")
        if spike_chans != self.spike_chans:
            raise ValueError(
                "Expected spike channels="
                f"{self.spike_chans} at runtime, got {spike_chans}"
            )

        rgb_flat = rgb_feat.reshape(bsz * steps, 3, height, width)
        spike_flat = spike_feat.reshape(bsz * steps, spike_chans, height, width)

        rgb_context = self.rgb_context_encoder(rgb_flat)
        pase_feat = self.pase(spike_flat)
        fused = self.fusion_body(rgb_context + pase_feat)

        delta = self.fusion_writeback_head["delta"](fused).reshape(bsz, steps, 3, height, width)
        gate_logits = self.fusion_writeback_head["gate"](fused).reshape(bsz, steps, 3, height, width)
        gate = torch.sigmoid(gate_logits)

        effective_update = self.alpha.view(1, 1, 3, 1, 1) * gate * delta
        out = rgb_feat + effective_update

        if self.enable_diagnostics:
            self._last_diagnostics = {
                "pase_norm": float(pase_feat.detach().float().norm(dim=1).mean().item()),
                "body_norm": float(fused.detach().float().norm(dim=1).mean().item()),
                "delta_norm": float(delta.detach().float().abs().mean().item()),
                "gate_mean": float(gate.detach().float().mean().item()),
                "effective_update_norm": float(effective_update.detach().float().abs().mean().item()),
                "warmup_stage": self._warmup_stage,
            }
        else:
            self._last_diagnostics = {"warmup_stage": self._warmup_stage}

        return out


__all__ = ["PaseResidualFusionOperator"]
