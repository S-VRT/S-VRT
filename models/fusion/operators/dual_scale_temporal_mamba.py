from contextlib import contextmanager, nullcontext
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class _MambaBlock(nn.Module):
    def __init__(self, model_dim: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore
        except (ImportError, ModuleNotFoundError):
            self.mamba = None
            return
        self.norm = nn.LayerNorm(model_dim)
        self.mamba = Mamba(d_model=model_dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.mamba is None:
            raise RuntimeError("mamba_ssm is required for dual_scale_temporal_mamba fusion operator.")
        if not tokens.is_cuda:
            raise RuntimeError("mamba_ssm is required for dual_scale_temporal_mamba fusion operator with CUDA tensors.")
        return tokens + self.mamba(self.norm(tokens))


class _ResidualMambaStage(nn.Module):
    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([_MambaBlock(dim, d_state, d_conv, expand) for _ in range(depth)])
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            seq = block(seq)
        return seq + self.ffn(self.ffn_norm(seq))


class DualScaleTemporalMambaFusionOperator(nn.Module):
    expects_structured_early = True
    frame_contract = "collapsed"

    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, operator_params: Dict):
        super().__init__()
        if rgb_chans != 3:
            raise ValueError("DualScaleTemporalMambaFusionOperator requires rgb_chans=3.")
        if spike_chans <= 0:
            raise ValueError("DualScaleTemporalMambaFusionOperator requires spike_chans>0.")
        if out_chans != 3:
            raise ValueError("DualScaleTemporalMambaFusionOperator requires out_chans=3.")

        self.spike_chans = spike_chans
        token_dim = int(operator_params.get("token_dim", 48))
        patch_stride = int(operator_params.get("patch_stride", 4))
        d_state = int(operator_params.get("d_state", 32))
        d_conv = int(operator_params.get("d_conv", 4))
        expand = int(operator_params.get("expand", 2))
        local_layers = int(operator_params.get("local_layers", 1))
        global_layers = int(operator_params.get("global_layers", 1))
        alpha_init = float(operator_params.get("alpha_init", 0.05))
        gate_bias_init = float(operator_params.get("gate_bias_init", -2.0))
        self.enable_diagnostics = bool(operator_params.get("enable_diagnostics", False))

        self.spike_projector = nn.Sequential(
            nn.Conv2d(1, token_dim, kernel_size=3, stride=patch_stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
        )
        self.rgb_context_encoder = nn.Sequential(
            nn.Conv2d(3, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.local_stage = _ResidualMambaStage(token_dim, d_state, d_conv, expand, local_layers)
        self.summary_gate = nn.Linear(token_dim, 1)
        self.global_stage = _ResidualMambaStage(token_dim, d_state, d_conv, expand, global_layers)
        self.fusion_body = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Conv2d(token_dim, 3, kernel_size=1)
        self.gate_head = nn.Conv2d(token_dim, 3, kernel_size=1)
        self.alpha = nn.Parameter(torch.full((1, 3, 1, 1), alpha_init))
        self._last_diagnostics = {"warmup_stage": "full"}
        self.timer = None

        nn.init.normal_(self.delta_head.weight, std=1e-3)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.normal_(self.gate_head.weight, std=1e-3)
        nn.init.constant_(self.gate_head.bias, gate_bias_init)

    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    def set_timer(self, timer) -> None:
        self.timer = timer

    def _timer(self, name: str):
        if self.timer is None:
            return nullcontext()
        return self.timer.timer(name)

    @contextmanager
    def _profiled_timer(self, name: str):
        if self.timer is None:
            yield
            return
        range_ctx = self.timer.profile_range(name) if hasattr(self.timer, "profile_range") else nullcontext()
        with range_ctx:
            with self._timer(name):
                yield

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != 5:
            raise ValueError("dual_scale_temporal_mamba expects rgb with shape [B,T,3,H,W].")
        if spike_feat.dim() != 5:
            raise ValueError("dual_scale_temporal_mamba expects spike with shape [B,T,S,H,W].")

        bsz, steps, rgb_chans, height, width = rgb_feat.shape
        spike_bsz, spike_steps, spike_chans, spike_h, spike_w = spike_feat.shape
        if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_h, spike_w):
            raise ValueError("rgb and spike must share batch, time, height, and width dimensions")
        if rgb_chans != 3:
            raise ValueError(f"Expected rgb channels=3, got {rgb_chans}")
        if spike_chans != self.spike_chans:
            raise ValueError(f"Expected spike channels={self.spike_chans}, got {spike_chans}")

        with self._profiled_timer("dual_scale_spike_project"):
            spike_flat = spike_feat.reshape(bsz * steps * spike_chans, 1, height, width)
            spike_low = self.spike_projector(spike_flat)
            _, token_dim, low_h, low_w = spike_low.shape
            patch_tokens = low_h * low_w
            spike_low = spike_low.reshape(bsz, steps, spike_chans, token_dim, low_h, low_w)

        with self._profiled_timer("dual_scale_local_mamba"):
            local_seq = spike_low.permute(0, 1, 4, 5, 2, 3).reshape(
                bsz * steps * patch_tokens, spike_chans, token_dim
            )
            local_seq = self.local_stage(local_seq)

        with self._profiled_timer("dual_scale_summary"):
            gate_logits = self.summary_gate(local_seq)
            gate = torch.softmax(gate_logits, dim=1)
            frame_summary = (gate * local_seq).sum(dim=1).reshape(bsz, steps, patch_tokens, token_dim)

        with self._profiled_timer("dual_scale_global_mamba"):
            global_seq = frame_summary.permute(0, 2, 1, 3).reshape(bsz * patch_tokens, steps, token_dim)
            global_seq = self.global_stage(global_seq)
            global_feat = global_seq.reshape(bsz, patch_tokens, steps, token_dim).permute(0, 2, 3, 1)
            spike_ctx = global_feat.reshape(bsz * steps, token_dim, low_h, low_w)
            spike_ctx = F.interpolate(spike_ctx, size=(height, width), mode="bilinear", align_corners=False)

        with self._profiled_timer("dual_scale_writeback"):
            rgb_ctx = self.rgb_context_encoder(rgb_feat.reshape(bsz * steps, 3, height, width))
            fused = self.fusion_body(rgb_ctx + spike_ctx)
            delta = self.delta_head(fused).reshape(bsz, steps, 3, height, width)
            gate_logits = self.gate_head(fused).reshape(bsz, steps, 3, height, width)
            gate_map = torch.sigmoid(gate_logits)
            effective_update = self.alpha.view(1, 1, 3, 1, 1) * gate_map * delta
            out = rgb_feat + effective_update

        if self.enable_diagnostics:
            self._last_diagnostics = {
                "local_norm": float(local_seq.detach().float().norm(dim=-1).mean().item()),
                "global_norm": float(global_seq.detach().float().norm(dim=-1).mean().item()),
                "summary_gate_mean": float(gate.detach().float().mean().item()),
                "effective_update_norm": float(effective_update.detach().float().abs().mean().item()),
                "warmup_stage": "full",
            }
        else:
            self._last_diagnostics = {"warmup_stage": "full"}
        return out


__all__ = ["DualScaleTemporalMambaFusionOperator"]
