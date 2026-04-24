from typing import Dict

import torch
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
            raise RuntimeError("mamba_ssm is required for mamba fusion operator.")
        if not tokens.is_cuda:
            raise RuntimeError(
                "mamba_ssm is required for mamba fusion operator with CUDA tensors."
            )
        return tokens + self.mamba(self.norm(tokens))


class MambaFusionOperator(nn.Module):
    expects_structured_early = True
    frame_contract = "collapsed"

    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, operator_params: Dict):
        super().__init__()
        if rgb_chans != 3:
            raise ValueError("MambaFusionOperator requires rgb_chans=3.")
        if spike_chans != 1:
            raise ValueError("MambaFusionOperator requires spike_chans=1 at construction time.")
        if out_chans != 3:
            raise ValueError("MambaFusionOperator requires out_chans=3.")

        self.rgb_chans = rgb_chans
        self.spike_chans = spike_chans
        self.out_chans = out_chans
        self.operator_params = operator_params

        model_dim = int(operator_params.get("model_dim", 48))
        d_state = int(operator_params.get("d_state", 32))
        d_conv = int(operator_params.get("d_conv", 4))
        expand = int(operator_params.get("expand", 2))
        num_layers = int(operator_params.get("num_layers", 3))
        init_gate_bias = float(operator_params.get("init_gate_bias", -5.0))

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, model_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1),
        )
        self.spike_token_proj = nn.Conv2d(1, model_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [_MambaBlock(model_dim=model_dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)]
        )
        self.correction_head = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_dim, 3, kernel_size=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_dim, 3, kernel_size=1),
        )

        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, init_gate_bias)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != 5:
            raise ValueError("mamba early fusion expects rgb with shape [B, T, 3, H, W].")
        if spike_feat.dim() != 5:
            raise ValueError("mamba early fusion expects spike with shape [B, T, S, H, W].")

        bsz, steps, rgb_chans, height, width = rgb_feat.shape
        spike_bsz, spike_steps, spike_bins, spike_height, spike_width = spike_feat.shape
        if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_height, spike_width):
            raise ValueError("rgb and spike must share batch, time, height, and width dimensions")
        if rgb_chans != 3:
            raise ValueError(f"Expected rgb channels=3, got {rgb_chans}")

        rgb_flat = rgb_feat.reshape(bsz * steps, 3, height, width)
        rgb_ctx = self.rgb_encoder(rgb_flat).reshape(bsz, steps, -1, height, width)

        spike_flat = spike_feat.reshape(bsz * steps * spike_bins, 1, height, width)
        spike_tokens = self.spike_token_proj(spike_flat).reshape(bsz, steps, spike_bins, -1, height, width)
        tokens = spike_tokens + rgb_ctx.unsqueeze(2)

        model_dim = tokens.size(3)
        seq = tokens.permute(0, 1, 4, 5, 2, 3).reshape(bsz * steps * height * width, spike_bins, model_dim)
        for block in self.blocks:
            seq = block(seq)
        pooled = seq.mean(dim=1).reshape(bsz, steps, height, width, model_dim).permute(0, 1, 4, 2, 3)

        pooled_flat = pooled.reshape(bsz * steps, model_dim, height, width)
        correction = self.correction_head(pooled_flat).reshape(bsz, steps, 3, height, width)
        gate = torch.sigmoid(self.gate_head(pooled_flat)).reshape(bsz, steps, 3, height, width)
        return rgb_feat + gate * correction


__all__ = ['MambaFusionOperator']
