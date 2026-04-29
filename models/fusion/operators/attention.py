from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class _AttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: float, attn_drop: float, proj_drop: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"Attention block requires model_dim divisible by num_heads, got {model_dim} and {num_heads}."
            )
        hidden_dim = max(model_dim, int(model_dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + self.proj_drop(attn_out)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class AttentionFusionOperator(nn.Module):
    expects_structured_early = True
    frame_contract = "collapsed"

    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, operator_params: Dict):
        super().__init__()
        if rgb_chans != 3:
            raise ValueError("AttentionFusionOperator requires rgb_chans=3.")
        if spike_chans != 1:
            raise ValueError("AttentionFusionOperator requires spike_chans=1 at construction time.")
        if out_chans != 3:
            raise ValueError("AttentionFusionOperator requires out_chans=3.")

        token_dim = int(operator_params.get("token_dim", operator_params.get("model_dim", 48)))
        token_stride = int(operator_params.get("token_stride", 4))
        num_layers = int(operator_params.get("num_layers", 3))
        num_heads = int(operator_params.get("num_heads", 4))
        mlp_ratio = float(operator_params.get("mlp_ratio", 2.0))
        attn_drop = float(operator_params.get("attn_drop", 0.0))
        proj_drop = float(operator_params.get("proj_drop", 0.0))
        alpha_init = float(operator_params.get("alpha_init", 0.05))
        gate_bias_init = float(operator_params.get("gate_bias_init", operator_params.get("init_gate_bias", -2.0)))
        enable_diagnostics = bool(operator_params.get("enable_diagnostics", False))

        self.enable_diagnostics = enable_diagnostics
        self.rgb_context_encoder = nn.Sequential(
            nn.Conv2d(3, token_dim, kernel_size=3, stride=token_stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spike_token_encoder = nn.Sequential(
            nn.Conv2d(1, token_dim, kernel_size=3, stride=token_stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
        )
        self.attention_token_mixer = nn.ModuleList(
            [
                _AttentionBlock(
                    model_dim=token_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(num_layers)
            ]
        )
        self.fusion_writeback_head = nn.ModuleDict(
            {
                "body": nn.Sequential(
                    nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                "delta": nn.Conv2d(token_dim, 3, kernel_size=1),
                "gate": nn.Conv2d(token_dim, 3, kernel_size=1),
            }
        )
        self.alpha = nn.Parameter(torch.full((1, 3, 1, 1), alpha_init))
        self._warmup_stage = "full"
        self._last_diagnostics: Dict = {"warmup_stage": "full"}
        self._last_explain: dict | None = None

        nn.init.normal_(self.fusion_writeback_head["delta"].weight, std=1e-3)
        nn.init.zeros_(self.fusion_writeback_head["delta"].bias)
        nn.init.normal_(self.fusion_writeback_head["gate"].weight, std=1e-3)
        nn.init.constant_(self.fusion_writeback_head["gate"].bias, gate_bias_init)

    def set_warmup_stage(self, stage) -> None:
        normalized = "full" if stage in {None, "", "full"} else str(stage).strip().lower()
        if normalized not in {"full", "writeback_only", "token_mixer"}:
            raise ValueError(f"Unsupported Attention warmup stage: {stage!r}")
        self._warmup_stage = normalized
        token_trainable = normalized != "writeback_only"
        for module in (self.rgb_context_encoder, self.spike_token_encoder, self.attention_token_mixer):
            for param in module.parameters():
                param.requires_grad_(token_trainable)
        for param in self.fusion_writeback_head.parameters():
            param.requires_grad_(True)
        self.alpha.requires_grad_(True)

    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    def explain(self) -> dict:
        if self._last_explain is None:
            return {}
        return dict(self._last_explain)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != 5:
            raise ValueError("attention early fusion expects rgb with shape [B, T, 3, H, W].")
        if spike_feat.dim() != 5:
            raise ValueError("attention early fusion expects spike with shape [B, T, S, H, W].")

        bsz, steps, rgb_chans, height, width = rgb_feat.shape
        spike_bsz, spike_steps, spike_bins, spike_height, spike_width = spike_feat.shape
        if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_height, spike_width):
            raise ValueError("rgb and spike must share batch, time, height, and width dimensions")
        if rgb_chans != 3:
            raise ValueError(f"Expected rgb channels=3, got {rgb_chans}")

        rgb_flat = rgb_feat.reshape(bsz * steps, 3, height, width)
        rgb_low = self.rgb_context_encoder(rgb_flat)
        _, token_dim, token_h, token_w = rgb_low.shape
        rgb_low = rgb_low.reshape(bsz, steps, token_dim, token_h, token_w)

        spike_flat = spike_feat.reshape(bsz * steps * spike_bins, 1, height, width)
        spike_low = self.spike_token_encoder(spike_flat).reshape(bsz, steps, spike_bins, token_dim, token_h, token_w)

        spike_tokens = spike_low.permute(0, 1, 4, 5, 2, 3).reshape(
            bsz * steps * token_h * token_w, spike_bins, token_dim
        )
        rgb_tokens = rgb_low.permute(0, 1, 3, 4, 2).reshape(
            bsz * steps * token_h * token_w, 1, token_dim
        )
        seq = (spike_tokens + rgb_tokens).contiguous()
        for block in self.attention_token_mixer:
            seq = block(seq)

        pooled = seq.mean(dim=1).reshape(bsz, steps, token_h, token_w, token_dim).permute(0, 1, 4, 2, 3)
        fused_low = pooled + rgb_low

        writeback = self.fusion_writeback_head["body"](fused_low.reshape(bsz * steps, token_dim, token_h, token_w))
        delta_low = self.fusion_writeback_head["delta"](writeback)
        gate_logits_low = self.fusion_writeback_head["gate"](writeback)
        delta = F.interpolate(delta_low, size=(height, width), mode="bilinear", align_corners=False).reshape(
            bsz, steps, 3, height, width
        )
        gate_logits = F.interpolate(
            gate_logits_low,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(bsz, steps, 3, height, width)
        gate = torch.sigmoid(gate_logits)
        effective_update = self.alpha.view(1, 1, 3, 1, 1) * gate * delta
        out = rgb_feat + effective_update

        token_energy = spike_tokens.detach().float().norm(dim=-1).mean(dim=-1)
        token_energy = token_energy.reshape(bsz, steps, token_h, token_w)
        token_energy = F.interpolate(
            token_energy.reshape(bsz * steps, 1, token_h, token_w),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(bsz, steps, height, width)
        self._last_explain = {
            "gate": gate.detach(),
            "delta": delta.detach(),
            "effective_update": effective_update.detach(),
            "token_energy": token_energy,
        }

        if self.enable_diagnostics:
            self._last_diagnostics = {
                "token_norm": float(spike_tokens.detach().float().norm(dim=-1).mean().item()),
                "attention_norm": float(seq.detach().float().norm(dim=-1).mean().item()),
                "delta_norm": float(delta.detach().float().abs().mean().item()),
                "gate_mean": float(gate.detach().float().mean().item()),
                "effective_update_norm": float(effective_update.detach().float().abs().mean().item()),
                "warmup_stage": self._warmup_stage,
            }
        else:
            self._last_diagnostics = {"warmup_stage": self._warmup_stage}
        return out


__all__ = ["AttentionFusionOperator"]
