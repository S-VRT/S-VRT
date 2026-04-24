from __future__ import annotations

import torch
from torch import nn


class GatedFusionOperator(nn.Module):
    """Additive gated fusion: out = rgb + gate(concat) * correction(concat).

    Base path is identity (no rgb_proj), so colors are preserved by design.
    correction last layer and gate pre-sigmoid bias are zero-initialized so
    the operator starts as a pure passthrough.
    """

    def __init__(
        self,
        rgb_chans: int,
        spike_chans: int,
        out_chans: int,
        operator_params: dict,
    ):
        super().__init__()
        self.rgb_chans = rgb_chans
        self.spike_chans = spike_chans
        self.out_chans = out_chans
        if out_chans != rgb_chans:
            raise ValueError(
                f'GatedFusionOperator requires out_chans == rgb_chans for additive residual, '
                f'got out_chans={out_chans}, rgb_chans={rgb_chans}'
            )
        hidden_chans = int(operator_params.get('hidden_chans', 32))
        in_chans = rgb_chans + spike_chans

        self.correction = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_chans, out_chans, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_chans, out_chans, kernel_size=1),
            nn.Sigmoid(),
        )

        # correction output starts at 0
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        # gate pre-sigmoid bias = -5 → Sigmoid(-5) ≈ 0.007
        nn.init.constant_(self.gate[2].bias, -5.0)
        self._last_explain: dict | None = None

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != spike_feat.dim():
            raise ValueError('rgb and spike must have the same number of dimensions')
        if rgb_feat.dim() == 5:
            bsz, steps, rgb_chans, height, width = rgb_feat.shape
            spike_bsz, spike_steps, spike_chans, spike_height, spike_width = spike_feat.shape
            if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_height, spike_width):
                raise ValueError('rgb and spike must share batch, time, height, and width dimensions')
            if rgb_chans != self.rgb_chans:
                raise ValueError(f'Expected rgb channels={self.rgb_chans}, got {rgb_chans}')
            if spike_chans != self.spike_chans:
                raise ValueError(f'Expected spike channels={self.spike_chans}, got {spike_chans}')
            rgb_flat = rgb_feat.reshape(bsz * steps, rgb_chans, height, width)
            spike_flat = spike_feat.reshape(bsz * steps, spike_chans, height, width)
            concat = torch.cat([rgb_flat, spike_flat], dim=1)
            gate = self.gate(concat)
            correction = self.correction(concat)
            effective_update = gate * correction
            self._last_explain = {
                'gate': gate.reshape(bsz, steps, self.out_chans, height, width).detach(),
                'correction': correction.reshape(bsz, steps, self.out_chans, height, width).detach(),
                'effective_update': effective_update.reshape(bsz, steps, self.out_chans, height, width).detach(),
            }
            out = rgb_flat + effective_update
            return out.reshape(bsz, steps, self.out_chans, height, width)
        if rgb_feat.dim() == 4:
            bsz, rgb_chans, height, width = rgb_feat.shape
            spike_bsz, spike_chans, spike_height, spike_width = spike_feat.shape
            if (bsz, height, width) != (spike_bsz, spike_height, spike_width):
                raise ValueError('rgb and spike must share batch, height, and width dimensions')
            if rgb_chans != self.rgb_chans:
                raise ValueError(f'Expected rgb channels={self.rgb_chans}, got {rgb_chans}')
            if spike_chans != self.spike_chans:
                raise ValueError(f'Expected spike channels={self.spike_chans}, got {spike_chans}')
            concat = torch.cat([rgb_feat, spike_feat], dim=1)
            gate = self.gate(concat)
            correction = self.correction(concat)
            effective_update = gate * correction
            self._last_explain = {
                'gate': gate.detach(),
                'correction': correction.detach(),
                'effective_update': effective_update.detach(),
            }
            return rgb_feat + effective_update
        raise ValueError('Expected rgb and spike features with 4 or 5 dimensions')

    def explain(self) -> dict[str, torch.Tensor]:
        if self._last_explain is None:
            return {}
        return dict(self._last_explain)


__all__ = ['GatedFusionOperator']
