import torch
import torch.nn as nn

from mmvrt.models.layers.drop_path import DropPath


class SGP(nn.Module):
    """Scalable Granularity Perception Layer (SGP) - migrated from legacy code."""

    def __init__(self, dim, w=3, k=3, reduction=4):
        super().__init__()
        self.dim = dim
        self.w = w
        self.k = k
        self.kw = w * k
        self.reduction = reduction

        # Instant-level branch
        self.fc_main = nn.Linear(dim, dim)
        self.fc_instant_gate = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
        )

        # Window-level branch (depthwise Conv1d)
        self.conv_w_main = nn.Conv1d(dim, dim, kernel_size=w, groups=dim, padding=(w - 1) // 2, bias=False)
        self.conv_kw_main = nn.Conv1d(dim, dim, kernel_size=self.kw, groups=dim, padding=(self.kw - 1) // 2, bias=False)
        self.conv_w_gate = nn.Conv1d(dim, dim, kernel_size=w, groups=dim, padding=(w - 1) // 2, bias=False)

    def forward(self, x):
        # x: (B_, N, C)
        B_, N, C = x.shape
        x_mean = x.mean(dim=1)  # (B_, C)
        instant_gate = self.fc_instant_gate(x_mean).view(B_, 1, C)
        instant_main = self.fc_main(x)  # (B_, N, C)
        instant_out = instant_gate * instant_main

        x_seq = x.permute(0, 2, 1)  # (B_, C, N)
        conv_w_out = self.conv_w_main(x_seq)
        conv_kw_out = self.conv_kw_main(x_seq)
        window_conv_out = (conv_w_out + conv_kw_out).permute(0, 2, 1)  # (B_, N, C)

        window_gate = self.conv_w_gate(x_seq).permute(0, 2, 1)
        window_out = window_gate * window_conv_out

        out = x + instant_out + window_out
        return out


class SGPBlock(nn.Module):
    """SGP Block compatible with existing residual block pattern."""

    def __init__(self, dim, norm_layer=nn.LayerNorm, drop_path: float = 0.0, sgp_w: int = 3, sgp_k: int = 3, sgp_reduction: int = 4):
        super().__init__()
        self.norm = norm_layer(dim)
        self.sgp = SGP(dim, w=sgp_w, k=sgp_k, reduction=sgp_reduction)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x: (B_, N, C)
        return x + self.drop_path(self.sgp(self.norm(x)))


