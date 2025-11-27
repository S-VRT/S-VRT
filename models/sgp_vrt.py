"""
TriDet-inspired Scalable Granularity Perception (SGP) Layer for S-VRT.

This module implements the SGP layer from TriDet paper (arXiv:2303.07347) as an
optional replacement for pure self-attention branches in VRT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SGP(nn.Module):
    """Scalable Granularity Perception Layer from TriDet.

    Implements the formula: f_SGP(x) = φ(x) · FC(x) + ψ(x) · (Conv_w(x) + Conv_{kw}(x)) + x

    Args:
        dim (int): Number of input channels (embedding dimension).
        w (int): Kernel size for the primary convolution in window-level branch. Default: 3.
        k (int): Multiplier for the large kernel size in window-level branch. Default: 3.
        reduction (int): Reduction ratio for the instant-level gating branch. Default: 4.
    """

    def __init__(self, dim, w=3, k=3, reduction=4):
        super().__init__()
        self.dim = dim
        self.w = w
        self.k = k
        self.kw = w * k
        self.reduction = reduction

        # Instant-level branch: FC(x) over channel dimension
        self.fc_main = nn.Linear(dim, dim)

        # Instant-level gating: φ(x), SE-style MLP operating on (B_, C)
        self.fc_instant_gate = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim)
        )

        # Window-level branch: Conv_w(x) and Conv_{kw}(x)
        # Depthwise convs on sequence dimension, operating on (B_, C, N)
        self.conv_w_main = nn.Conv1d(
            dim, dim, kernel_size=w, groups=dim,
            padding=(w - 1) // 2, bias=False
        )
        self.conv_kw_main = nn.Conv1d(
            dim, dim, kernel_size=self.kw, groups=dim,
            padding=(self.kw - 1) // 2, bias=False
        )

        # Window-level gating: ψ(x)
        self.conv_w_gate = nn.Conv1d(
            dim, dim, kernel_size=w, groups=dim,
            padding=(w - 1) // 2, bias=False
        )

    def forward(self, x):
        """
        Args:
            x: (B_, N, C) tensor where B_ = batch_size * num_windows,
                N = sequence length (tokens in window), C = embedding dim

        Returns:
            (B_, N, C) tensor
        """
        B_, N, C = x.shape

        # Instant-level branch: φ(x) · FC(x)
        # Global average pooling over sequence dimension N
        x_mean = x.mean(dim=1)  # (B_, C)
        instant_gate = self.fc_instant_gate(x_mean).view(B_, 1, C)  # (B_, 1, C)
        instant_main = self.fc_main(x)  # (B_, N, C)
        instant_out = instant_gate * instant_main  # (B_, N, C)

        # Window-level branch: ψ(x) · (Conv_w(x) + Conv_{kw}(x))
        # Rearrange for Conv1d: (B_, N, C) -> (B_, C, N)
        x_seq = x.permute(0, 2, 1)  # (B_, C, N)

        conv_w_out = self.conv_w_main(x_seq)   # (B_, C, N)
        conv_kw_out = self.conv_kw_main(x_seq)  # (B_, C, N)
        window_conv_out = conv_w_out + conv_kw_out  # (B_, C, N)
        window_conv_out = window_conv_out.permute(0, 2, 1)  # (B_, N, C)

        window_gate = self.conv_w_gate(x_seq)  # (B_, C, N)
        window_gate = window_gate.permute(0, 2, 1)  # (B_, N, C)
        window_out = window_gate * window_conv_out  # (B_, N, C)

        # Final output: x + instant_out + window_out
        out = x + instant_out + window_out

        return out


class SGPBlock(nn.Module):
    """SGP Block that mirrors the existing Transformer block structure.

    This replaces only the self-attention sub-layer while keeping LN, DropPath,
    and residual connection. The FFN/MLP branch remains external.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        drop_path (float): Stochastic depth rate. Default: 0.0.
        sgp_w (int): Kernel size for SGP window-level branch. Default: 3.
        sgp_k (int): Multiplier for SGP large kernel. Default: 3.
        sgp_reduction (int): Reduction ratio for SGP instant-level branch. Default: 4.
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, drop_path=0.0,
                 sgp_w=3, sgp_k=3, sgp_reduction=4):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.sgp = SGP(dim, w=sgp_w, k=sgp_k, reduction=sgp_reduction)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (B_, N, C) tensor

        Returns:
            (B_, N, C) tensor
        """
        x = x + self.drop_path(self.sgp(self.norm(x)))
        return x


# Import DropPath from the parent network_vrt module
try:
    from .network_vrt import DropPath
except ImportError:
    # Fallback if importing fails
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample."""
        def __init__(self, drop_prob=None):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output
