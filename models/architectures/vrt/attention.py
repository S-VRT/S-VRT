import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import lru_cache

from models.utils.init import trunc_normal_

# Flash Attention imports with fallback
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None


class WindowAttention(nn.Module):
    """ Window based multi-head mutual attention and self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        mut_attn (bool): If True, add mutual attention to the module. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, mut_attn=True, use_flash_attn=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mut_attn = mut_attn
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if self.mut_attn:
            self.register_buffer("position_bias",
                                 self.get_sine_position_encoding(window_size[1:], dim // 2, normalize=True))
            self.qkv_mut = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(2 * dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x_out = self.attention(q, k, v, mask, (B_, N, C), relative_position_encoding=True)

        if self.mut_attn:
            qkv = self.qkv_mut(x + self.position_bias.repeat(1, 2, 1)).reshape(B_, N, 3, self.num_heads,
                                                                               C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
            (q1, q2), (k1, k2), (v1, v2) = torch.chunk(qkv[0], 2, dim=2), torch.chunk(qkv[1], 2, dim=2), torch.chunk(
                qkv[2], 2, dim=2)
            x1_aligned = self.attention(q2, k1, v1, mask, (B_, N // 2, C), relative_position_encoding=False)
            x2_aligned = self.attention(q1, k2, v2, mask, (B_, N // 2, C), relative_position_encoding=False)
            x_out = torch.cat([torch.cat([x1_aligned, x2_aligned], 1), x_out], 2)

        x = self.proj(x_out)
        return x

    def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):
        B_, N, C = x_shape

        if relative_position_encoding:
            # Self-attention: has relative position bias → SDPA mem-efficient backend
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
            # shape: (1, num_heads, N, N)
            attn_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                # mask: (nW, N, N) → (B_, 1, N, N) for broadcast
                shift_mask = mask[:, :N, :N].unsqueeze(1)  # (nW, 1, N, N)
                shift_mask = shift_mask.expand(B_ // nW, nW, 1, N, N).reshape(B_, 1, N, N)
                attn_bias = attn_bias + shift_mask

            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=self.scale)
        else:
            # Mutual attention: no relative position bias
            if (self.use_flash_attn
                    and mask is None
                    and q.dtype in (torch.float16, torch.bfloat16)):
                # flash_attn_func expects (batch, seqlen, nheads, headdim)
                q_ = q.transpose(1, 2)
                k_ = k.transpose(1, 2)
                v_ = v.transpose(1, 2)
                x = flash_attn_func(q_, k_, v_, softmax_scale=self.scale)
                return x.reshape(B_, N, C)
            else:
                attn_mask = None
                if mask is not None:
                    nW = mask.shape[0]
                    shift_mask = mask[:, :N, :N].unsqueeze(1)
                    attn_mask = shift_mask.expand(B_ // nW, nW, 1, N, N).reshape(B_, 1, N, N)
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)

        x = x.transpose(1, 2).reshape(B_, N, C)
        return x

    def get_position_index(self, window_size):
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()


__all__ = ['WindowAttention']


