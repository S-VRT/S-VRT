import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import List, Optional
import numpy as np
from einops.layers.torch import Rearrange

from mmvrt.models.layers.attention import Mlp_GEGLU, WindowAttention
from mmvrt.models.layers.attention_utils import window_partition, window_reverse, get_window_size, compute_mask
from mmvrt.models.layers.drop_path import DropPath
from mmvrt.models.motion.flow_ops import flow_warp
from mmvrt.models.layers.deform_conv import DCNv2PackFlowGuided


class TMSA(nn.Module):
    """ Temporal Mutual Self Attention (TMSA) - ported from legacy implementation. """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=(6, 8, 8),
        shift_size=(0, 0, 0),
        mut_attn=True,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_checkpoint_attn=False,
        use_checkpoint_ffn=False,
        use_sgp=False,
        sgp_w=3,
        sgp_k=3,
        sgp_reduction=4
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn
        self.use_sgp = use_sgp

        assert 0 <= self.shift_size[0] < self.window_size[0]
        assert 0 <= self.shift_size[1] < self.window_size[1]
        assert 0 <= self.shift_size[2] < self.window_size[2]

        self.norm1 = norm_layer(dim)
        # Use WindowAttention by default; legacy SGP handled via separate block when needed
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, mut_attn=mut_attn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        if not (self.use_sgp and not hasattr(self.attn, 'mut_attn')):
            x = self.norm1(x)

        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1), mode='constant')

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)

        if hasattr(self.attn, 'mut_attn'):
            attn_windows = self.attn(x_windows, mask=attn_mask)
        else:
            attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, D + 0, H + 0, W + 0)

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if not (self.use_sgp and not hasattr(self.attn, 'mut_attn')):
            x = self.drop_path(x)

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class TMSAG(nn.Module):
    """ Temporal Mutual Self Attention Group (TMSAG). """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=(6, 8, 8), shift_size=None,
                 mut_attn=True, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_path=0., norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False, use_checkpoint_ffn=False, use_sgp=False, sgp_w=3, sgp_k=3, sgp_reduction=4):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size
        self.blocks = nn.ModuleList([
            TMSA(dim=dim,
                 input_resolution=input_resolution,
                 num_heads=num_heads,
                 window_size=window_size,
                 shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                 mut_attn=mut_attn,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                 norm_layer=norm_layer,
                 use_checkpoint_attn=use_checkpoint_attn,
                 use_checkpoint_ffn=use_checkpoint_ffn,
                 use_sgp=use_sgp,
                 sgp_w=sgp_w,
                 sgp_k=sgp_k,
                 sgp_reduction=sgp_reduction)
            for i in range(depth)
        ])

    def forward(self, x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = x.permute(0, 2, 3, 4, 1)  # n d h w c
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)
        x = x.permute(0, 4, 1, 2, 3)  # b c d h w
        return x


class RTMSA(nn.Module):
    """Residual Temporal Mutual Self Attention (RTMSA) - used in stage 8 of VRT."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint_attn=False, use_checkpoint_ffn=None, use_sgp=False,
                 sgp_w=3, sgp_k=3, sgp_reduction=4):
        super().__init__()
        self.residual_group = TMSAG(dim=dim,
                                    input_resolution=input_resolution,
                                    depth=depth,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    mut_attn=False,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop_path=drop_path,
                                    norm_layer=norm_layer,
                                    use_checkpoint_attn=use_checkpoint_attn,
                                    use_checkpoint_ffn=use_checkpoint_ffn,
                                    use_sgp=use_sgp,
                                    sgp_w=sgp_w,
                                    sgp_k=sgp_k,
                                    sgp_reduction=sgp_reduction)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


class Stage(nn.Module):
    """Stage combining residual attention groups and parallel warping (pa_deform)."""

    def __init__(self, in_dim, dim, input_resolution, depth, num_heads, window_size, mul_attn_ratio=0.75,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path=0., norm_layer=nn.LayerNorm, pa_frames=2,
                 deformable_groups=16, reshape=None, max_residue_magnitude=10, use_checkpoint_attn=False,
                 use_checkpoint_ffn=False, use_sgp=False, sgp_w=3, sgp_k=3, sgp_reduction=4):
        super().__init__()
        self.pa_frames = pa_frames
        if reshape == 'none':
            self.reshape = nn.Sequential(Rearrange('n c d h w -> n d h w c'),
                                         nn.LayerNorm(dim),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'down':
            self.reshape = nn.Sequential(Rearrange('n c d (h neih) (w neiw) -> n d h w (neiw neih c)', neih=2, neiw=2),
                                         nn.LayerNorm(4 * in_dim), nn.Linear(4 * in_dim, dim),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'up':
            self.reshape = nn.Sequential(Rearrange('n (neiw neih c) d h w -> n d (h neih) (w neiw) c', neih=2, neiw=2),
                                         nn.LayerNorm(in_dim // 4), nn.Linear(in_dim // 4, dim),
                                         Rearrange('n d h w c -> n c d h w'))
        else:
            self.reshape = nn.Identity()

        self.residual_group1 = TMSAG(dim=dim,
                                     input_resolution=input_resolution,
                                     depth=int(depth * mul_attn_ratio),
                                     num_heads=num_heads,
                                     window_size=(2, window_size[1], window_size[2]),
                                     mut_attn=True,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     use_checkpoint_attn=use_checkpoint_attn,
                                     use_checkpoint_ffn=use_checkpoint_ffn,
                                     use_sgp=False,
                                     sgp_w=sgp_w,
                                     sgp_k=sgp_k,
                                     sgp_reduction=sgp_reduction)
        self.linear1 = nn.Linear(dim, dim)

        self.residual_group2 = TMSAG(dim=dim,
                                     input_resolution=input_resolution,
                                     depth=depth - int(depth * mul_attn_ratio),
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mut_attn=False,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     use_checkpoint_attn=True,
                                     use_checkpoint_ffn=use_checkpoint_ffn,
                                     use_sgp=use_sgp,
                                     sgp_w=sgp_w,
                                     sgp_k=sgp_k,
                                     sgp_reduction=sgp_reduction)
        self.linear2 = nn.Linear(dim, dim)

        if self.pa_frames:
            # Use channel-aware in_channels for deform conv: concatenation of neighbors
            pa_in_channels = dim * (pa_frames // 2)
            self.pa_deform = DCNv2PackFlowGuided(pa_in_channels, dim, 3, padding=1, deformable_groups=deformable_groups,
                                                 max_residue_magnitude=max_residue_magnitude, pa_frames=pa_frames)
            fuse_channels = dim * (1 + 2 * (pa_frames // 2))
            self.pa_fuse = Mlp_GEGLU(fuse_channels, fuse_channels, dim)
            # helper to safely call pa_deform with runtime channel validation
            def _call_pa_deform(x_first: torch.Tensor, x_flow_warpeds: list, x_current: torch.Tensor, flows: list):
                expected_conv_in = (1 + self.pa_frames // 2) * self.pa_deform.in_channels + self.pa_deform.pa_frames
                actual_conv_in = sum([t.shape[1] for t in x_flow_warpeds]) + x_current.shape[1] + sum([f.shape[1] for f in flows])
                if actual_conv_in != expected_conv_in:
                    diff = expected_conv_in - actual_conv_in
                    B, H, W = x_current.shape[0], x_current.shape[2], x_current.shape[3]
                    if diff > 0:
                        pad_tensor = x_current.new_zeros((B, diff, H, W))
                        flows = flows + [pad_tensor]
                    else:
                        surplus = -diff
                        new_flows = []
                        for f in flows:
                            if surplus == 0:
                                new_flows.append(f)
                                continue
                            c = f.shape[1]
                            if c <= surplus:
                                surplus -= c
                                continue
                            else:
                                new_flows.append(f[:, :c - surplus, :, :])
                                surplus = 0
                        flows = new_flows
                return self.pa_deform(x_first, x_flow_warpeds, x_current, flows)
            self._call_pa_deform = _call_pa_deform

    def forward(self, x, flows_backward, flows_forward):
        x = self.reshape(x)
        x = self.linear1(self.residual_group1(x).transpose(1, 4)).transpose(1, 4) + x
        x = self.linear2(self.residual_group2(x).transpose(1, 4)).transpose(1, 4) + x

        if self.pa_frames:
            x = x.transpose(1, 2)
            x_backward, x_forward = getattr(self, f'get_aligned_feature_{self.pa_frames}frames')(x, flows_backward, flows_forward)
            x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)

        return x

    def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')
            x_backward.insert(0, self._call_pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')
            x_forward.append(self._call_pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 4 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n, 1, -1):
            x_i = x[:, i - 1, ...]
            flow1 = flows_backward[0][:, i - 2, ...]
            if i == n:
                x_ii = torch.zeros_like(x[:, n - 2, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, n - 3, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_backward[1][:, i - 2, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
            x_backward.insert(0,
                self._call_pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(-1, n - 2):
            x_i = x[:, i + 1, ...]
            flow1 = flows_forward[0][:, i + 1, ...]
            if i == -1:
                x_ii = torch.zeros_like(x[:, 1, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_forward[1][:, i, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
            x_forward.append(
                self._call_pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_6frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 6 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n + 1, 2, -1):
            x_i = x[:, i - 2, ...]
            flow1 = flows_backward[0][:, i - 3, ...]
            if i == n + 1:
                x_ii = torch.zeros_like(x[:, -1, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, -1, ...])
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            elif i == n:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = x[:, i, ...]
                flow3 = flows_backward[2][:, i - 3, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i+3 aligned towards i
            x_backward.insert(0,
                              self._call_pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                             x[:, i - 3, ...], [flow1, flow2, flow3]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow1 = flows_forward[0][:, i, ...]
            if i == 0:
                x_ii = torch.zeros_like(x[:, 0, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            elif i == 1:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = x[:, i - 2, ...]
                flow3 = flows_forward[2][:, i - 2, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i-3 aligned towards i
            x_forward.append(self._call_pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                            x[:, i + 1, ...], [flow1, flow2, flow3]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]


