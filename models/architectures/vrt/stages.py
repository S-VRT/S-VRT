import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from models.utils.flow import flow_warp

from models.architectures.vrt.attention import WindowAttention
from models.blocks.mlp import Mlp_GEGLU
from models.blocks.dcn import get_deformable_module
from models.utils.windows import get_window_size, compute_mask, window_partition, window_reverse
from models.utils.init import DropPath

class TMSA(nn.Module):
    """ Temporal Mutual Self Attention (TMSA).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        use_sgp (bool): If True, use SGP instead of self-attention. Default: False.
        sgp_w (int): Kernel size for SGP window-level branch. Default: 3.
        sgp_k (int): Multiplier for SGP large kernel. Default: 3.
        sgp_reduction (int): Reduction ratio for SGP instant-level branch. Default: 4.
    """
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(6,8,8),
                 shift_size=(0,0,0),
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
                 sgp_reduction=4,
                 sgp_use_partitioned=True,
                 use_flash_attn=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn
        self.use_sgp = use_sgp
        self.use_flash_attn = use_flash_attn

        assert 0 <= self.shift_size[0] < self.window_size[0]
        assert 0 <= self.shift_size[1] < self.window_size[1]
        assert 0 <= self.shift_size[2] < self.window_size[2]

        self.norm1 = norm_layer(dim)
        # Use SGPWrapper as a drop-in replacement for self-attention when requested
        if self.use_sgp and not mut_attn:
            # lazy import to avoid circular import at module import time
            from models.blocks.sgp import SGPWrapper
            # instantiate SGPWrapper (it may expose `use_inner` to indicate full-block behavior)
            self.attn = SGPWrapper(dim=dim, kernel_size=sgp_w, k=sgp_k, path_pdrop=drop_path, sgp_reduction=sgp_reduction, sgp_use_partitioned=sgp_use_partitioned)
            # record whether the injected attn contains internal identity/FFN
            self._sgp_use_inner = getattr(self.attn, 'use_inner', True)
            # decide outer drop_path behavior:
            # - if attn is full-block (use_inner=True): attn handles its own residual/FFN -> outer should be identity
            # - if attn is operator-only (use_inner=False): outer block should apply DropPath/residual as usual
            if self._sgp_use_inner:
                self.drop_path = nn.Identity()
            else:
                self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        else:
            self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, mut_attn=mut_attn, use_flash_attn=use_flash_attn)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # According to Figure 4, when using SGP, replace the second norm (norm2) with GroupNorm
        if self.use_sgp and not mut_attn:
            # Find a num_groups that divides dim evenly
            for num_groups in [32, 16, 8, 4, 2, 1]:
                if dim % num_groups == 0:
                    break
            self.norm2 = nn.GroupNorm(num_groups, dim)
        else:
            self.norm2 = norm_layer(dim)
        self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape

        # Check if we should skip window partitioning for 5D SGP input
        skip_window_partition = (self.use_sgp and not getattr(self.attn, 'mut_attn', False) and
                                 not getattr(self.attn, 'sgp_use_partitioned', True))

        if skip_window_partition:
            # SGP with 5D input: skip window partitioning
            attn_out = self.attn(x)  # SGPWrapper handles 5D input directly
            return attn_out

        # Standard path: WindowAttention or SGP with partitioned input
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1,2,3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)

        # Apply attention (WindowAttention or SGPWrapper with partitioned input)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1,2,3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        x = self.drop_path(x)
        return x

    def forward_part2(self, x):
        if isinstance(self.norm2, nn.GroupNorm):
            # GroupNorm expects [..., C] format, need to reshape for 5D input [B,D,H,W,C]
            B, D, H, W, C = x.shape
            x_reshaped = x.view(B*D*H*W, C)
            x_norm = self.norm2(x_reshaped)
            x_norm = x_norm.view(B, D, H, W, C)
        else:
            # Standard LayerNorm for 5D input
            x_norm = self.norm2(x)
        return self.drop_path(self.mlp(x_norm))

    def forward(self, x, mask_matrix):
        # Decide behavior based on whether SGP is injected and whether it is full-block (use_inner)
        if self.use_sgp and not getattr(self.attn, 'mut_attn', False):
            # SGP path
            if getattr(self, '_sgp_use_inner', True):
                # full-block: the injected attn implements its own residual/FFN internally,
                # so call it and DO NOT add outer residual/FFN here.
                if self.use_checkpoint_attn:
                    x = torch.utils.checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
                else:
                    x = self.forward_part1(x, mask_matrix)
                # outer FFN/residual intentionally skipped
            else:
                # operator-only: attn returns operator output (no internal identity/FFN).
                # We must preserve the original outer residual + FFN semantics.
                if self.use_checkpoint_attn:
                    attn_out = torch.utils.checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
                else:
                    attn_out = self.forward_part1(x, mask_matrix)

                # apply outer DropPath (configured in __init__ according to use_inner)
                x = x + self.drop_path(attn_out)

                # outer FFN
                if self.use_checkpoint_ffn:
                    x = x + torch.utils.checkpoint.checkpoint(self.forward_part2, x)
                else:
                    x = x + self.forward_part2(x)
        else:
            # Original WindowAttention path: preserve previous behavior (outer residual + FFN)
            if self.use_checkpoint_attn:
                x = x + torch.utils.checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
            else:
                x = x + self.forward_part1(x, mask_matrix)

            if self.use_checkpoint_ffn:
                x = x + torch.utils.checkpoint.checkpoint(self.forward_part2, x)
            else:
                x = x + self.forward_part2(x)

        return x


class TMSAG(nn.Module):
    """ Temporal Mutual Self Attention Group (TMSAG).

    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Input resolution.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        use_sgp (bool): If True, use SGP instead of self-attention. Default: False.
        sgp_w (int): Kernel size for SGP window-level branch. Default: 3.
        sgp_k (int): Multiplier for SGP large kernel. Default: 3.
        sgp_reduction (int): Reduction ratio for SGP instant-level branch. Default: 4.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=[6,8,8], shift_size=None, mut_attn=True, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint_attn=False, use_checkpoint_ffn=False, use_sgp=False, sgp_w=3, sgp_k=3, sgp_reduction=4, sgp_use_partitioned=True, use_flash_attn=True):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size
        self.use_flash_attn = use_flash_attn

        self.blocks = nn.ModuleList([
            TMSA(
                dim=dim,
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
                sgp_reduction=sgp_reduction,
                sgp_use_partitioned=sgp_use_partitioned,
                use_flash_attn=use_flash_attn
            )
            for i in range(depth)])

    def forward(self, x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class RTMSA(nn.Module):
    """ Residual Temporal Mutual Self Attention (RTMSA). Only used in stage 8.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        use_sgp (bool): If True, use SGP instead of self-attention. Default: False.
        sgp_w (int): Kernel size for SGP window-level branch. Default: 3.
        sgp_k (int): Multiplier for SGP large kernel. Default: 3.
        sgp_reduction (int): Reduction ratio for SGP instant-level branch. Default: 4.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint_attn=False, use_checkpoint_ffn=None, use_sgp=False, sgp_w=3, sgp_k=3, sgp_reduction=4, sgp_use_partitioned=True):
        super(RTMSA, self).__init__()
        self.residual_group = TMSAG(dim=dim,
                                    input_resolution=input_resolution,
                                    depth=depth,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    mut_attn=False,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop_path=drop_path,
                                    norm_layer=norm_layer,
                                    use_checkpoint_attn=use_checkpoint_attn,
                                    use_checkpoint_ffn=use_checkpoint_ffn,
                                    use_sgp=use_sgp,
                                    sgp_w=sgp_w,
                                    sgp_k=sgp_k,
                                    sgp_reduction=sgp_reduction,
                                    sgp_use_partitioned=sgp_use_partitioned
                                    )
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


class Stage(nn.Module):
    """Residual Temporal Mutual Self Attention Group and Parallel Warping.

    Args:
        in_dim (int): Number of input channels.
        dim (int): Number of channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        reshape (str): Downscale (down), upscale (up) or keep the size (none).
        max_residue_magnitude (float): Maximum magnitude of the residual of optical flow.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        use_sgp (bool): If True, use SGP instead of self-attention. Default: False.
        sgp_w (int): Kernel size for SGP window-level branch. Default: 3.
        sgp_k (int): Multiplier for SGP large kernel. Default: 3.
        sgp_reduction (int): Reduction ratio for SGP instant-level branch. Default: 4.
        dcn_config (dict): DCN configuration {'type': 'DCNv2'/'DCNv4', 'apply_softmax': bool}.
    """
    def __init__(self,
                 in_dim,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mul_attn_ratio=0.75,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                pa_frames=2,
                deformable_groups=1,
                 reshape=None,
                 max_residue_magnitude=10,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 use_sgp=False,
                 sgp_w=3,
                 sgp_k=3,
                 sgp_reduction=4,
                 sgp_use_partitioned=True,
                 use_flash_attn=True,
                 opt=None,
                 dcn_config=None):
        super(Stage, self).__init__()
        self.pa_frames = pa_frames
        self.use_flash_attn = use_flash_attn

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
                                     sgp_reduction=sgp_reduction,
                                     sgp_use_partitioned=sgp_use_partitioned
                                     )
        self.linear1 = nn.Linear(dim, dim)

        self.residual_group2 = TMSAG(dim=dim,
                                     input_resolution=input_resolution,
                                     depth=depth - int(depth * mul_attn_ratio),
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mut_attn=False,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     use_checkpoint_attn=True,
                                     use_checkpoint_ffn=use_checkpoint_ffn,
                                     use_sgp=use_sgp,
                                     sgp_w=sgp_w,
                                     sgp_k=sgp_k,
                                     sgp_reduction=sgp_reduction,
                                     sgp_use_partitioned=sgp_use_partitioned
                                     )
        self.linear2 = nn.Linear(dim, dim)

        if self.pa_frames:
            DCNClass = get_deformable_module({'dcn': dcn_config})
            self.pa_deform = DCNClass(dim, dim, 3, padding=1, deformable_groups=deformable_groups,
                                     max_residue_magnitude=max_residue_magnitude, pa_frames=pa_frames)
            self.pa_fuse = Mlp_GEGLU(dim * (1 + 2), dim * (1 + 2), dim)

    def forward(self, x, flows_backward, flows_forward, fusion_hook=None, stage_idx=None, spike_ctx=None):
        x = self.reshape(x)
        x = self.linear1(self.residual_group1(x).transpose(1, 4)).transpose(1, 4) + x
        x = self.linear2(self.residual_group2(x).transpose(1, 4)).transpose(1, 4) + x

        if self.pa_frames:
            x = x.transpose(1, 2)
            x_backward, x_forward = getattr(self, f'get_aligned_feature_{self.pa_frames}frames')(x, flows_backward, flows_forward)
            x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)

        if fusion_hook is not None and stage_idx is not None and spike_ctx is not None:
            x = fusion_hook(stage_idx=stage_idx, x=x, spike_ctx=spike_ctx)

        return x

    def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')
            x_backward.insert(0, self.pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_forward.append(self.pa_deform(x_i, [flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')], x[:, i + 1, ...], [flow]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
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

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')
            x_backward.insert(0,
                self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]))

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

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')
            x_forward.append(
                self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_6frames(self, x, flows_backward, flows_forward):
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

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')
            x_backward.insert(0,
                              self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                             x[:, i - 3, ...], [flow1, flow2, flow3]))

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

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')
            x_forward.append(self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                            x[:, i + 1, ...], [flow1, flow2, flow3]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

__all__ = ['TMSA', 'TMSAG', 'RTMSA', 'Stage']
