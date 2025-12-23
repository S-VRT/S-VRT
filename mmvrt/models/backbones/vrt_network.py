"""Adapter exposing the legacy VRT implementation under the new mmvrt package.

This module keeps a thin compatibility layer so `mmvrt.models.backbones.VRTBackbone`
can import `VRT` from a package-local path while the heavy implementation still
resides in the legacy `models` tree. During a later refactor we will split the
large legacy file into smaller `layers/` and `motion/` modules.
"""
from typing import Any

from mmvrt.models.backbones.vrt_impl import VRT  # migrated implementation
from mmvrt.models.motion.spynet import SpyNet

__all__ = ["VRT", "SpyNet"]

"""VRT network implementation composed from migrated submodules.

This implements a practical subset of the original VRT constructor and forward
by composing the migrated building blocks under `mmvrt.models.layers` and
`mmvrt.models.motion`. The goal is functional parity sufficient for training
and inference while keeping the implementation modular.
"""
from typing import Any, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from mmvrt.models.layers.blocks import Stage, RTMSA
from mmvrt.models.layers.upsample import Upsample
from mmvrt.models.motion.spynet import SpyNet
from mmvrt.models.motion.flow_ops import flow_warp


class VRT(nn.Module):
    """Practical VRT composed from migrated components.

    Note: this implementation focuses on modular composition (Stage, SpyNet,
    Upsample) and aims to be compatible with the restorer/head interface.
    """

    def __init__(
        self,
        upscale: int = 1,
        in_chans: int = 3,
        out_chans: int = 3,
        img_size: list = [6, 64, 64],
        window_size: list = [6, 8, 8],
        depths: list = [8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
        indep_reconsts: list = [11, 12],
        embed_dims: list = [120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
        num_heads: list = [6] * 13,
        mul_attn_ratio: float = 0.75,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_path_rate: float = 0.2,
        norm_layer: nn.Module = nn.LayerNorm,
        spynet_path: Optional[str] = None,
        pa_frames: int = 2,
        deformable_groups: int = 16,
        recal_all_flows: bool = False,
        nonblind_denoising: bool = False,
        use_checkpoint_attn: bool = False,
        use_checkpoint_ffn: bool = False,
        no_checkpoint_attn_blocks: list = [],
        no_checkpoint_ffn_blocks: list = [],
        use_sgp: bool = False,
        sgp_w: int = 3,
        sgp_k: int = 3,
        sgp_reduction: int = 4,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.img_size = img_size

        # conv_first: when parallel alignment (pa_frames) is used the input is expanded
        # by concatenating aligned neighbors. Legacy VRT uses a 9x channel expansion
        # for the (current + backward + forward) concatenation when pa_frames is enabled.
        if self.pa_frames:
            conv_first_in_chans = in_chans * 9
        else:
            conv_first_in_chans = in_chans
        self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        if self.pa_frames:
            self.spynet = SpyNet(spynet_path, [2, 3, 4, 5])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        reshapes = ['none', 'down', 'down', 'down', 'up', 'up', 'up']
        scales = [1, 2, 4, 8, 4, 2, 1]

        # create stages 1-7
        for i in range(7):
            setattr(self, f'stage{i+1}',
                    Stage(
                        in_dim=embed_dims[i-1] if i-1 >= 0 else embed_dims[0],
                        dim=embed_dims[i],
                        input_resolution=(img_size[0], img_size[1] // scales[i], img_size[2] // scales[i]),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        mul_attn_ratio=mul_attn_ratio,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                        norm_layer=norm_layer,
                        pa_frames=pa_frames,
                        deformable_groups=deformable_groups,
                        reshape=reshapes[i],
                        max_residue_magnitude=10 / scales[i],
                        use_checkpoint_attn=False if i in no_checkpoint_attn_blocks else use_checkpoint_attn,
                        use_checkpoint_ffn=False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn,
                        use_sgp=use_sgp,
                        sgp_w=sgp_w,
                        sgp_k=sgp_k,
                        sgp_reduction=sgp_reduction
                    )
            )

        # stage8: RTMSA blocks for later depths
        self.stage8 = nn.ModuleList([nn.Sequential(
            Rearrange('n c d h w ->  n d h w c'),
            nn.LayerNorm(embed_dims[6]),
            nn.Linear(embed_dims[6], embed_dims[7]),
            Rearrange('n d h w c -> n c d h w')
        )])
        for i in range(7, len(depths)):
            self.stage8.append(
                RTMSA(dim=embed_dims[i],
                      input_resolution=img_size,
                      depth=depths[i],
                      num_heads=num_heads[i],
                      window_size=[1, window_size[1], window_size[2]] if i in indep_reconsts else window_size,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      qk_scale=qk_scale,
                      drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                      norm_layer=norm_layer,
                      use_checkpoint_attn=False if i in no_checkpoint_attn_blocks else use_checkpoint_attn,
                      use_checkpoint_ffn=False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn,
                      use_sgp=use_sgp,
                      sgp_w=sgp_w,
                      sgp_k=sgp_k,
                      sgp_reduction=sgp_reduction
                      )
            )

        self.norm = norm_layer(embed_dims[-1])
        self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])

        # reconstruction head logic
        if self.pa_frames:
            if self.upscale == 1:
                self.conv_last = nn.Conv3d(embed_dims[0], out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            else:
                num_feat = 64
                self.conv_before_upsample = nn.Sequential(
                    nn.Conv3d(embed_dims[0], num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                    nn.LeakyReLU(inplace=True))
                self.upsample = Upsample(upscale, num_feat)
                self.conv_last = nn.Conv3d(num_feat, out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        else:
            num_feat = 64
            self.linear_fuse = nn.Conv2d(embed_dims[0] * img_size[0], num_feat, kernel_size=1, stride=1)
            self.conv_last = nn.Conv2d(num_feat, out_chans, kernel_size=7, stride=1, padding=0)

    def init_weights(self, pretrained: Optional[str] = None, strict: bool = True):
        # minimal weight init for new components
        if isinstance(pretrained, str):
            pass

    def extract_rgb(self, x: torch.Tensor, channels: int = 3) -> torch.Tensor:
        return x[:, :, :min(channels, x.size(2)), :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: a practical pipeline similar to the legacy network."""
        if self.pa_frames:
            x_lq = x.clone()
            x_lq_rgb = self.extract_rgb(x_lq)
            flows_backward, flows_forward = self.get_flows(x)

            x_backward, x_forward = self.get_aligned_image_2frames(x, flows_backward[0], flows_forward[0])
            x = torch.cat([x, x_backward, x_forward], 2)

            if self.upscale == 1:
                x = self.conv_first(x.transpose(1, 2))
                x = x + self.conv_after_body(self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
                x = self.conv_last(x).transpose(1, 2)
                return x + x_lq_rgb
            else:
                x = self.conv_first(x.transpose(1, 2))
                x = x + self.conv_after_body(self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
                x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)
                _, _, C, H, W = x.shape
                x_lq_rgb = torch.nn.functional.interpolate(x_lq_rgb, size=(C, H, W), mode='trilinear', align_corners=False)
                return x + x_lq_rgb
        else:
            x = self.conv_first(x.transpose(1, 2))
            x = x + self.conv_after_body(self.forward_features(x, [], []).transpose(1, 4)).transpose(1, 4)
            x = torch.cat(torch.unbind(x, 2), 1)
            x = self.conv_last(F.leaky_relu(self.linear_fuse(x), 0.2))
            x = torch.stack(torch.split(x, 3, dim=1), 1)
            return x

    def forward_features(self, x: torch.Tensor, flows_backward: List, flows_forward: List) -> torch.Tensor:
        x1 = self.stage1(x, flows_backward[0::4] if flows_backward else [], flows_forward[0::4] if flows_forward else [])
        x2 = self.stage2(x1, flows_backward[1::4] if flows_backward else [], flows_forward[1::4] if flows_forward else [])
        x3 = self.stage3(x2, flows_backward[2::4] if flows_backward else [], flows_forward[2::4] if flows_forward else [])
        x4 = self.stage4(x3, flows_backward[3::4] if flows_backward else [], flows_forward[3::4] if flows_forward else [])
        x = self.stage5(x4, flows_backward[2::4] if flows_backward else [], flows_forward[2::4] if flows_forward else [])
        x = self.stage6(x + x3, flows_backward[1::4] if flows_backward else [], flows_forward[1::4] if flows_forward else [])
        x = self.stage7(x + x2, flows_backward[0::4] if flows_backward else [], flows_forward[0::4] if flows_forward else [])
        x = x + x1
        for layer in self.stage8:
            x = layer(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x

    def get_flows(self, x: torch.Tensor):
        ''' Get flows for 2 frames, 4 frames or 6 frames.'''
        if self.pa_frames == 2:
            flows_backward, flows_forward = self.get_flow_2frames(x)
        elif self.pa_frames == 4:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames
            flows_forward = flows_forward_2frames + flows_forward_4frames
        elif self.pa_frames == 6:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward_6frames, flows_forward_6frames = self.get_flow_6frames(flows_forward_2frames, flows_backward_2frames, flows_forward_4frames, flows_backward_4frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames + flows_backward_6frames
            flows_forward = flows_forward_2frames + flows_forward_4frames + flows_forward_6frames
        else:
            flows_backward, flows_forward = [], []

        return flows_backward, flows_forward

    def get_flow_2frames(self, x: torch.Tensor):
        b, n, c, h, w = x.size()
        x_flow = self.extract_rgb(x)
        c_flow = x_flow.size(2)
        x_1 = x_flow[:, :-1, :, :, :].reshape(-1, c_flow, h, w)
        x_2 = x_flow[:, 1:, :, :, :].reshape(-1, c_flow, h, w)
        flows_backward = self.spynet(x_1, x_2)
        flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_backward, range(4))]
        flows_forward = self.spynet(x_2, x_1)
        flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_forward, range(4))]
        return flows_backward, flows_forward

    def get_flow_4frames(self, flows_forward, flows_backward):
        """Get flow between t and t+2 from (t,t+1) and (t+1,t+2)."""
        # backward
        d = flows_forward[0].shape[1]
        flows_backward2 = []
        for flows in flows_backward:
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows[:, i - 1, :, :, :]  # flow from i+1 to i
                flow_n2 = flows[:, i, :, :, :]  # flow from i+2 to i+1
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+2 to i
            flows_backward2.append(torch.stack(flow_list, 1))

        # forward
        flows_forward2 = []
        for flows in flows_forward:
            flow_list = []
            for i in range(1, d):
                flow_n1 = flows[:, i, :, :, :]  # flow from i-1 to i
                flow_n2 = flows[:, i - 1, :, :, :]  # flow from i-2 to i-1
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-2 to i
            flows_forward2.append(torch.stack(flow_list, 1))

        return flows_backward2, flows_forward2

    def get_flow_6frames(self, flows_forward, flows_backward, flows_forward2, flows_backward2):
        """Get flow between t and t+3 from (t,t+2) and (t+2,t+3)."""
        # backward
        d = flows_forward2[0].shape[1]
        flows_backward3 = []
        for flows, flows2 in zip(flows_backward, flows_backward2):
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i+2 to i
                flow_n2 = flows[:, i + 1, :, :, :]  # flow from i+3 to i+2
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+3 to i
            flows_backward3.append(torch.stack(flow_list, 1))

        # forward
        flows_forward3 = []
        for flows, flows2 in zip(flows_forward, flows_forward2):
            flow_list = []
            for i in range(2, d + 1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i-2 to i
                flow_n2 = flows[:, i - 2, :, :, :]  # flow from i-3 to i-2
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-3 to i
            flows_forward3.append(torch.stack(flow_list, 1))

        return flows_backward3, flows_forward3

    def get_aligned_image_2frames(self, x: torch.Tensor, flows_backward, flows_forward):
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[:, i - 1, ...]
            x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4'))
        x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4'))
        x_backward = torch.stack(x_backward, 1)
        x_forward = torch.stack(x_forward, 1)
        return x_backward, x_forward


