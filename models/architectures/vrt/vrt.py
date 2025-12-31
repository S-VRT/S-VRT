
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from models.flows import compute_flows_2frames
from models.optical_flow import create_optical_flow
from models.architectures.vrt.stages import Stage, RTMSA
from models.blocks.mlp import Mlp_GEGLU
from models.utils.flow import flow_warp
from models.utils.init import trunc_normal_


class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat):
        class Transpose_Dim12(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
                m.append(Transpose_Dim12())
                m.append(nn.PixelShuffle(2))
                m.append(Transpose_Dim12())
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 9 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(Transpose_Dim12())
            m.append(nn.PixelShuffle(3))
            m.append(Transpose_Dim12())
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        else:
            raise ValueError(f'scale {scale} is not supported.')
        super(Upsample, self).__init__(*m)


class VRT(nn.Module):
    """ Video Restoration Transformer (VRT).
        A PyTorch impl of : `VRT: A Video Restoration Transformer`  -
          https://arxiv.org/pdf/2201.00000

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        out_chans (int): Number of output image channels. Default: 3.
        img_size (int | tuple(int)): Size of input image. Default: [6, 64, 64].
        window_size (int | tuple(int)): Window size. Default: (6,8,8).
        depths (list[int]): Depths of each Transformer stage.
        indep_reconsts (list[int]): Layers that extract features of different frames independently.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        spynet_path (str): Pretrained SpyNet model path.
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        recal_all_flows (bool): If True, derive (t,t+2) and (t,t+3) flows from (t,t+1). Default: False.
        nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
        no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
        use_sgp (bool): If True, use SGP instead of self-attention. Default: False.
        sgp_w (int): Kernel size for SGP window-level branch. Default: 3.
        sgp_k (int): Multiplier for SGP large kernel. Default: 3.
        sgp_reduction (int): Reduction ratio for SGP instant-level branch. Default: 4.
    """
    def __init__(self,
                 upscale=4,
                 in_chans=3,
                 out_chans=3,
                 img_size=[6, 64, 64],
                 window_size=[6, 8, 8],
                 depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
                 indep_reconsts=[11, 12],
                 embed_dims=[120] * 7 + [180] * 6,
                 num_heads=[6] * 13,
                 mul_attn_ratio=0.75,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 spynet_path=None,
                 optical_flow=None,
                 pa_frames=2,
                 deformable_groups=16,
                 recal_all_flows=False,
                 nonblind_denoising=False,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 no_checkpoint_attn_blocks=[],
                 no_checkpoint_ffn_blocks=[],
                 use_sgp=False,
                 sgp_w=3,
                 sgp_k=3,
                 sgp_reduction=4,
                 opt=None):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.recal_all_flows = recal_all_flows
        self.nonblind_denoising = nonblind_denoising
        self.use_sgp = use_sgp

        if self.pa_frames:
            if self.nonblind_denoising:
                conv_first_in_chans = in_chans * 9 + 1
            else:
                conv_first_in_chans = in_chans * 9
        else:
            conv_first_in_chans = in_chans
        self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        if self.pa_frames:
            # Instantiate a pluggable optical-flow backend via factory.
            # Accept configuration from `optical_flow` dict (from options) or fall back to spynet defaults.
            of_cfg = optical_flow or {}
            module_name = of_cfg.get('module', 'spynet')
            checkpoint_path = of_cfg.get('checkpoint', spynet_path)
            params = of_cfg.get('params', {})
            # create optical flow module but do NOT move it to a specific device here;
            # upper layers (training script / select_network) are responsible for device placement.
            self.spynet = create_optical_flow(module=module_name,
                                              checkpoint=checkpoint_path,
                                              return_levels=[2, 3, 4, 5],
                                              **(params or {}))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # normalize num_heads to a list matching depths length
        if isinstance(num_heads, int):
            num_heads = [num_heads] * len(depths)
        elif num_heads is None:
            num_heads = [1] * len(depths)
        reshapes = ['none', 'down', 'down', 'down', 'up', 'up', 'up']
        scales = [1, 2, 4, 8, 4, 2, 1]
        use_checkpoint_attns = [False if i in no_checkpoint_attn_blocks else use_checkpoint_attn for i in range(len(depths))]
        use_checkpoint_ffns = [False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn for i in range(len(depths))]

        for i in range(7):
            setattr(self, f'stage{i + 1}',
                    Stage(
                        in_dim=embed_dims[i - 1],
                        dim=embed_dims[i],
                        input_resolution=(img_size[0], img_size[1] // scales[i], img_size[2] // scales[i]),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        mul_attn_ratio=mul_attn_ratio,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                        norm_layer=norm_layer,
                        pa_frames=pa_frames,
                        deformable_groups=deformable_groups,
                        reshape=reshapes[i],
                        max_residue_magnitude=10 / scales[i],
                        use_checkpoint_attn=use_checkpoint_attns[i],
                        use_checkpoint_ffn=use_checkpoint_ffns[i],
                        use_sgp=use_sgp,
                        sgp_w=sgp_w,
                        sgp_k=sgp_k,
                        sgp_reduction=sgp_reduction,
                        opt=opt
                        )
                    )

        # stage8
        self.stage8 = nn.ModuleList(
            [nn.Sequential(
               Rearrange('n c d h w ->  n d h w c'),
                nn.LayerNorm(embed_dims[6]),
                nn.Linear(embed_dims[6], embed_dims[7]),
                Rearrange('n d h w c -> n c d h w')
            )]
        )
        for i in range(7, len(depths)):
            # construct RTMSA blocks for stage8 to match original behavior
            self.stage8.append(
                RTMSA(dim=embed_dims[i],
                      input_resolution=img_size,
                      depth=depths[i],
                      num_heads=num_heads[i],
                      window_size=[1, window_size[1], window_size[2]] if i in indep_reconsts else window_size,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                      norm_layer=norm_layer,
                      use_checkpoint_attn=use_checkpoint_attns[i],
                      use_checkpoint_ffn=use_checkpoint_ffns[i],
                      use_sgp=use_sgp,
                      sgp_w=sgp_w,
                      sgp_k=sgp_k,
                      sgp_reduction=sgp_reduction
                      )
            )

        self.norm = norm_layer(embed_dims[-1])
        self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])

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
            self.linear_fuse = nn.Conv2d(embed_dims[0]*img_size[0], num_feat, kernel_size=1 , stride=1)
            self.conv_last = nn.Conv2d(num_feat, out_chans , kernel_size=7 , stride=1, padding=0)

    def reflection_pad2d(self, x, pad=1):
        x = torch.cat([torch.flip(x[:, :, 1:pad+1, :], [2]), x, torch.flip(x[:, :, -pad-1:-1, :], [2])], 2)
        x = torch.cat([torch.flip(x[:, :, :, 1:pad+1], [3]), x, torch.flip(x[:, :, :, -pad-1:-1], [3])], 3)
        return x

    def extract_rgb(self, x, channels=3):
        return x[:, :, :min(channels, x.size(2)), :, :]

    def set_timer(self, timer):
        """Inject a Timer instance for optional timing measurement."""
        self.timer = timer

    def forward(self, x):
        # x: (N, D, C, H, W)
        timer = getattr(self, 'timer', None)

        if self.pa_frames:
            if self.nonblind_denoising:
                x, noise_level_map = x[:, :, :self.in_chans, :, :], x[:, :, self.in_chans:, :, :]

            x_lq = x.clone()
            x_lq_rgb = self.extract_rgb(x_lq)

            if timer is not None:
                with timer.timer('flow_estimation'):
                    flows_backward, flows_forward = self.get_flows(x)
            else:
                flows_backward, flows_forward = self.get_flows(x)

            if timer is not None:
                with timer.timer('flow_warp'):
                    x_backward, x_forward = self.get_aligned_image_2frames(x,  flows_backward[0], flows_forward[0])
            else:
                x_backward, x_forward = self.get_aligned_image_2frames(x,  flows_backward[0], flows_forward[0])

            x = torch.cat([x, x_backward, x_forward], 2)

            if self.nonblind_denoising:
                x = torch.cat([x, noise_level_map], 2)

            if x.size(2) != self.conv_first.in_channels:
                raise ValueError("Channel mismatch after SGP alignment.")

            if self.upscale == 1:
                if timer is not None:
                    with timer.timer('conv_first'):
                        x = self.conv_first(x.transpose(1, 2))
                    with timer.timer('forward_features'):
                        x_features = self.forward_features(x, flows_backward, flows_forward)
                    with timer.timer('conv_after_body'):
                        x = x + self.conv_after_body(x_features.transpose(1, 4)).transpose(1, 4)
                    with timer.timer('conv_last'):
                        x = self.conv_last(x).transpose(1, 2)
                else:
                    x = self.conv_first(x.transpose(1, 2))
                    x = x + self.conv_after_body(self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
                    x = self.conv_last(x).transpose(1, 2)
                return x + x_lq_rgb
            else:
                if timer is not None:
                    with timer.timer('conv_first'):
                        x = self.conv_first(x.transpose(1, 2))
                    with timer.timer('forward_features'):
                        x_features = self.forward_features(x, flows_backward, flows_forward)
                    with timer.timer('conv_after_body'):
                        x = x + self.conv_after_body(x_features.transpose(1, 4)).transpose(1, 4)
                    with timer.timer('upsample'):
                        x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)
                else:
                    x = self.conv_first(x.transpose(1, 2))
                    x = x + self.conv_after_body(self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
                    x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)
                _, _, C, H, W = x.shape
                x_lq_rgb = torch.nn.functional.interpolate(
                    x_lq_rgb, size=(C, H, W), mode='trilinear', align_corners=False)
                return x + x_lq_rgb
        else:
            x_mean = x.mean([1,3,4], keepdim=True)
            x = x - x_mean
            if timer is not None:
                with timer.timer('conv_first'):
                    x = self.conv_first(x.transpose(1, 2))
                with timer.timer('forward_features'):
                    x_features = self.forward_features(x, [], [])
                with timer.timer('conv_after_body'):
                    x = x + self.conv_after_body(x_features.transpose(1, 4)).transpose(1, 4)
            else:
                x = self.conv_first(x.transpose(1, 2))
                x = x + self.conv_after_body(self.forward_features(x, [], []).transpose(1, 4)).transpose(1, 4)

            x = torch.cat(torch.unbind(x , 2) , 1)
            x = self.conv_last(self.reflection_pad2d(F.leaky_relu(self.linear_fuse(x), 0.2), pad=3))
            x = torch.stack(torch.split(x, dim=1, split_size_or_sections=3), 1)
            return x + self.extract_rgb(x_mean)

    def get_flows(self, x):
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
        return flows_backward, flows_forward

    def get_flow_2frames(self, x):
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
        d = flows_forward[0].shape[1]
        flows_backward2 = []
        for flows in flows_backward:
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows[:, i - 1, :, :, :]
                flow_n2 = flows[:, i, :, :, :]
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_backward2.append(torch.stack(flow_list, 1))

        flows_forward2 = []
        for flows in flows_forward:
            flow_list = []
            for i in range(1, d):
                flow_n1 = flows[:, i, :, :, :]
                flow_n2 = flows[:, i - 1, :, :, :]
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_forward2.append(torch.stack(flow_list, 1))

        return flows_backward2, flows_forward2

    def get_flow_6frames(self, flows_forward, flows_backward, flows_forward2, flows_backward2):
        d = flows_forward2[0].shape[1]
        flows_backward3 = []
        for flows, flows2 in zip(flows_backward, flows_backward2):
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows2[:, i - 1, :, :, :]
                flow_n2 = flows[:, i + 1, :, :, :]
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_backward3.append(torch.stack(flow_list, 1))

        flows_forward3 = []
        for flows, flows2 in zip(flows_forward, flows_forward2):
            flow_list = []
            for i in range(2, d + 1):
                flow_n1 = flows2[:, i - 1, :, :, :]
                flow_n2 = flows[:, i - 2, :, :, :]
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_forward3.append(torch.stack(flow_list, 1))

        return flows_backward3, flows_forward3

    def get_aligned_image_2frames(self, x, flows_backward, flows_forward):
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
        expected_channels = self.in_chans * 4
        if x_backward.size(2) != expected_channels or x_forward.size(2) != expected_channels:
            raise ValueError("SGP alignment produced mismatched channels.")

        return [x_backward, x_forward]

    def forward_features(self, x, flows_backward, flows_forward):
        x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4])
        x2 = self.stage2(x1, flows_backward[1::4], flows_forward[1::4])
        x3 = self.stage3(x2, flows_backward[2::4], flows_forward[2::4])
        x4 = self.stage4(x3, flows_backward[3::4], flows_forward[3::4])
        x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4])
        x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4])
        x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4])
        x = x + x1

        for layer in self.stage8:
            x = layer(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        return x


__all__ = ['VRT']



