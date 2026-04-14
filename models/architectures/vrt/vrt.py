import logging
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
from models.spk_encoder import PixelAdaptiveSpikeEncoder
from models.fusion.factory import create_fusion_operator, create_fusion_adapter
from models.fusion.reducers import build_restoration_reducer

LOGGER = logging.getLogger(__name__)
INPUT_PATH_CONCAT = "concat_path"
INPUT_PATH_DUAL = "dual_path"
INPUT_PATH_DUAL_FALLBACK = "dual_fallback_to_concat_path"


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
        optical_flow (dict): Configuration for the optical flow module.
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
                 sgp_use_partitioned=True,
                 use_flash_attn=True,
                 dcn_config=None,  # DCN configuration dict: {'type': 'DCNv2'/'DCNv4', 'apply_softmax': bool}
                 opt=None):  # Global configuration for initialization
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.recal_all_flows = recal_all_flows
        self.nonblind_denoising = nonblind_denoising
        self.use_sgp = use_sgp
        self.use_flash_attn = use_flash_attn
        raw_input_mode = ((opt or {}).get('netG', {}) or {}).get('input_mode', 'concat')
        self.input_mode = str(raw_input_mode).strip().lower()
        if self.input_mode not in {'concat', 'dual'}:
            raise ValueError(
                f"[VRT] Unsupported netG.input_mode={raw_input_mode!r}; expected 'concat' or 'dual'."
            )

        # Parse DCN configuration
        self.dcn_config = dcn_config or {}
        self.dcn_type = self.dcn_config.get('type', 'DCNv2')
        self.dcn_apply_softmax = self.dcn_config.get('apply_softmax', False)

        spike_encoder_cfg = ((opt or {}).get('netG', {}) or {}).get('spike_encoder', {})
        self.spike_encoder_enabled = bool(spike_encoder_cfg.get('enable', False))
        self.spike_encoder_type = str(spike_encoder_cfg.get('type', 'pase')).lower()
        self.spike_encoder = None
        if self.spike_encoder_enabled:
            if self.spike_encoder_type == 'pase':
                pase_params = spike_encoder_cfg.get('params', {}) or {}
                self.spike_encoder = PixelAdaptiveSpikeEncoder(
                    in_chans=in_chans,
                    out_chans=in_chans,
                    kernel_size=pase_params.get('kernel_size', 3),
                    hidden_chans=pase_params.get('hidden_chans', 32),
                    normalize_kernel=pase_params.get('normalize_kernel', True),
                )
            else:
                raise ValueError(f"Unsupported spike encoder type: {self.spike_encoder_type}")

        fusion_cfg = ((opt or {}).get('netG', {}) or {}).get('fusion', {})
        self.fusion_enabled = bool(fusion_cfg.get('enable', False))
        self.fusion_cfg = fusion_cfg
        self.fusion_operator = None
        self.fusion_adapter = None
        if self.fusion_enabled:
            fusion_placement = str(fusion_cfg.get('placement', 'early'))
            fusion_mode = fusion_cfg.get('mode', 'replace')
            fusion_out_chans = int(fusion_cfg.get('out_chans', 3))
            inject_stages = fusion_cfg.get('inject_stages', [])
            early_cfg = (fusion_cfg.get('early', {}) or {})
            middle_cfg = (fusion_cfg.get('middle', {}) or {})
            early_out_chans = int(early_cfg.get('out_chans', fusion_out_chans))
            middle_out_chans = int(middle_cfg.get('out_chans', fusion_out_chans))
            spike_input_chans = self.in_chans - 3

            middle_rgb_chans = None
            middle_spike_chans = None
            if self.input_mode == 'dual' and self.in_chans <= 3:
                raise ValueError(
                    f"[VRT] input_mode=dual requires in_chans>3 (rgb+spike), got in_chans={self.in_chans}."
                )
            if fusion_placement in {'middle', 'hybrid'}:
                if spike_input_chans <= 0:
                    raise ValueError(
                        f"[VRT] input_mode={self.input_mode}, placement={fusion_placement} "
                        f"requires spike channels in input (in_chans>3). Got in_chans={self.in_chans}."
                    )
                if inject_stages:
                    if not isinstance(inject_stages, (list, tuple)):
                        raise ValueError("Fusion inject_stages must be a list of stage indices.")
                    invalid_stages = [
                        stage for stage in inject_stages
                        if not isinstance(stage, int) or isinstance(stage, bool) or stage < 1 or stage > 7
                    ]
                    if invalid_stages:
                        raise ValueError(
                            f"Fusion inject_stages must be integers in [1, 7], got {invalid_stages}."
                        )
                    stage_dims = [embed_dims[stage - 1] for stage in inject_stages]
                    unique_dims = sorted(set(stage_dims))
                    if len(unique_dims) > 1:
                        raise ValueError(
                            "Fusion inject_stages span multiple feature dims; "
                            f"got dims {unique_dims} for stages {sorted(set(inject_stages))}."
                        )
                    middle_rgb_chans = unique_dims[0]
                    if middle_out_chans != middle_rgb_chans:
                        raise ValueError(
                            f"[VRT] input_mode={self.input_mode}, placement={fusion_placement} "
                            "requires middle out_chans to match injected stage dim. "
                            f"Got out_chans={middle_out_chans}, expected {middle_rgb_chans}."
                        )
                else:
                    middle_rgb_chans = embed_dims[0]
                middle_spike_chans = 1 if fusion_placement == 'hybrid' else spike_input_chans

            full_t_required = (
                fusion_placement == 'hybrid'
                or (fusion_placement == 'early' and bool(early_cfg.get('expand_to_full_t', False)))
            )
            if full_t_required:
                datasets_cfg = ((opt or {}).get('datasets', {}) or {})
                train_cfg = (datasets_cfg.get('train', {}) or {})
                test_cfg = (datasets_cfg.get('test', {}) or {})
                train_nested_recon = ((train_cfg.get('spike', {}) or {}).get('reconstruction', None))
                test_nested_recon = ((test_cfg.get('spike', {}) or {}).get('reconstruction', None))
                recon_cfg = train_nested_recon or test_nested_recon
                if recon_cfg is None:
                    recon_cfg = train_cfg.get('spike_reconstruction', None)
                if recon_cfg is None:
                    recon_cfg = test_cfg.get('spike_reconstruction', None)
                if recon_cfg is None:
                    recon_cfg = ((opt or {}).get('netG', {}) or {}).get('spike_reconstruction', None)
                if recon_cfg is None:
                    recon_cfg = (opt or {}).get('spike_reconstruction', None)
                if isinstance(recon_cfg, dict):
                    recon_type = recon_cfg.get('type', 'spikecv_tfp')
                elif recon_cfg is None:
                    recon_type = 'spikecv_tfp'
                else:
                    recon_type = recon_cfg
                if str(recon_type).strip().lower() != 'spikecv_tfp':
                    raise ValueError("full-T early fusion requires spikecv_tfp")

            operator_name = fusion_cfg.get('operator', 'concat')
            operator_params = fusion_cfg.get('operator_params', {})
            if fusion_placement == 'early':
                self.fusion_operator = create_fusion_operator(
                    operator_name=operator_name,
                    rgb_chans=3,
                    spike_chans=1,
                    out_chans=early_out_chans,
                    operator_params=operator_params,
                )
                self.fusion_adapter = create_fusion_adapter(
                    placement=fusion_placement,
                    operator=self.fusion_operator,
                    mode=fusion_mode,
                    inject_stages=inject_stages,
                    spike_chans=spike_input_chans,
                )
            elif fusion_placement == 'middle':
                self.fusion_operator = create_fusion_operator(
                    operator_name=operator_name,
                    rgb_chans=middle_rgb_chans,
                    spike_chans=middle_spike_chans,
                    out_chans=middle_out_chans,
                    operator_params=operator_params,
                )
                self.fusion_adapter = create_fusion_adapter(
                    placement=fusion_placement,
                    operator=self.fusion_operator,
                    mode=fusion_mode,
                    inject_stages=inject_stages,
                )
            else:
                early_operator = create_fusion_operator(
                    operator_name=operator_name,
                    rgb_chans=3,
                    spike_chans=1,
                    out_chans=early_out_chans,
                    operator_params=operator_params,
                )
                middle_operator = create_fusion_operator(
                    operator_name=operator_name,
                    rgb_chans=middle_rgb_chans,
                    spike_chans=middle_spike_chans,
                    out_chans=middle_out_chans,
                    operator_params=operator_params,
                )
                self.fusion_operator = early_operator
                self.fusion_adapter = create_fusion_adapter(
                    placement=fusion_placement,
                    operator=early_operator,
                    early_operator=early_operator,
                    middle_operator=middle_operator,
                    mode=fusion_mode,
                    inject_stages=inject_stages,
                    spike_chans=spike_input_chans,
                )

        if self.fusion_enabled and fusion_placement in {'early', 'hybrid'}:
            effective_in_chans = early_out_chans
        else:
            effective_in_chans = in_chans

        if self.pa_frames:
            if self.nonblind_denoising:
                conv_first_in_chans = effective_in_chans * 9 + 1
            else:
                conv_first_in_chans = effective_in_chans * 9
        else:
            conv_first_in_chans = effective_in_chans
        self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.output_mode = ((opt or {}).get('netG', {}) or {}).get('output_mode', 'restoration')
        if self.output_mode not in {'restoration', 'interpolation'}:
            raise ValueError(
                f"output_mode must be 'restoration' or 'interpolation', got '{self.output_mode}'"
            )
        reducer_cfg = ((opt or {}).get("netG", {}) or {}).get("restoration_reducer", {})
        if self.output_mode == "restoration":
            self.restoration_reducer = build_restoration_reducer(reducer_cfg)
        else:
            self.restoration_reducer = None
        if self.fusion_enabled and fusion_placement in {'early', 'hybrid'}:
            self.spike_bins = in_chans - 3
        else:
            self.spike_bins = 1

        if self.pa_frames:
            # Instantiate a pluggable optical-flow backend via factory.
            # Accept configuration from `optical_flow` dict (from options).
            of_cfg = optical_flow or {}
            module_name = of_cfg.get('module', 'spynet')
            checkpoint_path = of_cfg.get('checkpoint')
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
                        sgp_use_partitioned=sgp_use_partitioned,
                        use_flash_attn=use_flash_attn,
                        dcn_config={'type': self.dcn_type, 'apply_softmax': self.dcn_apply_softmax})
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
                      sgp_reduction=sgp_reduction,
                      sgp_use_partitioned=sgp_use_partitioned
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

    def build_spike_context(self, x, expand_to_full_t=False):
        if x.dim() != 5:
            return None
        if x.size(2) <= 3:
            return None
        spike = x[:, :, 3:, :, :]
        if spike.numel() == 0:
            return None
        if expand_to_full_t:
            bsz, steps, time_dim, height, width = spike.shape
            spike = spike.reshape(bsz, steps * time_dim, 1, height, width)
        return spike.transpose(1, 2)

    def set_timer(self, timer):
        """Inject a Timer instance for optional timing measurement."""
        self.timer = timer

    def set_input_path_marker(self, marker):
        valid = {INPUT_PATH_CONCAT, INPUT_PATH_DUAL, INPUT_PATH_DUAL_FALLBACK}
        if marker not in valid:
            raise ValueError(f"[VRT] Unsupported input path marker {marker!r}, expected one of {sorted(valid)}.")
        self._input_path_marker = marker

    def _resolve_input_path_marker(self):
        marker = getattr(self, "_input_path_marker", None)
        if marker in {INPUT_PATH_CONCAT, INPUT_PATH_DUAL, INPUT_PATH_DUAL_FALLBACK}:
            return marker
        return INPUT_PATH_DUAL if self.input_mode == "dual" else INPUT_PATH_CONCAT

    def forward(self, x, flow_spike=None):
        # x: (N, D, C, H, W)
        timer = getattr(self, 'timer', None)
        path_marker = self._resolve_input_path_marker()
        LOGGER.info("[VRT] input_path=%s", path_marker)
        if hasattr(self, "_input_path_marker"):
            delattr(self, "_input_path_marker")

        fusion_placement = self.fusion_cfg.get('placement', 'early')
        fusion_hook = None
        spike_ctx = None
        if self.fusion_enabled and fusion_placement in {'middle', 'hybrid'}:
            spike_ctx = self.build_spike_context(
                x,
                expand_to_full_t=(fusion_placement == 'hybrid'),
            )
            fusion_hook = self.fusion_adapter

        if self.pa_frames:
            if self.nonblind_denoising:
                x, noise_level_map = x[:, :, :self.in_chans, :, :], x[:, :, self.in_chans:, :, :]

            x_lq = x.clone()
            x_lq_rgb = self.extract_rgb(x_lq)
            spike_bins = 1

            if self.fusion_enabled and fusion_placement in {'early', 'hybrid'}:
                rgb = x[:, :, :3, :, :]
                spike = x[:, :, 3:, :, :]
                spike_bins = spike.shape[2]
                x = self.fusion_adapter(rgb=rgb, spike=spike)

            if timer is not None:
                with timer.timer('flow_estimation'):
                    flows_backward, flows_forward = self.get_flows(x, flow_spike=flow_spike)
            else:
                flows_backward, flows_forward = self.get_flows(x, flow_spike=flow_spike)

            if timer is not None:
                with timer.timer('flow_warp'):
                    x_backward, x_forward = self.get_aligned_image_2frames(x,  flows_backward[0], flows_forward[0])
            else:
                x_backward, x_forward = self.get_aligned_image_2frames(x,  flows_backward[0], flows_forward[0])

            if self.spike_encoder_enabled:
                if self.spike_encoder is None:
                    raise RuntimeError("Spike encoder is enabled but not initialized.")
                b, d, c, h, w = x.shape
                x_encoded = self.spike_encoder(x.reshape(b * d, c, h, w)).reshape(b, d, c, h, w)
                x = torch.cat([x_encoded, x_backward, x_forward], 2)
            else:
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
                        x_features = self.forward_features(
                            x,
                            flows_backward,
                            flows_forward,
                            fusion_hook=fusion_hook,
                            spike_ctx=spike_ctx,
                        )
                    with timer.timer('conv_after_body'):
                        x = x + self.conv_after_body(x_features.transpose(1, 4)).transpose(1, 4)
                    with timer.timer('conv_last'):
                        x = self.conv_last(x).transpose(1, 2)
                else:
                    x = self.conv_first(x.transpose(1, 2))
                    x = x + self.conv_after_body(
                        self.forward_features(
                            x,
                            flows_backward,
                            flows_forward,
                            fusion_hook=fusion_hook,
                            spike_ctx=spike_ctx,
                        ).transpose(1, 4)
                    ).transpose(1, 4)
                    x = self.conv_last(x).transpose(1, 2)
                if self.output_mode == "restoration" and spike_bins > 1:
                    if self.restoration_reducer is None:
                        raise RuntimeError("restoration_reducer must be initialized for restoration mode.")
                    reduced = self.restoration_reducer(x=x, spike_bins=spike_bins, base_rgb=x_lq_rgb)
                    return reduced + x_lq_rgb
                if self.output_mode == "interpolation" and spike_bins > 1:
                    bsz, frames = x_lq_rgb.shape[:2]
                    chans, height, width = x_lq_rgb.shape[2:]
                    x_lq_rgb_exp = (
                        x_lq_rgb.unsqueeze(2)
                        .expand(bsz, frames, spike_bins, chans, height, width)
                        .reshape(bsz, frames * spike_bins, chans, height, width)
                    )
                    return x + x_lq_rgb_exp
                return x + x_lq_rgb
            else:
                if timer is not None:
                    with timer.timer('conv_first'):
                        x = self.conv_first(x.transpose(1, 2))
                    with timer.timer('forward_features'):
                        x_features = self.forward_features(
                            x,
                            flows_backward,
                            flows_forward,
                            fusion_hook=fusion_hook,
                            spike_ctx=spike_ctx,
                        )
                    with timer.timer('conv_after_body'):
                        x = x + self.conv_after_body(x_features.transpose(1, 4)).transpose(1, 4)
                    with timer.timer('upsample'):
                        x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)
                else:
                    x = self.conv_first(x.transpose(1, 2))
                    x = x + self.conv_after_body(
                        self.forward_features(
                            x,
                            flows_backward,
                            flows_forward,
                            fusion_hook=fusion_hook,
                            spike_ctx=spike_ctx,
                        ).transpose(1, 4)
                    ).transpose(1, 4)
                    x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)
                _, _, C, H, W = x.shape
                x_lq_rgb = torch.nn.functional.interpolate(
                    x_lq_rgb, size=(C, H, W), mode='trilinear', align_corners=False)
                if self.output_mode == "restoration" and spike_bins > 1:
                    if self.restoration_reducer is None:
                        raise RuntimeError("restoration_reducer must be initialized for restoration mode.")
                    reduced = self.restoration_reducer(x=x, spike_bins=spike_bins, base_rgb=x_lq_rgb)
                    return reduced + x_lq_rgb
                if self.output_mode == "interpolation" and spike_bins > 1:
                    bsz, frames = x_lq_rgb.shape[:2]
                    chans, height, width = x_lq_rgb.shape[2:]
                    x_lq_rgb_exp = (
                        x_lq_rgb.unsqueeze(2)
                        .expand(bsz, frames, spike_bins, chans, height, width)
                        .reshape(bsz, frames * spike_bins, chans, height, width)
                    )
                    return x + x_lq_rgb_exp
                return x + x_lq_rgb
        else:
            x_mean = x.mean([1,3,4], keepdim=True)
            x = x - x_mean
            if timer is not None:
                with timer.timer('conv_first'):
                    x = self.conv_first(x.transpose(1, 2))
                with timer.timer('forward_features'):
                    x_features = self.forward_features(
                        x,
                        [],
                        [],
                        fusion_hook=fusion_hook,
                        spike_ctx=spike_ctx,
                    )
                with timer.timer('conv_after_body'):
                    x = x + self.conv_after_body(x_features.transpose(1, 4)).transpose(1, 4)
            else:
                x = self.conv_first(x.transpose(1, 2))
                x = x + self.conv_after_body(
                    self.forward_features(
                        x,
                        [],
                        [],
                        fusion_hook=fusion_hook,
                        spike_ctx=spike_ctx,
                    ).transpose(1, 4)
                ).transpose(1, 4)

            x = torch.cat(torch.unbind(x , 2) , 1)
            x = self.conv_last(self.reflection_pad2d(F.leaky_relu(self.linear_fuse(x), 0.2), pad=3))
            x = torch.stack(torch.split(x, dim=1, split_size_or_sections=3), 1)
            return x + self.extract_rgb(x_mean)

    def get_flows(self, x, flow_spike=None):
        if self.pa_frames == 2:
            flows_backward, flows_forward = self.get_flow_2frames(x, flow_spike=flow_spike)
        elif self.pa_frames == 4:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x, flow_spike=flow_spike)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames
            flows_forward = flows_forward_2frames + flows_forward_4frames
        elif self.pa_frames == 6:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x, flow_spike=flow_spike)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward_6frames, flows_forward_6frames = self.get_flow_6frames(flows_forward_2frames, flows_backward_2frames, flows_forward_4frames, flows_backward_4frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames + flows_backward_6frames
            flows_forward = flows_forward_2frames + flows_forward_4frames + flows_forward_6frames
        return flows_backward, flows_forward

    def get_flow_2frames(self, x, flow_spike=None):
        b, n, c, h, w = x.size()
        
        # Check if we should use spike inputs for optical flow (e.g., SCFlow)
        if getattr(self.spynet, 'input_type', 'rgb') == 'spike':
            if flow_spike is None:
                raise ValueError("SCFlow requires flow_spike input [B,T,25,H,W].")
            if flow_spike.ndim != 5:
                raise ValueError(f"SCFlow requires flow_spike ndim=5 [B,T,25,H,W], got {tuple(flow_spike.shape)}")
            if flow_spike.size(0) != b or flow_spike.size(1) != n:
                raise ValueError(
                    f"SCFlow requires flow_spike matching [B,T], got flow_spike={tuple(flow_spike.shape)} for x={tuple(x.shape)}"
                )
            if flow_spike.size(2) != 25:
                raise ValueError(f"SCFlow requires flow_spike channels=25, got {flow_spike.size(2)}")
            if flow_spike.size(3) != h or flow_spike.size(4) != w:
                raise ValueError(
                    f"SCFlow requires flow_spike spatial size {(h, w)}, got {(flow_spike.size(3), flow_spike.size(4))}"
                )
            x_flow = flow_spike
            c_flow = x_flow.size(2)
            x_1 = x_flow[:, :-1, ...].reshape(-1, c_flow, h, w)
            x_2 = x_flow[:, 1:, ...].reshape(-1, c_flow, h, w)
        else:
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

    def forward_features(self, x, flows_backward, flows_forward, fusion_hook=None, spike_ctx=None):
        x1 = self.stage1(
            x,
            flows_backward[0::4],
            flows_forward[0::4],
            fusion_hook=fusion_hook,
            stage_idx=1,
            spike_ctx=spike_ctx,
        )
        x2 = self.stage2(
            x1,
            flows_backward[1::4],
            flows_forward[1::4],
            fusion_hook=fusion_hook,
            stage_idx=2,
            spike_ctx=spike_ctx,
        )
        x3 = self.stage3(
            x2,
            flows_backward[2::4],
            flows_forward[2::4],
            fusion_hook=fusion_hook,
            stage_idx=3,
            spike_ctx=spike_ctx,
        )
        x4 = self.stage4(
            x3,
            flows_backward[3::4],
            flows_forward[3::4],
            fusion_hook=fusion_hook,
            stage_idx=4,
            spike_ctx=spike_ctx,
        )
        x = self.stage5(
            x4,
            flows_backward[2::4],
            flows_forward[2::4],
            fusion_hook=fusion_hook,
            stage_idx=5,
            spike_ctx=spike_ctx,
        )
        x = self.stage6(
            x + x3,
            flows_backward[1::4],
            flows_forward[1::4],
            fusion_hook=fusion_hook,
            stage_idx=6,
            spike_ctx=spike_ctx,
        )
        x = self.stage7(
            x + x2,
            flows_backward[0::4],
            flows_forward[0::4],
            fusion_hook=fusion_hook,
            stage_idx=7,
            spike_ctx=spike_ctx,
        )
        x = x + x1

        for layer in self.stage8:
            x = layer(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        return x


__all__ = ['VRT']
