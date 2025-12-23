from typing import Any, Optional
import torch
import torch.nn as nn

from mmvrt.registry import MODELS


@MODELS.register_module()
class DeblurHead(nn.Module):
    """Reconstruction head compatible with legacy `network_vrt.py` options.

    This head implements the reconstruction branches found in the original VRT:
    - Parallel-alignment (pa_frames) + conv3d reconstruction for video deblurring (upscale==1)
    - Parallel-alignment + conv_before_upsample + Upsample + conv_last for video SR (upscale>1)
    - Non-parallel path that uses linear_fuse + conv_last (2D conv) for frame-interpolation / non-pa mode

    Args:
        in_channels (int): number of feature channels from backbone (embed_dims[0]).
        out_channels (int): output image channels (default 3).
        pa_frames (bool): whether backbone used PA frames (default True).
        upscale (int): upsampling factor (1 for deblurring).
        num_feat (int): intermediate feature channels for SR branch.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        pa_frames: bool = True,
        upscale: int = 1,
        num_feat: int = 64,
    ):
        super().__init__()
        self.pa_frames = pa_frames
        self.upscale = int(upscale)
        self.out_channels = out_channels

        if self.pa_frames:
            if self.upscale == 1:
                # video deblurring branch: conv3d -> output channels
                self.conv_last = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                self._mode = "pa_deblur"
            else:
                # SR branch: conv_before_upsample + Upsample + conv_last (3D)
                # Import Upsample from the migrated VRT network implementation
                try:
                    from mmvrt.models.backbones.network_vrt import Upsample
                except Exception:
                    # Fallback: small PixelShuffle-based upsample (keeps lightweight)
                    class Upsample(nn.Module):
                        def __init__(self, scale, num_feat):
                            super().__init__()
                            self.scale = scale
                            self.conv = nn.Conv3d(num_feat, num_feat * (scale ** 2), kernel_size=(1, 3, 3), padding=(0, 1, 1))
                            self.ps = nn.PixelShuffle(scale)

                        def forward(self, x):
                            # Expect (B, C, D, H, W) -> apply conv3d then upsample spatially
                            b, c, d, h, w = x.shape
                            out = self.conv(x)
                            # collapse temporal for pixelshuffle then reshape back (simplified)
                            out = out.view(b * d, -1, h, w)
                            out = self.ps(out)
                            oc = out.shape[1]
                            oh = out.shape[2]
                            ow = out.shape[3]
                            out = out.view(b, d, oc, oh, ow).permute(0, 2, 1, 3, 4)
                            return out

                self.conv_before_upsample = nn.Sequential(
                    nn.Conv3d(in_channels, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                    nn.LeakyReLU(inplace=True),
                )
                self.upsample = Upsample(self.upscale, num_feat)
                self.conv_last = nn.Conv3d(num_feat, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                self._mode = "pa_sr"
            # init weights similar to legacy: convs use kaiming for conv layers
            for m in self.conv_before_upsample.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.kaiming_normal_(self.conv_last.weight, mode='fan_out', nonlinearity='relu')
            if self.conv_last.bias is not None:
                nn.init.zeros_(self.conv_last.bias)
        else:
            # non-pa mode: fuse temporal frames and apply 2D conv reconstruction
            # The legacy used: linear_fuse = Conv2d(embed_dims[0]*img_size[0], num_feat, kernel_size=1)
            # Here we provide a flexible linear_fuse compatible with stacked temporal channel input.
            self.linear_fuse = nn.Conv2d(in_channels, num_feat, kernel_size=1, stride=1)
            self.conv_last = nn.Conv2d(num_feat, out_channels, kernel_size=7, stride=1, padding=0)
            self._mode = "non_pa"
            # initialize linear_fuse and conv_last similarly to legacy patterns
            nn.init.kaiming_normal_(self.linear_fuse.weight, mode='fan_out', nonlinearity='relu')
            if self.linear_fuse.bias is not None:
                nn.init.zeros_(self.linear_fuse.bias)
            nn.init.kaiming_normal_(self.conv_last.weight, mode='fan_out', nonlinearity='relu')
            if self.conv_last.bias is not None:
                nn.init.zeros_(self.conv_last.bias)

    def forward(self, x: torch.Tensor, img_temporal: Optional[int] = None) -> torch.Tensor:
        """Forward accepts backbone output in common shapes:
        - (B, C, D, H, W) or (B, T, C, H, W) or (B, T, C, H, W) depending on upstream.
        If the input is (B, T, C, H, W), we try to adapt to expected conv dims.
        """
        if x is None:
            return x

        # Normalize common shapes:
        # If (B, T, C, H, W), convert to (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] != x.shape[2]:
            # Could be (B, T, C, H, W) or (B, C, D, H, W); detect common pattern:
            b, a, c, h, w = x.shape
            # If channel dim equals 3 or typical small number -> assume (B, T, C, H, W)
            if c in (1, 3, 4):
                x = x.permute(0, 2, 1, 3, 4)  # -> (B, C, T, H, W)

        # Now common shape (B, C, D, H, W)
        if self._mode == "pa_deblur":
            if x.dim() == 5:
                out = self.conv_last(x)
            elif x.dim() == 4:
                # (B, C, H, W) -> convert to (B, C, 1, H, W)
                out = self.conv_last(x.unsqueeze(2)).squeeze(2)
            else:
                raise ValueError(f"Unsupported tensor dim for pa_deblur head: {x.dim()}")
            return out
        elif self._mode == "pa_sr":
            if x.dim() != 5:
                # try to promote to 5D
                if x.dim() == 4:
                    x = x.unsqueeze(2)
                else:
                    raise ValueError(f"Unsupported tensor dim for pa_sr head: {x.dim()}")
            feat = self.conv_before_upsample(x)
            up = self.upsample(feat)
            out = self.conv_last(up)
            return out
        else:  # non_pa
            # Expect (B, C, D, H, W) or (B, C, H, W)
            if x.dim() == 5:
                b, c, d, h, w = x.shape
                # fuse temporal by concatenation along channels: (B, C*d, H, W)
                x2 = x.permute(0, 2, 1, 3, 4).contiguous().view(b, c * d, h, w)
            elif x.dim() == 4:
                x2 = x
            else:
                raise ValueError(f"Unsupported tensor dim for non-pa head: {x.dim()}")
            fused = self.linear_fuse(x2)
            out = self.conv_last(fused)
            return out


