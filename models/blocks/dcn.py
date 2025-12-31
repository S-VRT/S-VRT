import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from torch.nn.modules.utils import _pair, _single


class ModulatedDeformConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()


class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """Flow-guided deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
        pa_frames (int): The number of parallel warping frames. Default: 2.

    Ref:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.

    """
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.pa_frames = kwargs.pop('pa_frames', 2)
        super(DCNv2PackFlowGuided, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d((1+self.pa_frames//2) * self.in_channels + self.pa_frames, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 9 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x, x_flow_warpeds, x_current, flows):
        out = self.conv_offset(torch.cat(x_flow_warpeds + [x_current] + flows, dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        if self.pa_frames == 2:
            offset = offset + flows[0].flip(1).repeat(1, offset.size(1)//2, 1, 1)
        elif self.pa_frames == 4:
            offset1, offset2 = torch.chunk(offset, 2, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2], dim=1)
        elif self.pa_frames == 6:
            offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
            offset1, offset2, offset3 = torch.chunk(offset, 3, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset3 = offset3 + flows[2].flip(1).repeat(1, offset3.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2, offset3], dim=1)

        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)


class DCNv4PackFlowGuided(nn.Module):
    """Pure DCNv4 implementation with flow-guided input enhancement.

    This class provides a clean DCNv4 implementation that incorporates optical flow
    information by enhancing the input features, allowing DCNv4's native offset
    generation to be influenced by motion cues.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True, max_residue_magnitude=10, pa_frames=2):
        super(DCNv4PackFlowGuided, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.max_residue_magnitude = max_residue_magnitude
        self.pa_frames = pa_frames

        # Import DCNv4 from our local copy
        try:
            from models.op.dcnv4 import DCNv4
        except ImportError:
            raise ImportError("DCNv4 module not found. Please ensure DCNv4 is properly installed by running: "
                            "cd models/op/dcnv4 && python setup.py develop")

        # Calculate enhanced input channels that include flow information
        # For pa_frames=2: [x, x_warped, x_current, flow_expanded] -> 4 components
        # x, x_warped, x_current: each has in_channels
        # flow_expanded: has in_channels//2 channels (expanded flow)
        self.enhanced_channels = in_channels * 3 + (in_channels // 2)

        # Ensure kernel_size is compatible with DCNv4 (must be square)
        if isinstance(kernel_size, (tuple, list)):
            if kernel_size[0] != kernel_size[1]:
                raise ValueError(f"DCNv4 only supports square kernels, got {kernel_size}")
            dcn_kernel_size = kernel_size[0]
        else:
            dcn_kernel_size = kernel_size

        # Find deformable_groups that satisfies DCNv4's constraint: channels_per_group % 16 == 0
        channels_per_group = self.enhanced_channels // deformable_groups
        if channels_per_group % 16 != 0:
            # Try to find a suitable deformable_groups
            for candidate_groups in range(deformable_groups, 0, -1):
                if self.enhanced_channels % candidate_groups == 0:
                    candidate_channels_per_group = self.enhanced_channels // candidate_groups
                    if candidate_channels_per_group % 16 == 0:
                        actual_deformable_groups = candidate_groups
                        break
            else:
                # If no suitable group found, use group=1 (may be less efficient)
                actual_deformable_groups = 1
        else:
            actual_deformable_groups = deformable_groups

        # Create DCNv4 instance with proper configuration
        # Use offset_scale to control deformation magnitude, as done in FlashInternImage
        self.dcn = DCNv4(
            channels=self.enhanced_channels,
            kernel_size=dcn_kernel_size,
            stride=stride,
            pad=padding,
            dilation=dilation,
            group=actual_deformable_groups,
            offset_scale=1.0,  # Can be adjusted based on VRT's needs
            center_feature_scale=False,
            remove_center=False,
            output_bias=bias
        )

        # Flow-guided offset generation (same as DCNv2PackFlowGuided)
        self.conv_offset = nn.Sequential(
            nn.Conv2d((1+self.pa_frames//2) * self.in_channels + self.pa_frames, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 9 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        """Initialize offset convolution weights."""
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x, x_flow_warpeds, x_current, flows):
        """Pure DCNv4 forward pass with flow-guided input enhancement.

        This method concatenates optical flow information with input features,
        allowing DCNv4's native offset generation to be influenced by motion cues
        without any DCNv2 legacy code.

        Args:
            x: Input feature map [B, C, H, W]
            x_flow_warpeds: Warped features from neighboring frames (list)
            x_current: Current frame features [B, C, H, W]
            flows: Optical flow information (list of [B, 2, H, W])

        Returns:
            Deformably convolved output [B, out_channels, H, W]
        """
        B, C, H, W = x.shape

        # Prepare flow information for concatenation
        flow_features = []
        for flow in flows:
            # Expand flow from [B, 2, H, W] to [B, C//2, H, W] to match feature dimensions
            flow_expanded = flow.repeat(1, C // 2, 1, 1)
            flow_features.append(flow_expanded)

        # Concatenate all inputs: [x, x_flow_warpeds[0], x_current, flow_features[0]]
        # This gives DCNv4 rich contextual information including motion cues
        if len(x_flow_warpeds) > 0 and len(flow_features) > 0:
            enhanced_input = torch.cat([x, x_flow_warpeds[0], x_current, flow_features[0]], dim=1)
        else:
            # Fallback for cases with no flow information
            enhanced_input = torch.cat([x, x_current], dim=1)

        # Ensure channel count matches DCNv4's expectation
        actual_channels = enhanced_input.shape[1]
        if actual_channels != self.enhanced_channels:
            # Adjust channels to match what DCNv4 expects
            if actual_channels < self.enhanced_channels:
                # Pad with zeros
                padding_channels = self.enhanced_channels - actual_channels
                padding = torch.zeros(B, padding_channels, H, W,
                                    device=enhanced_input.device, dtype=enhanced_input.dtype)
                enhanced_input = torch.cat([enhanced_input, padding], dim=1)
            else:
                # Truncate (shouldn't happen in normal usage)
                enhanced_input = enhanced_input[:, :self.enhanced_channels, :, :]

        # Convert to DCNv4 format: [B, H*W, C] -> [B, H, W, C]
        enhanced_bhlc = enhanced_input.view(B, self.enhanced_channels, -1).permute(0, 2, 1).contiguous()
        # DCNv4 expects [N, H*W, C] but internally converts to [N, H, W, C]
        # So we pass [B, H*W, C] directly

        # Apply pure DCNv4
        output = self.dcn(enhanced_bhlc)

        # Convert back to NCHW and take only output channels
        # DCNv4 outputs [N, H*W, C], convert back to [N, C, H, W]
        output = output.permute(0, 2, 1).view(B, self.enhanced_channels, H, W)
        output = output[:, :self.out_channels, :, :].contiguous()

        return output


def get_deformable_module(opt):
    """Factory function to get the appropriate deformable convolution module based on config.

    Args:
        opt: Configuration dictionary

    Returns:
        Deformable convolution class (DCNv2PackFlowGuided or DCNv4PackFlowGuided)
    """
    # Support multiple config paths for backward compatibility
    dcn_type = None

    # Check netG section
    if 'netG' in opt and opt['netG'] and 'dcn_type' in opt['netG']:
        dcn_type = opt['netG']['dcn_type']

    # Default to DCNv2 for backward compatibility
    if dcn_type is None:
        return DCNv2PackFlowGuided

    if dcn_type == 'DCNv4':
        return DCNv4PackFlowGuided
    elif dcn_type == 'DCNv2':
        return DCNv2PackFlowGuided
    else:
        # Unknown type, default to DCNv2
        return DCNv2PackFlowGuided


__all__ = ['ModulatedDeformConv', 'ModulatedDeformConvPack', 'DCNv2PackFlowGuided', 'DCNv4PackFlowGuided', 'get_deformable_module']


