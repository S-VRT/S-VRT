import torch
import torch.nn.functional as F
from torch import nn


def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    # If keep_prob <= 0.0 then the path is always dropped; return zeros directly
    if keep_prob <= 0.0:
        return torch.zeros_like(x)

    # Create a Bernoulli mask in a numerically-stable way
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    mask = (rand_tensor < keep_prob).to(x.dtype)

    # Scale the surviving paths by 1/keep_prob to preserve expectation
    output = x * mask / keep_prob
    return output


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels
        x_fp32 = x.float()
        mu = torch.mean(x_fp32, dim=1, keepdim=True)
        res_x = x_fp32 - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)
        if self.affine:
            out = out * self.weight.float() + self.bias.float()
        return out.to(x.dtype)


class SGPBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            path_pdrop=0.0,  # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            downsample_type='max',
            init_conv_vars=1,  # init gaussian variance for the weight
            use_inner=True,  # if True, keep internal identity + FFN (TriDet original). If False, disable them.
            sgp_reduction=4  # reduction ratio for instant-level branch MLP
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        self.kernel_size = kernel_size
        self.stride = n_ds_stride
        self.sgp_reduction = sgp_reduction

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)
        # Find a num_groups that divides n_embd evenly for GroupNorm
        for num_groups in [32, 16, 8, 4, 2, 1]:
            if n_embd % num_groups == 0:
                break
        self.gn = nn.GroupNorm(num_groups, n_embd)
        # control whether SGPBlock keeps its internal identity + FFN
        # When integrating into VRT we will often want use_inner=False so outer block handles residual+FFN.
        self.use_inner = use_inner

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)

        # Instant-level branch: global average pooling -> MLP compression -> final FC
        reduced_dim = n_embd // self.sgp_reduction
        self.instant_mlp = nn.Sequential(
            nn.Conv1d(n_embd, reduced_dim, 1, stride=1, padding=0),
            act_layer(),
            nn.Conv1d(reduced_dim, n_embd, 1, stride=1, padding=0)
        )
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # input
        if n_ds_stride > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)

        # Initialize instant-level MLP
        for layer in self.instant_mlp:
            if isinstance(layer, nn.Conv1d):
                torch.nn.init.normal_(layer.weight, 0, init_conv_vars)
                torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x, mask):
        # X shape: B, C, T
        if x.dim() != 3:
            raise ValueError(f"SGPBlock expects input shape [B, C, T] where T is temporal length; got {x.shape}.")
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()

        out = self.ln(x)
        # Following Figure 4: instant-level and window-level branches
        # instant-level: phi gates fc branch
        # Global average pooling -> MLP compression -> ReLU -> phi [B, C, 1]
        global_avg = out.mean(dim=-1, keepdim=True)  # [B, C, 1]
        phi = torch.relu(self.instant_mlp(global_avg))  # [B, C, 1] through MLP compression
        fc_branch = self.fc(out)  # pointwise conv [B, C, T]

        # window-level: psi gates conv branches
        psi = self.psi(out)  # produces psi for window-level gating [B, C, T]
        convw = self.convw(out)  # window branch 1 [B, C, T]
        convkw = self.convkw(out)  # window branch 2 [B, C, T]

        # Combine according to Figure 4:
        # - fused always computes the conv/instant mixture
        # - if use_inner is True (original TriDet), add internal identity and internal FFN
        fused = phi * fc_branch + psi * (convw + convkw)

        if self.use_inner:
            # preserve original TriDet internal identity shortcut
            fused = fused + out

        out = x * out_mask + self.drop_path_out(fused)

        if self.use_inner:
            # internal FFN (only when use_inner enabled)
            out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool()


class SGPWrapper(nn.Module):
    """
    Wrapper class to adapt SGPBlock for VRT architecture.
    Converts VRT's [B,D,H,W,C] format to SGPBlock's expected [B*H*W,C,D] format.
    When sgp_use_partitioned=True, accepts window-partitioned [B*nW, Wd*Wh*Ww, C] format directly.
    Implements the correct residual structure according to Figure 4.
    """

    def __init__(self, dim, kernel_size=3, k=1.5, group=1, path_pdrop=0.0,
                 act_layer=nn.GELU, downsample_type='max', init_conv_vars=1, use_inner=True, sgp_reduction=4, sgp_use_partitioned=True):
        super().__init__()
        self.dim = dim
        self.sgp_use_partitioned = sgp_use_partitioned  # Whether to expect partitioned input

        # Use parameters passed from config, fallback to defaults
        self.sgp_w = kernel_size  # Main kernel size (sgp_w from config)
        self.sgp_k = k if isinstance(k, (int, float)) else 3  # Multiplier (sgp_k from config), fallback to 3
        self.sgp_reduction = sgp_reduction  # Reduction ratio for instant-level branch

        # Calculate kernel sizes for SGP
        up_size = round((self.sgp_w + 1) * self.sgp_k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.sgp_block = SGPBlock(
            n_embd=dim,
            kernel_size=self.sgp_w,
            k=k,
            group=group,
            path_pdrop=path_pdrop,
            act_layer=act_layer,
            use_inner=use_inner,
            downsample_type=downsample_type,
            init_conv_vars=init_conv_vars,
            sgp_reduction=self.sgp_reduction
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, D, H, W, C] - VRT format (when sgp_use_partitioned=False)
               or [B*nW, Wd*Wh*Ww, C] - window-partitioned format (when sgp_use_partitioned=True)
            mask: Optional mask tensor
        Returns:
            y: Same shape as input
        """
        if self.sgp_use_partitioned:
            # Expect window-partitioned input: [B*nW, Wd*Wh*Ww, C]
            if x.dim() != 3:
                raise ValueError(f"SGPWrapper with sgp_use_partitioned=True expects window-partitioned input shape [B*nW, Wd*Wh*Ww, C]; got {x.shape}.")
            B_nW, seq_len, C = x.shape

            # Convert to channel-first for Conv1D: [B*nW, C, seq_len]
            x_conv = x.permute(0, 2, 1)  # [B*nW, C, seq_len]

            # For SGP, we ignore the attention mask from WindowAttention
            # and create a simple validity mask (all ones since window-partitioned data is always valid)
            mask = torch.ones(B_nW, 1, seq_len, device=x.device, dtype=x.dtype)

            # Apply SGPBlock
            y_conv, _ = self.sgp_block(x_conv, mask)  # [B*nW, C, seq_len]

            # Convert back to sequence format: [B*nW, seq_len, C]
            y = y_conv.permute(0, 2, 1)  # [B*nW, seq_len, C]

            return y
        else:
            # Original VRT format: [B, D, H, W, C]
            if x.dim() != 5:
                raise ValueError(f"SGPWrapper with sgp_use_partitioned=False expects input shape [B, D, H, W, C]; got {x.shape}. "
                               f"Do not pass window_partition output. For temporal SGP, use 5D input directly.")
            B, D, H, W, C = x.shape

            # Reshape to [B*H*W, D, C] - each spatial position becomes a sequence
            x_seq = x.permute(0, 2, 3, 1, 4).reshape(B*H*W, D, C)  # [B*H*W, D, C]

            # Convert to channel-first for Conv1D: [B*H*W, C, D]
            x_conv = x_seq.permute(0, 2, 1)  # [B*H*W, C, D]

            # Create mask if not provided (all ones for temporal SGP)
            if mask is None:
                mask = torch.ones(B*H*W, 1, D, device=x.device, dtype=x.dtype)

            # Apply SGPBlock
            y_conv, _ = self.sgp_block(x_conv, mask)  # [B*H*W, C, D]

            # Convert back to sequence format: [B*H*W, D, C]
            y_seq = y_conv.permute(0, 2, 1)  # [B*H*W, D, C]

            # Reshape back to spatial format: [B, D, H, W, C]
            y = y_seq.reshape(B, H, W, D, C).permute(0, 3, 1, 2, 4)  # [B, D, H, W, C]

            return y
