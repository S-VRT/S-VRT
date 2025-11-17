# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
VRT (Video Restoration Transformer) 网络模型实现
该文件包含了用于视频恢复任务的Transformer架构，包括：
- 可变形卷积模块（Deformable Convolution）
- 光流估计网络（SpyNet）
- 窗口注意力机制（Window Attention）
- 时序互自注意力（Temporal Mutual Self Attention）
- 视频恢复Transformer主网络（VRT）
"""

import os
import warnings
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from distutils.version import LooseVersion
from torch.nn.modules.utils import _pair, _single
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from einops.layers.torch import Rearrange


class ModulatedDeformConv(nn.Module):
    """可调制可变形卷积（Modulated Deformable Convolution）基础类
    
    可变形卷积允许卷积核的采样位置根据输入内容自适应调整，相比标准卷积能更好地处理
    几何形变和运动。该类定义了可变形卷积的基本参数和权重初始化。
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int or tuple): 卷积核大小
        stride (int): 步长，默认为1
        padding (int): 填充大小，默认为0
        dilation (int): 膨胀率，默认为1
        groups (int): 分组卷积的组数，默认为1
        deformable_groups (int): 可变形卷积的组数，默认为1
        bias (bool): 是否使用偏置，默认为True
    """

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
        self.kernel_size = _pair(kernel_size)  # 将kernel_size转换为元组格式
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # 为了与nn.Conv2d兼容
        self.transposed = False
        self.output_padding = _single(0)

        # 初始化卷积权重：形状为 (out_channels, in_channels//groups, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        """初始化权重和偏置
        
        使用Xavier初始化方法，根据输入通道数和卷积核大小计算标准差
        """
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)  # 计算标准差
        self.weight.data.uniform_(-stdv, stdv)  # 均匀分布初始化权重
        if self.bias is not None:
            self.bias.data.zero_()  # 偏置初始化为0

    # def forward(self, x, offset, mask):
    #     return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
    #                                  self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """可调制可变形卷积封装类，可以像普通卷积层一样使用
    
    该类在基础可变形卷积的基础上，添加了一个额外的卷积层来自动学习偏移量（offset）
    和调制掩码（mask），使得可变形卷积可以端到端训练，无需手动提供偏移量。
    
    Args:
        in_channels (int): 输入通道数，与nn.Conv2d相同
        out_channels (int): 输出通道数，与nn.Conv2d相同
        kernel_size (int or tuple[int]): 卷积核大小，与nn.Conv2d相同
        stride (int or tuple[int]): 步长，与nn.Conv2d相同
        padding (int or tuple[int]): 填充大小，与nn.Conv2d相同
        dilation (int or tuple[int]): 膨胀率，与nn.Conv2d相同
        groups (int): 分组卷积的组数，与nn.Conv2d相同
        bias (bool or str): 是否使用偏置。如果指定为'auto'，将由norm_cfg决定
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        # 用于学习偏移量和调制掩码的卷积层
        # 输出通道数 = deformable_groups * 3 * kernel_size[0] * kernel_size[1]
        # 其中：3 = 2个偏移量通道（x和y方向）+ 1个调制掩码通道
        # kernel_size[0] * kernel_size[1] 是每个采样点的偏移量
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
        """初始化权重
        
        先调用父类的初始化方法，然后将偏移量卷积层的权重和偏置初始化为0
        这样可以确保初始状态下偏移量为0，相当于标准卷积
        """
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()  # 偏移量权重初始化为0
            self.conv_offset.bias.data.zero_()  # 偏移量偏置初始化为0

    # def forward(self, x):
    #     out = self.conv_offset(x)
    #     o1, o2, mask = torch.chunk(out, 3, dim=1)
    #     offset = torch.cat((o1, o2), dim=1)
    #     mask = torch.sigmoid(mask)
    #     return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
    #                                  self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """使用截断正态分布填充张量的内部实现函数（无梯度计算）
    
    该方法基于截断均匀分布和逆CDF变换来生成截断正态分布的随机值。
    参考：https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    
    Args:
        tensor: 要填充的n维张量
        mean: 正态分布的均值
        std: 正态分布的标准差
        a: 最小值截断边界
        b: 最大值截断边界
    """
    def norm_cdf(x):
        """计算标准正态分布的累积分布函数（CDF）"""
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # 检查均值是否在合理范围内
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # 使用截断均匀分布生成值，然后使用逆CDF变换得到截断正态分布
        # 获取上下限的CDF值
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # 在[low, up]范围内均匀填充张量，然后转换到[2*low-1, 2*up-1]
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # 使用逆CDF变换得到截断标准正态分布
        tensor.erfinv_()

        # 变换到指定的均值和标准差
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # 裁剪确保值在指定范围内
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """使用截断正态分布填充输入张量
    
    从截断正态分布 :math:`\mathcal{N}(\text{mean}, \text{std}^2)` 中采样值，
    超出 :math:`[a, b]` 范围的值会被重新采样直到在边界内。
    当 :math:`a \leq \text{mean} \leq b` 时，该方法效果最好。
    
    参考：https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    
    Args:
        tensor: n维torch.Tensor
        mean: 正态分布的均值，默认为0.0
        std: 正态分布的标准差，默认为1.0
        a: 最小截断值，默认为-2.0
        b: 最大截断值，默认为2.0

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """随机深度（Stochastic Depth）Drop Path函数
    
    在残差块的主路径中按样本随机丢弃路径，用于正则化和提高模型泛化能力。
    参考：https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    
    Args:
        x: 输入张量
        drop_prob: 丢弃概率，默认为0.0
        training: 是否处于训练模式，默认为False
    
    Returns:
        处理后的张量
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # 创建与输入张量维度兼容的形状：(batch_size, 1, 1, ...)
    # 这样可以处理不同维度的张量，不仅仅是2D卷积网络
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化：0或1
    # 缩放输出以保持期望值不变
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """随机深度（Stochastic Depth）Drop Path模块
    
    在残差块的主路径中按样本随机丢弃路径的PyTorch模块。
    参考：https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    
    Args:
        drop_prob: 丢弃概率，如果为None则不进行丢弃
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            处理后的张量
        """
        return drop_path(x, self.drop_prob, self.training)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    """使用光流对图像或特征图进行变形（warping）
    
    根据光流场对输入图像或特征图进行空间变换，实现帧对齐。
    这是视频处理中的关键操作，用于将不同帧的特征对齐到参考帧。
    
    Args:
        x (Tensor): 输入张量，形状为 (n, c, h, w)
        flow (Tensor): 光流场，形状为 (n, h, w, 2)，包含x和y方向的位移
        interp_mode (str): 插值模式，可选 'nearest'、'bilinear' 或 'nearest4'，默认为 'bilinear'
        padding_mode (str): 填充模式，可选 'zeros'、'border' 或 'reflection'，默认为 'zeros'
        align_corners (bool): 是否对齐角点。PyTorch 1.3之前默认为True，之后为False。这里使用True作为默认值
        use_pad_mask (bool): 仅用于PWCNet，在通道维度上先用1填充x，然后根据填充维度的grid_sample结果生成掩码

    Returns:
        Tensor: 变形后的图像或特征图
    """
    # assert x.size()[-2:] == flow.size()[1:3] # 暂时关闭用于图像级位移的检查
    n, _, h, w = x.size()
    # 创建网格坐标
    # 注意：使用type_as可能导致TITAN RTX + PyTorch1.9.1上的非法内存访问
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device), 
                                     torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # 形状: (h, w, 2)，包含(x, y)坐标
    grid.requires_grad = False  # 网格坐标不需要梯度

    # 将光流添加到网格坐标，得到变形后的采样位置
    # 光流表示每个像素的位移，加上原始网格坐标得到目标采样位置
    vgrid = grid + flow

    # if use_pad_mask: # 用于PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # 将网格坐标缩放到[-1, 1]范围（grid_sample的要求）
    # PyTorch的grid_sample函数要求输入坐标在[-1, 1]范围内
    if interp_mode == 'nearest4':
        # 注意：此模式下对光流模型没有梯度，但结果较好
        # 使用4个最近邻点进行采样，返回4个通道的输出
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        # 对4个角点进行最近邻采样
        # 分别对左下、左上、右下、右上四个角点进行采样
        # 这样可以保留更多的空间信息，用于后续处理
        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)

        # 在通道维度上拼接4个输出
        # 输出通道数变为原来的4倍，包含4个角点的采样结果
        return torch.cat([output00, output01, output10, output11], 1)

    else:
        # 标准插值模式：双线性或最近邻
        # 将x和y坐标分别缩放到[-1, 1]范围
        # max(w-1, 1)和max(h-1, 1)用于处理单像素图像的情况
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        # 将x和y坐标堆叠为 (B, H, W, 2) 格式
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        # 使用grid_sample进行采样，根据插值模式选择双线性或最近邻
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # if use_pad_mask: # 用于PWCNet
    #     output = _flow_warp_masking(output)

    # TODO: 如果align_corners=False会怎样
    # align_corners=True表示角点像素的中心对齐，False表示角点像素的边界对齐
    return output


class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """光流引导的可变形对齐模块
    
    该模块结合光流信息和可变形卷积，实现更精确的帧对齐。
    它使用光流作为初始偏移量，然后学习残差偏移量来进一步细化对齐。
    参考：BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.
    
    Args:
        in_channels (int): 输入通道数，与nn.Conv2d相同
        out_channels (int): 输出通道数，与nn.Conv2d相同
        kernel_size (int or tuple[int]): 卷积核大小，与nn.Conv2d相同
        stride (int or tuple[int]): 步长，与nn.Conv2d相同
        padding (int or tuple[int]): 填充大小，与nn.Conv2d相同
        dilation (int or tuple[int]): 膨胀率，与nn.Conv2d相同
        groups (int): 分组卷积的组数，与nn.Conv2d相同
        bias (bool or str): 是否使用偏置。如果指定为'auto'，将由norm_cfg决定
        max_residue_magnitude (int): 偏移量残差的最大幅度，默认为10
        pa_frames (int): 并行变形的帧数，默认为2
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.pa_frames = kwargs.pop('pa_frames', 2)

        super(DCNv2PackFlowGuided, self).__init__(*args, **kwargs)

        # 构建偏移量学习网络
        # 输入包括：变形后的特征图 + 当前帧特征 + 光流
        # 输入通道数 = (1+pa_frames//2) * in_channels + pa_frames
        # 输出通道数 = 3 * 9 * deformable_groups (3=2个偏移量通道+1个掩码通道, 9=3x3卷积核)
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
        """初始化偏移量网络
        
        将最后一层卷积的权重和偏置初始化为0，确保初始状态下偏移量为0
        """
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x, x_flow_warpeds, x_current, flows):
        """前向传播
        
        使用光流引导的可变形卷积进行帧对齐。这是一个两阶段对齐过程：
        1. 粗对齐：使用光流进行初步变形对齐（在调用此函数前完成）
        2. 精细对齐：学习残差偏移量，在光流基础上进行微调，并使用调制掩码控制对齐强度
        
        可变形卷积的优势在于可以学习到比光流更精确的对齐，特别是对于遮挡、大位移等复杂情况。
        
        Args:
            x: 输入特征图，形状为 (B, C, H, W)，用于可变形卷积的输入（通常是当前帧）
            x_flow_warpeds: 通过光流变形后的特征图列表，每个元素形状为 (B, C, H, W)
                          这些是已经用光流进行粗对齐的相邻帧特征
            x_current: 当前帧的特征图，形状为 (B, C, H, W)，作为参考帧
                      用于计算偏移量和调制掩码
            flows: 光流列表，每个元素形状为 (B, 2, H, W)，包含x和y方向的位移
                  用于初始化偏移量，提供基础对齐信息
        
        Returns:
            对齐后的特征图，形状为 (B, C, H, W)
        """
        # 拼接所有输入：变形后的特征图 + 当前帧特征 + 光流
        # 这些信息用于学习残差偏移量和调制掩码
        out = self.conv_offset(torch.cat(x_flow_warpeds + [x_current] + flows, dim=1))
        # 将输出分成3部分：x方向偏移量、y方向偏移量、调制掩码
        # 偏移量用于精细调整对齐位置，调制掩码用于控制对齐强度
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # 计算偏移量：先通过tanh限制范围（[-max_residue_magnitude, max_residue_magnitude]），
        # 然后加上光流作为基础偏移量。这样偏移量 = 光流（粗对齐）+ 学习的残差（精细对齐）
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        if self.pa_frames == 2:
            # 2帧模式：将光流翻转（从(x,y)变为(y,x)以匹配卷积的坐标系统）并重复以匹配偏移量维度
            # 偏移量维度是 (B, 2*kernel_size*kernel_size, H, W)，需要将光流重复kernel_size*kernel_size次
            offset = offset + flows[0].flip(1).repeat(1, offset.size(1)//2, 1, 1)
        elif self.pa_frames == 4:
            # 4帧模式：分别处理两个偏移量（对应两帧的对齐）
            offset1, offset2 = torch.chunk(offset, 2, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2], dim=1)
        elif self.pa_frames == 6:
            # 6帧模式：分别处理三个偏移量（对应三帧的对齐）
            offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
            offset1, offset2, offset3 = torch.chunk(offset, 3, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset3 = offset3 + flows[2].flip(1).repeat(1, offset3.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2, offset3], dim=1)

        # 计算调制掩码：通过sigmoid将值限制在[0, 1]范围
        # 调制掩码用于控制每个采样点的贡献，值越大表示该点对齐质量越好
        mask = torch.sigmoid(mask)

        # 执行可变形卷积：使用学习的偏移量和调制掩码进行特征对齐
        # deform_conv2d会根据offset在每个采样点进行偏移采样，并用mask调制采样值
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, mask)


class BasicModule(nn.Module):
    """SpyNet的基础模块
    
    SpyNet光流估计网络的基本构建块，用于在单个尺度上估计光流。
    输入8通道（参考帧3通道 + 变形后的支持帧3通道 + 上采样光流2通道），
    输出2通道光流（x和y方向的位移）。
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        # 构建5层卷积网络，逐步提取特征并回归光流
        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))  # 输出2通道光流

    def forward(self, tensor_input):
        """前向传播
        
        Args:
            tensor_input: 输入张量，形状为 (B, 8, H, W)
        
        Returns:
            光流张量，形状为 (B, 2, H, W)
        """
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet光流估计网络架构
    
    SpyNet是一个多尺度的光流估计网络，采用从粗到细（coarse-to-fine）的策略。
    它通过在不同分辨率级别上迭代估计光流，逐步细化光流估计结果。
    
    Args:
        load_path (str): 预训练SpyNet模型的路径，默认为None。如果路径不存在会自动下载
        return_levels (list[int]): 返回不同尺度级别的光流，默认为[5]（仅返回最高分辨率）
    """

    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if not os.path.exists(load_path):
                import requests
                url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading SpyNet pretrained model from {url}')
                os.makedirs(os.path.dirname(load_path), exist_ok=True)
                open(load_path, 'wb').write(r.content)

            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        """预处理输入图像
        
        使用ImageNet的均值和标准差对输入图像进行归一化，这是SpyNet预训练模型的要求。
        
        Args:
            tensor_input: 输入图像张量，形状为 (B, 3, H, W)
        
        Returns:
            tensor_output: 归一化后的图像张量，形状为 (B, 3, H, W)
        """
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        """处理参考帧和支持帧，估计光流
        
        这是SpyNet的核心处理流程，采用从粗到细（coarse-to-fine）的策略：
        1. 构建多尺度图像金字塔（6个尺度级别）
        2. 从最粗尺度开始，逐步细化光流估计
        3. 在每个尺度上，使用上一尺度的光流作为初始值，估计残差光流
        4. 返回指定尺度级别的光流
        
        Args:
            ref: 参考帧，形状为 (B, 3, H_floor, W_floor)
            supp: 支持帧，形状为 (B, 3, H_floor, W_floor)
            w: 原始宽度
            h: 原始高度
            w_floor: 填充后的宽度（32的倍数）
            h_floor: 填充后的高度（32的倍数）
        
        Returns:
            flow_list: 光流列表，包含不同尺度级别的光流估计结果
        """
        flow_list = []

        # 预处理：归一化输入图像
        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        # 构建图像金字塔：通过平均池化下采样，生成5个更粗的尺度
        # 最终得到6个尺度级别：level 0（最粗）到 level 5（最细）
        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        # 初始化光流：在最粗尺度上初始化为零
        # 光流尺寸为当前尺度的一半（因为后续会放大2倍）
        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        # 从粗到细逐级估计光流
        for level in range(len(ref)):
            # 将上一尺度的光流上采样到当前尺度，并放大2倍（因为尺度变化了2倍）
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            # 如果上采样后的光流尺寸与当前尺度不匹配，进行填充
            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            # 使用光流将支持帧变形到参考帧的坐标系
            # 然后拼接：参考帧 + 变形后的支持帧 + 上采样的光流
            # 输入到basic_module估计残差光流
            flow = self.basic_module[level](torch.cat([
                ref[level],  # 参考帧
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),  # 变形后的支持帧
                upsampled_flow  # 上采样的光流
            ], 1)) + upsampled_flow  # 残差连接：加上上采样的光流

            # 如果当前尺度在返回列表中，保存该尺度的光流
            if level in self.return_levels:
                # 计算当前尺度相对于原始图像的缩放因子
                scale = 2**(5-level)  # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                # 将光流插值到目标尺寸
                flow_out = F.interpolate(input=flow, size=(h//scale, w//scale), mode='bilinear', align_corners=False)
                # 调整光流值：考虑填充后的尺寸与原始尺寸的比例
                flow_out[:, 0, :, :] *= float(w//scale) / float(w_floor//scale)  # x方向
                flow_out[:, 1, :, :] *= float(h//scale) / float(h_floor//scale)  # y方向
                flow_list.insert(0, flow_out)  # 插入到列表开头，保持从粗到细的顺序

        return flow_list

    def forward(self, ref, supp):
        """前向传播函数
        
        估计参考帧和支持帧之间的光流。首先将输入调整到32的倍数（SpyNet的要求），
        然后调用process方法进行多尺度光流估计。
        
        Args:
            ref: 参考帧，形状为 (B, 3, H, W)
            supp: 支持帧，形状为 (B, 3, H, W)
        
        Returns:
            flow: 光流张量，形状为 (B, 2, H, W) 或光流列表（如果return_levels包含多个值）
        """
        assert ref.size() == supp.size()

        # 获取原始尺寸
        h, w = ref.size(2), ref.size(3)
        # 将尺寸向上取整到32的倍数（SpyNet的要求）
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        # 将输入调整到32的倍数
        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        # 处理并估计光流
        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        # 如果只返回一个尺度，直接返回该光流；否则返回光流列表
        return flow_list[0] if len(flow_list) == 1 else flow_list


def window_partition(x, window_size):
    """将输入分割成窗口，注意力将在窗口内进行
    
    这是Swin Transformer风格的窗口分割操作，将输入特征图分割成不重叠的窗口，
    以便在每个窗口内独立计算注意力，降低计算复杂度。
    
    Args:
        x: 输入张量，形状为 (B, D, H, W, C)，其中B是批次大小，D是时间维度，H和W是空间维度，C是通道数
        window_size (tuple[int]): 窗口大小，格式为 (D_window, H_window, W_window)

    Returns:
        windows: 分割后的窗口，形状为 (B*num_windows, window_size[0]*window_size[1]*window_size[2], C)
    """
    B, D, H, W, C = x.shape
    # 将输入重新组织为窗口结构
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    # 调整维度顺序并展平，将每个窗口展平为一个序列
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """将窗口还原回原始输入形状，注意力已在窗口内完成
    
    这是window_partition的逆操作，将处理后的窗口重新组合成原始的特征图形状。
    
    Args:
        windows: 窗口张量，形状为 (B*num_windows, window_size[0]*window_size[1]*window_size[2], C)
        window_size (tuple[int]): 窗口大小，格式为 (D_window, H_window, W_window)
        B (int): 批次大小
        D (int): 时间维度大小
        H (int): 高度
        W (int): 宽度

    Returns:
        x: 还原后的特征图，形状为 (B, D, H, W, C)
    """
    # 将窗口重新组织为原始形状
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    # 调整维度顺序并展平回原始形状
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """获取窗口大小和偏移大小
    
    根据输入大小和指定的窗口大小，计算实际使用的窗口大小和偏移大小。
    如果输入尺寸小于窗口大小，则使用输入尺寸作为窗口大小，并将偏移大小设为0。
    这确保了窗口分割操作能够正确处理小尺寸输入。
    
    Args:
        x_size: 输入尺寸，格式为 (D, H, W)，其中D是时间维度，H是高度，W是宽度
        window_size: 指定的窗口大小，格式为 (D_window, H_window, W_window)
        shift_size: 指定的偏移大小，格式为 (D_shift, H_shift, W_shift)，可选
    
    Returns:
        如果shift_size为None，返回实际使用的窗口大小元组
        如果shift_size不为None，返回(实际窗口大小, 实际偏移大小)元组
    """
    # 初始化实际使用的窗口大小
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    
    # 检查每个维度：如果输入尺寸小于窗口大小，则使用输入尺寸作为窗口大小
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]  # 使用输入尺寸作为窗口大小
            if shift_size is not None:
                use_shift_size[i] = 0  # 如果窗口大小等于输入尺寸，偏移必须为0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """计算注意力掩码，用于处理窗口偏移后的注意力计算
    
    当使用shifted window attention时，窗口会被偏移，导致不同窗口之间的token需要交互。
    此函数生成掩码来标识哪些token属于同一个窗口，哪些属于不同窗口。
    使用@lru_cache装饰器缓存每个stage的结果，避免重复计算。
    
    Args:
        D (int): 时间维度大小
        H (int): 高度
        W (int): 宽度
        window_size (tuple[int]): 窗口大小，格式为 (D_window, H_window, W_window)
        shift_size (tuple[int]): 偏移大小，格式为 (D_shift, H_shift, W_shift)
        device: 计算设备（CPU或GPU）
    
    Returns:
        attn_mask: 注意力掩码，形状为 (num_windows, window_size[0]*window_size[1]*window_size[2], 
                  window_size[0]*window_size[1]*window_size[2])，相同窗口内的token为0，不同窗口为-100
    """
    # 创建图像掩码，将输入空间划分为9个区域（3x3x3=27个区域，但实际是3个时间切片x3个高度切片x3个宽度切片）
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    # 在时间、高度、宽度三个维度上分别划分三个区域：负窗口区域、偏移区域、正区域
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt  # 为每个区域分配不同的编号
                cnt += 1
    # 将掩码分割成窗口
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    # 计算窗口间的掩码：相同窗口的token相减为0，不同窗口的token相减不为0
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    # 将非零值（不同窗口）设为-100（softmax后会接近0），将零值（相同窗口）设为0
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class Upsample(nn.Sequential):
    """视频超分辨率的上采样模块
    
    使用3D卷积和PixelShuffle实现视频的上采样。支持2的幂次方（2^n）和3倍上采样。
    通过将时间维度和空间维度进行转置，使用2D PixelShuffle来处理3D视频数据。
    
    Args:
        scale (int): 上采样倍数。支持的倍数：2^n（如2, 4, 8等）和3
        num_feat (int): 中间特征的通道数
    """

    def __init__(self, scale, num_feat):
        # 需要PyTorch 1.8.1以上版本来支持5D PixelShuffle操作
        assert LooseVersion(torch.__version__) >= LooseVersion('1.8.1'), \
            'PyTorch version >= 1.8.1 to support 5D PixelShuffle.'

        class Transpose_Dim12(nn.Module):
            """转置张量的第1和第2维度
            
            用于在时间维度和批次维度之间进行转置，以便使用2D PixelShuffle处理3D视频数据。
            """

            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        m = []
        # 如果scale是2的幂次方（2^n），则通过多次2倍上采样实现
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            # 每次上采样2倍，需要log2(scale)次
            for _ in range(int(math.log(scale, 2))):
                # 3D卷积：将通道数扩展4倍（2x2=4，用于2倍上采样）
                m.append(nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
                # 转置维度以便使用2D PixelShuffle
                m.append(Transpose_Dim12())
                # PixelShuffle将通道维度重组为空间维度，实现2倍上采样
                m.append(nn.PixelShuffle(2))
                # 转置回原始维度顺序
                m.append(Transpose_Dim12())
                # 激活函数
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            # 最后一次卷积，保持通道数不变
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        # 如果scale是3，则直接进行3倍上采样
        elif scale == 3:
            # 3D卷积：将通道数扩展9倍（3x3=9，用于3倍上采样）
            m.append(nn.Conv3d(num_feat, 9 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(Transpose_Dim12())
            # PixelShuffle实现3倍上采样
            m.append(nn.PixelShuffle(3))
            m.append(Transpose_Dim12())
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class Mlp_GEGLU(nn.Module):
    """使用门控线性单元的多层感知机（GEGLU）
    
    这是Transformer中常用的前馈网络结构，使用Gated Linear Unit (GLU)变体。
    参考论文："GLU Variants Improve Transformer"。
    GEGLU通过门控机制（gate mechanism）来控制信息流，相比标准MLP有更好的性能。
    
    Args:
        in_features (int): 输入特征维度
        hidden_features (int, optional): 隐藏层特征维度，默认为in_features
        out_features (int, optional): 输出特征维度，默认为in_features
        act_layer: 激活函数类型，默认为nn.GELU
        drop (float): Dropout比率，默认为0
    
    Forward:
        Args:
            x: 输入张量，形状为 (B, D, H, W, C)
        Returns:
            x: 输出张量，形状为 (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 两个线性层用于门控机制：fc11用于门控值，fc12用于门控权重
        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # 输出投影层
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # GEGLU: 将fc11的输出经过激活函数后与fc12的输出相乘（门控机制）
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class WindowAttention(nn.Module):
    """基于窗口的多头互注意力和自注意力模块
    
    这是VRT的核心注意力机制，结合了两种注意力：
    1. 自注意力（Self Attention）：使用相对位置编码，在窗口内进行自注意力计算
    2. 互注意力（Mutual Attention）：使用正弦位置编码，实现帧间的对齐和交互
    
    互注意力机制是VRT的关键创新，它允许不同帧之间进行信息交换，这对于视频恢复任务非常重要。
    
    Args:
        dim (int): 输入通道数
        window_size (tuple[int]): 窗口大小，格式为 (时间长度, 高度, 宽度)
        num_heads (int): 注意力头数
        qkv_bias (bool, optional): 如果为True，为query、key、value添加可学习的偏置。默认为False
        qk_scale (float | None, optional): 如果设置，覆盖默认的qk缩放因子 head_dim ** -0.5
        mut_attn (bool): 如果为True，在模块中添加互注意力。默认为True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, mut_attn=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 注意力缩放因子，用于稳定训练
        self.scale = qk_scale or head_dim ** -0.5
        self.mut_attn = mut_attn

        # 自注意力部分：使用相对位置偏置
        # 相对位置偏置表：大小为 (2*Wd-1) * (2*Wh-1) * (2*Ww-1) * num_heads
        # 用于编码窗口内token之间的相对位置关系
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        # 相对位置索引，用于从偏置表中查找对应的偏置值
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        # 自注意力的QKV投影层
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 输出投影层（如果只有自注意力）
        self.proj = nn.Linear(dim, dim)

        # 互注意力部分：使用正弦位置编码
        if self.mut_attn:
            # 正弦位置编码，用于互注意力中的位置信息
            self.register_buffer("position_bias",
                                 self.get_sine_position_encoding(window_size[1:], dim // 2, normalize=True))
            # 互注意力的QKV投影层
            self.qkv_mut = nn.Linear(dim, dim * 3, bias=qkv_bias)
            # 输出投影层（自注意力+互注意力的拼接结果）
            self.proj = nn.Linear(2 * dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        # 使用截断正态分布初始化相对位置偏置表
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        """前向传播函数
        
        执行自注意力和互注意力的计算。互注意力允许不同帧之间进行信息交换，
        这对于视频恢复任务（如去模糊、超分辨率）非常重要。

        Args:
            x: 输入特征，形状为 (num_windows*B, N, C)，其中N是窗口内的token数
            mask: 注意力掩码，形状为 (num_windows, N, N) 或 None
                 掩码值为0表示允许注意力，-inf表示禁止注意力

        Returns:
            x: 输出特征，形状与输入相同
        """

        # 自注意力：在窗口内进行自注意力计算，使用相对位置编码
        B_, N, C = x.shape
        # 计算QKV：将输入投影为query、key、value
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        # 执行自注意力，使用相对位置编码
        x_out = self.attention(q, k, v, mask, (B_, N, C), relative_position_encoding=True)

        # 互注意力：实现帧间的对齐和交互
        if self.mut_attn:
            # 添加位置编码后计算QKV
            qkv = self.qkv_mut(x + self.position_bias.repeat(1, 2, 1)).reshape(B_, N, 3, self.num_heads,
                                                                               C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
            # 将QKV分成两部分，分别对应两帧
            (q1, q2), (k1, k2), (v1, v2) = torch.chunk(qkv[0], 2, dim=2), torch.chunk(qkv[1], 2, dim=2), torch.chunk(
                qkv[2], 2, dim=2)  # B_, nH, N/2, C
            # 互注意力：使用第2帧的query与第1帧的key和value计算，实现帧1到帧2的对齐
            x1_aligned = self.attention(q2, k1, v1, mask, (B_, N // 2, C), relative_position_encoding=False)
            # 互注意力：使用第1帧的query与第2帧的key和value计算，实现帧2到帧1的对齐
            x2_aligned = self.attention(q1, k2, v2, mask, (B_, N // 2, C), relative_position_encoding=False)
            # 拼接互注意力和自注意力的结果
            x_out = torch.cat([torch.cat([x1_aligned, x2_aligned], 1), x_out], 2)

        # 输出投影
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):
        """计算注意力
        
        标准的缩放点积注意力机制，可选地添加相对位置编码和掩码。

        Args:
            q: query张量，形状为 (B_, num_heads, N, head_dim)
            k: key张量，形状为 (B_, num_heads, N, head_dim)
            v: value张量，形状为 (B_, num_heads, N, head_dim)
            mask: 注意力掩码，形状为 (num_windows, N, N) 或 None
            x_shape: 输入形状元组 (B_, N, C)
            relative_position_encoding (bool): 是否使用相对位置编码

        Returns:
            x: 注意力输出，形状为 (B_, N, C)
        """
        B_, N, C = x_shape
        # 计算注意力分数：Q @ K^T，并应用缩放因子
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # 添加相对位置偏置（仅用于自注意力）
        if relative_position_encoding:
            # 从相对位置偏置表中查找对应的偏置值
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
            # 将偏置添加到注意力分数中
            attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # B_, nH, N, N

        # 应用掩码（如果提供）
        if mask is None:
            attn = self.softmax(attn)
        else:
            nW = mask.shape[0]
            # 将掩码添加到注意力分数中
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)

        # 计算加权和：注意力权重 @ value
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x

    def get_position_index(self, window_size):
        """获取窗口内每个token之间的相对位置索引
        
        用于从相对位置偏置表中查找对应的偏置值。将3D相对坐标映射到1D索引。

        Args:
            window_size (tuple[int]): 窗口大小，格式为 (D, H, W)

        Returns:
            relative_position_index: 相对位置索引矩阵，形状为 (Wd*Wh*Ww, Wd*Wh*Ww)
        """
        # 创建每个维度的坐标
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        # 生成3D网格坐标
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        # 计算相对坐标：每个token相对于其他token的位置
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        # 将相对坐标偏移到非负范围
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        # 将3D相对坐标映射到1D索引
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        return relative_position_index

    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """获取正弦位置编码
        
        生成2D空间位置的正弦位置编码，用于互注意力机制。
        这是Transformer中常用的位置编码方法，能够编码位置信息而不需要学习参数。

        Args:
            HW (tuple[int]): 空间维度大小，格式为 (H, W)
            num_pos_feats (int): 位置编码的特征维度，默认为64
            temperature (int): 温度参数，用于控制位置编码的频率，默认为10000
            normalize (bool): 是否归一化位置编码，默认为False
            scale (float, optional): 归一化时的缩放因子

        Returns:
            pos_embed: 位置编码，形状为 (1, H*W, num_pos_feats)
        """
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        # 创建位置坐标
        # 使用全1矩阵，通过累积求和生成位置索引
        not_mask = torch.ones([1, HW[0], HW[1]])
        # 在高度维度上累积求和，得到y坐标（从1到H）
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 累积求和得到y坐标
        # 在宽度维度上累积求和，得到x坐标（从1到W）
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 累积求和得到x坐标
        # 归一化位置坐标：将坐标缩放到[0, scale]范围
        if normalize:
            eps = 1e-6  # 防止除零
            # 除以最后一个值（最大值）进行归一化，然后乘以scale
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        # 计算不同频率的正弦和余弦函数
        # 生成频率序列：每个维度对应不同的频率，用于编码不同尺度的位置信息
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        # 计算温度缩放后的频率：使用温度参数控制频率的分布
        # 偶数索引和奇数索引使用相同的频率值（通过 dim_t // 2）
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        # 计算位置编码：使用正弦和余弦函数
        # 将位置坐标除以频率，得到相位值
        pos_x = x_embed[:, :, :, None] / dim_t  # (1, H, W, num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t  # (1, H, W, num_pos_feats)
        # 对偶数维度使用sin，对奇数维度使用cos
        # 这是Transformer中常用的位置编码方式，能够编码相对位置关系
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 拼接y和x的位置编码：先y后x，保持空间维度的一致性
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (1, num_pos_feats, H, W)

        # 展平空间维度并转置，得到 (1, H*W, num_pos_feats) 格式
        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()


class TMSA(nn.Module):
    """时间互相关自注意力模块（Temporal Mutual Self Attention, TMSA）
    
    这是VRT的核心模块，结合了窗口注意力和前馈网络。
    它使用shifted window attention机制，通过窗口偏移来增加感受野。
    模块包含两个部分：
    1. 注意力部分：使用WindowAttention进行自注意力和互注意力计算
    2. 前馈网络部分：使用GEGLU MLP进行特征变换
    
    Args:
        dim (int): 输入通道数
        input_resolution (tuple[int]): 输入分辨率，格式为 (D, H, W)
        num_heads (int): 注意力头数
        window_size (tuple[int]): 窗口大小，格式为 (D_window, H_window, W_window)
        shift_size (tuple[int]): 窗口偏移大小，用于shifted window attention
        mut_attn (bool): 如果为True，使用互注意力和自注意力。默认为True
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率，默认为2.0
        qkv_bias (bool, optional): 如果为True，为query、key、value添加可学习的偏置。默认为True
        qk_scale (float | None, optional): 如果设置，覆盖默认的qk缩放因子 head_dim ** -0.5
        drop_path (float, optional): 随机深度比率。默认为0.0
        act_layer (nn.Module, optional): 激活函数层。默认为nn.GELU
        norm_layer (nn.Module, optional): 归一化层。默认为nn.LayerNorm
        use_checkpoint_attn (bool): 如果为True，对注意力模块使用torch.checkpoint以节省内存。默认为False
        use_checkpoint_ffn (bool): 如果为True，对前馈网络模块使用torch.checkpoint以节省内存。默认为False
    """

    def __init__(self,
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
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        # 注意力部分的归一化层
        self.norm1 = norm_layer(dim)
        # 窗口注意力模块：包含自注意力和互注意力
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, mut_attn=mut_attn)
        # 随机深度：用于正则化和稳定训练
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 前馈网络部分的归一化层
        self.norm2 = norm_layer(dim)
        # 前馈网络：使用GEGLU激活函数的多层感知机
        self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        """前向传播的第一部分：注意力计算
        
        这是TMSA模块的核心部分，实现了窗口注意力机制，包括：
        1. 窗口分割：将特征图分割成不重叠的窗口，在每个窗口内独立计算注意力
        2. Shifted Window Attention：通过循环偏移增加窗口间的交互，扩大感受野
        3. 自注意力和互注意力：在窗口内进行帧内和帧间的信息交互
        4. 窗口合并：将窗口重新组合成完整的特征图
        
        支持shifted window attention机制以增加感受野，这是Swin Transformer的关键创新。

        Args:
            x: 输入特征，形状为 (B, D, H, W, C)，其中D是时间维度，H和W是空间维度
            mask_matrix: 注意力掩码矩阵，用于shifted window attention
                        当使用shifted window时，需要掩码来标识哪些token属于同一窗口

        Returns:
            x: 注意力输出，形状为 (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape
        # 根据输入大小调整窗口大小和偏移大小
        # 如果输入尺寸小于窗口大小，则使用输入尺寸作为窗口大小，偏移设为0
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        # 归一化：在注意力计算前进行层归一化
        x = self.norm1(x)

        # 将特征图填充到窗口大小的倍数，确保可以完整分割成窗口
        # 填充格式：(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]  # 时间维度填充
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]    # 高度填充
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]    # 宽度填充
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        # 循环偏移：实现shifted window attention
        # 通过循环偏移，使得相邻层的窗口边界不同，从而增加窗口间的交互
        if any(i > 0 for i in shift_size):
            # 将特征图循环偏移，使得窗口边界发生变化
            # 负偏移表示向左/上/前偏移，这样下一层的窗口会与当前层不同
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix  # 使用掩码来标识哪些token属于同一窗口
        else:
            shifted_x = x
            attn_mask = None  # 不使用shifted window时不需要掩码

        # 分割窗口：将特征图分割成不重叠的窗口
        # 输出形状：(B*nW, Wd*Wh*Ww, C)，其中nW是窗口数量，Wd*Wh*Ww是每个窗口的token数
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # 注意力计算：在窗口内进行自注意力和互注意力
        # 自注意力：帧内信息交互；互注意力：帧间信息交互和对齐
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # 合并窗口：将窗口重新组合成特征图
        # 先将窗口reshape为 (B*nW, Wd, Wh, Ww, C)，然后合并为完整特征图
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # 反向循环偏移：将特征图偏移回原始位置
        # 使用正向偏移来抵消之前的负偏移
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        # 移除填充：恢复到原始尺寸
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        # 应用随机深度（Stochastic Depth）：以一定概率跳过当前层，用于正则化和加速训练
        x = self.drop_path(x)

        return x

    def forward_part2(self, x):
        """前向传播的第二部分：前馈网络
        
        Args:
            x: 输入特征，形状为 (B, D, H, W, C)

        Returns:
            x: 前馈网络输出，形状为 (B, D, H, W, C)
        """
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """前向传播函数
        
        执行完整的TMSA模块计算，包括注意力部分和前馈网络部分。
        使用残差连接将输入与输出相加。

        Args:
            x: 输入特征，形状为 (B, D, H, W, C)
            mask_matrix: 循环偏移的注意力掩码

        Returns:
            x: 输出特征，形状为 (B, D, H, W, C)
        """

        # 注意力部分：使用残差连接
        if self.use_checkpoint_attn:
            # 使用checkpoint以节省内存（以计算时间为代价）
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        # 前馈网络部分：使用残差连接
        if self.use_checkpoint_ffn:
            # 使用checkpoint以节省内存
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class TMSAG(nn.Module):
    """时间互相关自注意力组（Temporal Mutual Self Attention Group, TMSAG）
    
    这是多个TMSA模块的组合，通过堆叠多个TMSA模块来增加网络的深度。
    交替使用普通窗口注意力和shifted window attention，以增加感受野。

    Args:
        dim (int): 特征通道数
        input_resolution (tuple[int]): 输入分辨率，格式为 (D, H, W)
        depth (int): 该阶段的深度（TMSA模块的数量）
        num_heads (int): 注意力头数
        window_size (tuple[int]): 局部窗口大小。默认为 (6, 8, 8)
        shift_size (tuple[int]): 互注意力和自注意力的偏移大小。如果为None，则使用window_size的一半。默认为None
        mut_attn (bool): 如果为True，使用互注意力和自注意力。默认为True
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率。默认为2.0
        qkv_bias (bool, optional): 如果为True，为query、key、value添加可学习的偏置。默认为True
        qk_scale (float | None, optional): 如果设置，覆盖默认的qk缩放因子 head_dim ** -0.5
        drop_path (float | tuple[float], optional): 随机深度比率。可以是单个值或每个模块的值列表。默认为0.0
        norm_layer (nn.Module, optional): 归一化层。默认为nn.LayerNorm
        use_checkpoint_attn (bool): 如果为True，对注意力模块使用torch.checkpoint以节省内存。默认为False
        use_checkpoint_ffn (bool): 如果为True，对前馈网络模块使用torch.checkpoint以节省内存。默认为False
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=[6, 8, 8],
                 shift_size=None,
                 mut_attn=True,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        # 如果未指定shift_size，则使用window_size的一半
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size

        # 构建TMSA模块列表
        # 偶数索引的模块使用普通窗口注意力（shift_size=[0,0,0]）
        # 奇数索引的模块使用shifted window attention（shift_size=self.shift_size）
        # 这种交替设计可以增加感受野，同时保持计算效率
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
                use_checkpoint_ffn=use_checkpoint_ffn
            )
            for i in range(depth)])

    def forward(self, x):
        """前向传播函数
        
        顺序执行所有TMSA模块，实现深度特征提取。

        Args:
            x: 输入特征，形状为 (B, C, D, H, W)

        Returns:
            x: 输出特征，形状为 (B, C, D, H, W)
        """
        # 计算注意力掩码（用于shifted window attention）
        B, C, D, H, W = x.shape
        # 根据输入大小调整窗口大小和偏移大小
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        # 将维度从 (B, C, D, H, W) 转换为 (B, D, H, W, C) 以便处理
        x = rearrange(x, 'b c d h w -> b d h w c')
        # 计算填充后的尺寸（向上取整到窗口大小的倍数）
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        # 计算注意力掩码（如果使用shifted window attention）
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        # 顺序执行所有TMSA模块
        for blk in self.blocks:
            x = blk(x, attn_mask)

        # 恢复原始尺寸并转换回 (B, C, D, H, W) 格式
        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x


class RTMSA(nn.Module):
    """残差时间互相关自注意力（Residual Temporal Mutual Self Attention, RTMSA）
    
    这是TMSAG的残差版本，仅在第8阶段使用。
    与TMSAG不同的是，RTMSA不使用互注意力（mut_attn=False），只使用自注意力。
    通过额外的线性层和残差连接来增强特征表示。

    Args:
        dim (int): 输入通道数
        input_resolution (tuple[int]): 输入分辨率，格式为 (D, H, W)
        depth (int): 块的数量（TMSA模块的数量）
        num_heads (int): 注意力头数
        window_size (int): 局部窗口大小
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率
        qkv_bias (bool, optional): 如果为True，为query、key、value添加可学习的偏置。默认为True
        qk_scale (float | None, optional): 如果设置，覆盖默认的qk缩放因子 head_dim ** -0.5
        drop_path (float | tuple[float], optional): 随机深度比率。默认为0.0
        norm_layer (nn.Module, optional): 归一化层。默认为nn.LayerNorm
        use_checkpoint_attn (bool): 如果为True，对注意力模块使用torch.checkpoint以节省内存。默认为False
        use_checkpoint_ffn (bool): 如果为True，对前馈网络模块使用torch.checkpoint以节省内存。默认为False
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=None
                 ):
        super(RTMSA, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        # TMSAG模块：不使用互注意力，只使用自注意力
        self.residual_group = TMSAG(dim=dim,
                                    input_resolution=input_resolution,
                                    depth=depth,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    mut_attn=False,  # 不使用互注意力
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop_path=drop_path,
                                    norm_layer=norm_layer,
                                    use_checkpoint_attn=use_checkpoint_attn,
                                    use_checkpoint_ffn=use_checkpoint_ffn
                                    )

        # 线性投影层：用于特征变换
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        """前向传播函数
        
        使用残差连接将输入与处理后的特征相加。

        Args:
            x: 输入特征，形状为 (B, C, D, H, W)

        Returns:
            x: 输出特征，形状为 (B, C, D, H, W)
        """
        # 转置维度以便进行线性变换，然后转置回原始维度
        # 使用残差连接：x + linear(TMSAG(x))
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


class Stage(nn.Module):
    """阶段模块：残差时间互相关自注意力组和并行变形
    
    这是VRT网络的一个完整阶段，包含：
    1. 并行变形对齐（Parallel Warping）：使用光流引导的可变形卷积进行帧对齐
    2. TMSAG模块：进行时间互相关自注意力计算
    3. 可选的尺寸调整（reshape）：下采样、上采样或保持尺寸不变
    
    该模块结合了光流对齐和Transformer注意力机制，实现高效的视频恢复。

    Args:
        in_dim (int): 输入通道数
        dim (int): 通道数
        input_resolution (tuple[int]): 输入分辨率，格式为 (D, H, W)
        depth (int): 块的数量（TMSA模块的数量）
        num_heads (int): 注意力头数
        mul_attn_ratio (float): 使用互注意力的层比例。默认为0.75
        window_size (int): 局部窗口大小
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率
        qkv_bias (bool, optional): 如果为True，为query、key、value添加可学习的偏置。默认为True
        qk_scale (float | None, optional): 如果设置，覆盖默认的qk缩放因子 head_dim ** -0.5
        drop_path (float | tuple[float], optional): 随机深度比率。默认为0.0
        norm_layer (nn.Module, optional): 归一化层。默认为nn.LayerNorm
        pa_frames (float): 变形的帧数。默认为2
        deformable_groups (float): 可变形卷积的组数。默认为16
        reshape (str): 尺寸调整方式：下采样（down）、上采样（up）或保持尺寸（none）
        max_residue_magnitude (float): 光流残差的最大幅度
        use_checkpoint_attn (bool): 如果为True，对注意力模块使用torch.checkpoint以节省内存。默认为False
        use_checkpoint_ffn (bool): 如果为True，对前馈网络模块使用torch.checkpoint以节省内存。默认为False
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
                 deformable_groups=16,
                 reshape=None,
                 max_residue_magnitude=10,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super(Stage, self).__init__()
        self.pa_frames = pa_frames

        # 尺寸调整模块：根据reshape参数进行下采样、上采样或保持尺寸不变
        if reshape == 'none':
            # 保持尺寸不变：只进行维度重排和归一化
            self.reshape = nn.Sequential(Rearrange('n c d h w -> n d h w c'),
                                         nn.LayerNorm(dim),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'down':
            # 下采样：将空间尺寸缩小2倍，通道数增加4倍，然后投影到目标维度
            # 使用pixel shuffle的逆操作：将2x2的空间区域合并，通道数增加4倍
            self.reshape = nn.Sequential(Rearrange('n c d (h neih) (w neiw) -> n d h w (neiw neih c)', neih=2, neiw=2),
                                         nn.LayerNorm(4 * in_dim), nn.Linear(4 * in_dim, dim),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'up':
            # 上采样：将通道数减少4倍，空间尺寸扩大2倍
            # 使用pixel shuffle：将通道维度展开为2x2的空间区域
            self.reshape = nn.Sequential(Rearrange('n (neiw neih c) d h w -> n d (h neih) (w neiw) c', neih=2, neiw=2),
                                         nn.LayerNorm(in_dim // 4), nn.Linear(in_dim // 4, dim),
                                         Rearrange('n d h w c -> n c d h w'))

        # 第一阶段：使用互注意力和自注意力的TMSAG模块
        # 时间窗口大小为2，使用互注意力机制（mut_attn=True）
        self.residual_group1 = TMSAG(dim=dim,
                                     input_resolution=input_resolution,
                                     depth=int(depth * mul_attn_ratio),  # 使用指定比例的层数
                                     num_heads=num_heads,
                                     window_size=(2, window_size[1], window_size[2]),  # 时间维度窗口大小为2
                                     mut_attn=True,  # 启用互注意力
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     use_checkpoint_attn=use_checkpoint_attn,
                                     use_checkpoint_ffn=use_checkpoint_ffn
                                     )
        self.linear1 = nn.Linear(dim, dim)  # 线性投影层

        # 第二阶段：仅使用自注意力的TMSAG模块
        # 使用完整的窗口大小，不使用互注意力（mut_attn=False）
        self.residual_group2 = TMSAG(dim=dim,
                                     input_resolution=input_resolution,
                                     depth=depth - int(depth * mul_attn_ratio),  # 剩余层数
                                     num_heads=num_heads,
                                     window_size=window_size,  # 使用完整的窗口大小
                                     mut_attn=False,  # 不使用互注意力
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     use_checkpoint_attn=True,  # 强制使用checkpoint以节省内存
                                     use_checkpoint_ffn=use_checkpoint_ffn
                                     )
        self.linear2 = nn.Linear(dim, dim)  # 线性投影层

        # 并行变形对齐模块：使用光流引导的可变形卷积进行帧对齐
        if self.pa_frames:
            # 可变形卷积模块：用于对齐相邻帧的特征
            self.pa_deform = DCNv2PackFlowGuided(dim, dim, 3, padding=1, deformable_groups=deformable_groups,
                                                 max_residue_magnitude=max_residue_magnitude, pa_frames=pa_frames)
            # 特征融合模块：融合原始特征、后向对齐特征和前向对齐特征
            # 输入维度为 dim * (1 + 2) = dim * 3（原始 + 后向 + 前向）
            self.pa_fuse = Mlp_GEGLU(dim * (1 + 2), dim * (1 + 2), dim)

    def forward(self, x, flows_backward, flows_forward):
        """前向传播函数
        
        执行以下步骤：
        1. 尺寸调整（reshape）
        2. 第一阶段TMSAG处理（互注意力+自注意力）
        3. 第二阶段TMSAG处理（仅自注意力）
        4. 并行变形对齐（如果启用）

        Args:
            x: 输入特征，形状为 (B, C, D, H, W)
            flows_backward: 后向光流列表，用于帧对齐
            flows_forward: 前向光流列表，用于帧对齐

        Returns:
            x: 输出特征，形状为 (B, C, D, H, W)
        """
        # 步骤1：尺寸调整（下采样、上采样或保持尺寸）
        x = self.reshape(x)
        
        # 步骤2：第一阶段TMSAG处理（使用互注意力和自注意力）
        # 转置维度以便进行线性变换，然后转置回原始维度，并使用残差连接
        x = self.linear1(self.residual_group1(x).transpose(1, 4)).transpose(1, 4) + x
        
        # 步骤3：第二阶段TMSAG处理（仅使用自注意力）
        # 同样使用残差连接
        x = self.linear2(self.residual_group2(x).transpose(1, 4)).transpose(1, 4) + x

        # 步骤4：并行变形对齐（如果启用）
        if self.pa_frames:
            # 将维度从 (B, C, D, H, W) 转换为 (B, D, C, H, W) 以便处理时间维度
            x = x.transpose(1, 2)
            # 根据pa_frames参数调用对应的对齐函数（2/4/6帧）
            x_backward, x_forward = getattr(self, f'get_aligned_feature_{self.pa_frames}frames')(x, flows_backward, flows_forward)
            # 融合原始特征、后向对齐特征和前向对齐特征
            # 拼接后维度为 (B, D, 3*C, H, W)，然后融合为 (B, D, C, H, W)
            x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)
            # 转换回 (B, C, D, H, W) 格式
            x = x.transpose(1, 2)

        return x

    def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
        """并行特征变形对齐（2帧模式）
        
        对每一帧，使用光流将相邻帧对齐到当前帧，然后使用可变形卷积进行精细对齐。
        
        Args:
            x: 输入特征，形状为 (B, D, C, H, W)
            flows_backward: 后向光流列表，flows_backward[0]形状为 (B, D-1, 2, H, W)
            flows_forward: 前向光流列表，flows_forward[0]形状为 (B, D-1, 2, H, W)

        Returns:
            [x_backward, x_forward]: 对齐后的后向和前向特征，每个形状为 (B, D, C, H, W)
        """
        # 后向对齐：从最后一帧向前处理
        n = x.size(1)  # 帧数
        x_backward = [torch.zeros_like(x[:, -1, ...])]  # 最后一帧的对齐特征初始化为零
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]  # 当前帧特征
            flow = flows_backward[0][:, i - 1, ...]  # 从帧i+1到帧i的光流
            # 使用光流将帧i+1变形对齐到帧i的位置
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            # 使用可变形卷积进行精细对齐，输入为当前帧、变形后的相邻帧、参考帧和光流
            x_backward.insert(0, self.pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

        # 前向对齐：从第一帧向后处理
        x_forward = [torch.zeros_like(x[:, 0, ...])]  # 第一帧的对齐特征初始化为零
        for i in range(0, n - 1):
            x_i = x[:, i, ...]  # 当前帧特征
            flow = flows_forward[0][:, i, ...]  # 从帧i-1到帧i的光流
            # 使用光流将帧i-1变形对齐到帧i的位置
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            # 使用可变形卷积进行精细对齐
            x_forward.append(self.pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
        """并行特征变形对齐（4帧模式）
        
        对每一帧，使用光流将相邻2帧（i+1, i+2或i-1, i-2）对齐到当前帧，然后使用可变形卷积进行精细对齐。
        相比2帧模式，4帧模式可以融合更多相邻帧的信息，提高对齐质量。
        
        Args:
            x: 输入特征，形状为 (B, D, C, H, W)
            flows_backward: 后向光流列表，flows_backward[0]为1帧间隔光流，flows_backward[1]为2帧间隔光流
            flows_forward: 前向光流列表，flows_forward[0]为1帧间隔光流，flows_forward[1]为2帧间隔光流

        Returns:
            [x_backward, x_forward]: 对齐后的后向和前向特征，每个形状为 (B, D, C, H, W)
        """
        # 后向对齐：从最后一帧向前处理，对齐帧i+1和i+2到帧i
        n = x.size(1)  # 帧数
        x_backward = [torch.zeros_like(x[:, -1, ...])]  # 最后一帧的对齐特征初始化为零
        for i in range(n, 1, -1):  # 从最后一帧向前遍历
            x_i = x[:, i - 1, ...]  # 当前帧i-1的特征（对应索引i-1）
            flow1 = flows_backward[0][:, i - 2, ...]  # 从帧i到帧i-1的1帧间隔光流
            # 处理边界情况：如果i是最后一帧，则i+2帧不存在，使用零填充
            if i == n:
                x_ii = torch.zeros_like(x[:, n - 2, ...])  # 帧i+2不存在，使用零填充
                flow2 = torch.zeros_like(flows_backward[1][:, n - 3, ...])  # 2帧间隔光流不存在
            else:
                x_ii = x[:, i, ...]  # 帧i的特征（对应索引i）
                flow2 = flows_backward[1][:, i - 2, ...]  # 从帧i+1到帧i-1的2帧间隔光流

            # 使用光流将帧i和帧i+1变形对齐到帧i-1的位置
            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # 帧i对齐到帧i-1
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # 帧i+1对齐到帧i-1
            # 使用可变形卷积进行精细对齐，融合两帧的信息
            x_backward.insert(0,
                self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]))

        # 前向对齐：从第一帧向后处理，对齐帧i-1和i-2到帧i
        x_forward = [torch.zeros_like(x[:, 0, ...])]  # 第一帧的对齐特征初始化为零
        for i in range(-1, n - 2):  # 从第一帧向后遍历
            x_i = x[:, i + 1, ...]  # 当前帧i+1的特征（对应索引i+1）
            flow1 = flows_forward[0][:, i + 1, ...]  # 从帧i到帧i+1的1帧间隔光流
            # 处理边界情况：如果i是-1（第一帧之前），则i-2帧不存在，使用零填充
            if i == -1:
                x_ii = torch.zeros_like(x[:, 1, ...])  # 帧i-2不存在，使用零填充
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])  # 2帧间隔光流不存在
            else:
                x_ii = x[:, i, ...]  # 帧i的特征（对应索引i）
                flow2 = flows_forward[1][:, i, ...]  # 从帧i-1到帧i+1的2帧间隔光流

            # 使用光流将帧i和帧i-1变形对齐到帧i+1的位置
            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # 帧i对齐到帧i+1
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # 帧i-1对齐到帧i+1
            # 使用可变形卷积进行精细对齐，融合两帧的信息
            x_forward.append(
                self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_6frames(self, x, flows_backward, flows_forward):
        """并行特征变形对齐（6帧模式）
        
        对每一帧，使用光流将相邻3帧（i+1, i+2, i+3或i-1, i-2, i-3）对齐到当前帧，然后使用可变形卷积进行精细对齐。
        相比2帧和4帧模式，6帧模式可以融合最多相邻帧的信息，提供最丰富的上下文信息，但计算成本也最高。
        
        Args:
            x: 输入特征，形状为 (B, D, C, H, W)
            flows_backward: 后向光流列表，flows_backward[0/1/2]分别为1/2/3帧间隔的光流
            flows_forward: 前向光流列表，flows_forward[0/1/2]分别为1/2/3帧间隔的光流

        Returns:
            [x_backward, x_forward]: 对齐后的后向和前向特征，每个形状为 (B, D, C, H, W)
        """
        # 后向对齐：从最后一帧向前处理，对齐帧i+1、i+2和i+3到帧i
        n = x.size(1)  # 帧数
        x_backward = [torch.zeros_like(x[:, -1, ...])]  # 最后一帧的对齐特征初始化为零
        for i in range(n + 1, 2, -1):  # 从最后一帧向前遍历
            x_i = x[:, i - 2, ...]  # 当前帧i-2的特征
            flow1 = flows_backward[0][:, i - 3, ...]  # 从帧i-1到帧i-2的1帧间隔光流
            # 处理边界情况：根据i的位置决定哪些帧存在
            if i == n + 1:  # 超出最后一帧，所有后续帧都不存在
                x_ii = torch.zeros_like(x[:, -1, ...])  # 帧i+1不存在
                flow2 = torch.zeros_like(flows_backward[1][:, -1, ...])  # 2帧间隔光流不存在
                x_iii = torch.zeros_like(x[:, -1, ...])  # 帧i+2不存在
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])  # 3帧间隔光流不存在
            elif i == n:  # 最后一帧，只有i+1帧存在
                x_ii = x[:, i - 1, ...]  # 帧i-1的特征
                flow2 = flows_backward[1][:, i - 3, ...]  # 从帧i到帧i-2的2帧间隔光流
                x_iii = torch.zeros_like(x[:, -1, ...])  # 帧i+2不存在
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])  # 3帧间隔光流不存在
            else:  # 正常情况，所有帧都存在
                x_ii = x[:, i - 1, ...]  # 帧i-1的特征
                flow2 = flows_backward[1][:, i - 3, ...]  # 从帧i到帧i-2的2帧间隔光流
                x_iii = x[:, i, ...]  # 帧i的特征
                flow3 = flows_backward[2][:, i - 3, ...]  # 从帧i+1到帧i-2的3帧间隔光流

            # 使用光流将三帧变形对齐到帧i-2的位置
            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # 帧i-1对齐到帧i-2
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # 帧i对齐到帧i-2
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # 帧i+1对齐到帧i-2
            # 使用可变形卷积进行精细对齐，融合三帧的信息
            x_backward.insert(0,
                              self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                             x[:, i - 3, ...], [flow1, flow2, flow3]))

        # 前向对齐：从第一帧向后处理，对齐帧i-1、i-2和i-3到帧i
        x_forward = [torch.zeros_like(x[:, 0, ...])]  # 第一帧的对齐特征初始化为零
        for i in range(0, n - 1):  # 从第一帧向后遍历
            x_i = x[:, i, ...]  # 当前帧i的特征
            flow1 = flows_forward[0][:, i, ...]  # 从帧i-1到帧i的1帧间隔光流
            # 处理边界情况：根据i的位置决定哪些帧存在
            if i == 0:  # 第一帧，所有前序帧都不存在
                x_ii = torch.zeros_like(x[:, 0, ...])  # 帧i-1不存在
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])  # 2帧间隔光流不存在
                x_iii = torch.zeros_like(x[:, 0, ...])  # 帧i-2不存在
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])  # 3帧间隔光流不存在
            elif i == 1:  # 第二帧，只有i-1帧存在
                x_ii = x[:, i - 1, ...]  # 帧i-1的特征
                flow2 = flows_forward[1][:, i - 1, ...]  # 从帧i-2到帧i的2帧间隔光流
                x_iii = torch.zeros_like(x[:, 0, ...])  # 帧i-2不存在
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])  # 3帧间隔光流不存在
            else:  # 正常情况，所有帧都存在
                x_ii = x[:, i - 1, ...]  # 帧i-1的特征
                flow2 = flows_forward[1][:, i - 1, ...]  # 从帧i-2到帧i的2帧间隔光流
                x_iii = x[:, i - 2, ...]  # 帧i-2的特征
                flow3 = flows_forward[2][:, i - 2, ...]  # 从帧i-3到帧i的3帧间隔光流

            # 使用光流将三帧变形对齐到帧i的位置
            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # 帧i-1对齐到帧i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # 帧i-2对齐到帧i
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # 帧i-3对齐到帧i
            # 使用可变形卷积进行精细对齐，融合三帧的信息
            x_forward.append(self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                            x[:, i + 1, ...], [flow1, flow2, flow3]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]


class VRT(nn.Module):
    """视频恢复Transformer（Video Restoration Transformer, VRT）
    
    这是一个基于Transformer架构的视频恢复网络，用于视频超分辨率、去模糊、去噪等任务。
    网络采用U型结构，包含编码器（下采样）和解码器（上采样）阶段，以及一个重建阶段。
    
    主要特点：
    1. 使用时间互相关自注意力（TMSA）机制捕获时间依赖关系
    2. 使用并行变形对齐（Parallel Warping）进行帧对齐
    3. 使用光流引导的可变形卷积进行精细对齐
    4. 采用U型结构实现多尺度特征提取
    
    论文：VRT: A Video Restoration Transformer
    URL: https://arxiv.org/pdf/2201.00000

    Args:
        upscale (int): 上采样因子。对于视频去模糊等任务设置为1。默认为4
        in_chans (int): 输入图像通道数。默认为3（RGB）
        out_chans (int): 输出图像通道数。默认为3（RGB）
        img_size (int | tuple(int)): 输入图像尺寸，格式为 [D, H, W]。默认为 [6, 64, 64]
        window_size (int | tuple(int)): 窗口大小，格式为 [D, H, W]。默认为 (6,8,8)
        depths (list[int]): 每个Transformer阶段的深度（层数）
        indep_reconsts (list[int]): 独立提取不同帧特征的层索引
        embed_dims (list[int]): 每个阶段的嵌入维度（通道数）
        num_heads (list[int]): 每个阶段的注意力头数
        mul_attn_ratio (float): 使用互注意力的层比例。默认为0.75
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率。默认为2
        qkv_bias (bool): 如果为True，为query、key、value添加可学习的偏置。默认为True
        qk_scale (float): 如果设置，覆盖默认的qk缩放因子 head_dim ** -0.5
        drop_path_rate (float): 随机深度比率。默认为0.2
        norm_layer (obj): 归一化层。默认为nn.LayerNorm
        spynet_path (str): 预训练的SpyNet模型路径
        pa_frames (float): 变形的帧数（2/4/6）。默认为2
        deformable_groups (float): 可变形卷积的组数。默认为16
        recal_all_flows (bool): 如果为True，从(t,t+1)光流推导(t,t+2)和(t,t+3)光流。默认为False
        nonblind_denoising (bool): 如果为True，进行非盲去噪实验。默认为False
        use_checkpoint_attn (bool): 如果为True，对注意力模块使用torch.checkpoint以节省内存。默认为False
        use_checkpoint_ffn (bool): 如果为True，对前馈网络模块使用torch.checkpoint以节省内存。默认为False
        no_checkpoint_attn_blocks (list[int]): 不使用torch.checkpoint的注意力模块层索引
        no_checkpoint_ffn_blocks (list[int]): 不使用torch.checkpoint的前馈网络模块层索引
    """

    def __init__(self,
                 upscale=4,
                 in_chans=3,
                 out_chans=3,
                 img_size=[6, 64, 64],
                 window_size=[6, 8, 8],
                 depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
                 indep_reconsts=[11, 12],
                 embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
                 num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                 mul_attn_ratio=0.75,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 spynet_path=None,
                 pa_frames=2,
                 deformable_groups=16,
                 recal_all_flows=False,
                 nonblind_denoising=False,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 no_checkpoint_attn_blocks=[],
                 no_checkpoint_ffn_blocks=[],
                 ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.recal_all_flows = recal_all_flows
        self.nonblind_denoising = nonblind_denoising

        # 第一个卷积层：将输入投影到嵌入空间
        # 如果使用并行变形对齐，输入包括原始帧、后向对齐帧和前向对齐帧（共1+2*4=9倍通道）
        # 对于非盲去噪，还需要额外的噪声水平图通道
        if self.pa_frames:
            if self.nonblind_denoising:
                conv_first_in_chans = in_chans*(1+2*4)+1  # 原始 + 后向4帧 + 前向4帧 + 噪声图
            else:
                conv_first_in_chans = in_chans*(1+2*4)  # 原始 + 后向4帧 + 前向4帧
        else:
            conv_first_in_chans = in_chans  # 仅原始输入
        self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # 主网络体
        # 光流估计网络：如果使用并行变形对齐，需要估计光流
        if self.pa_frames:
            self.spynet = SpyNet(spynet_path, [2, 3, 4, 5])  # 多尺度光流估计
        
        # 随机深度衰减规则：为每一层生成不同的drop_path率
        # 使用线性插值从0到drop_path_rate，为所有层分配不同的drop_path率
        # 这样可以实现渐进式的随机深度，早期层drop_path率较小，后期层较大
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # U型结构的配置
        # reshapes: 每个阶段的尺寸调整方式
        # - 'none': 保持尺寸不变（阶段1；阶段8 的设置在后续独立定义）
        # - 'down': 下采样（阶段2-4，编码器路径）
        # - 'up': 上采样（阶段6-7，解码器路径）
        reshapes = ['none', 'down', 'down', 'down', 'up', 'up', 'up']  # 前3个阶段下采样，后3个阶段上采样
        # scales: 每个阶段相对于输入的空间缩放因子
        # 编码器路径：1 -> 2 -> 4 -> 8（逐步下采样）
        # 解码器路径：8 -> 4 -> 2 -> 1（逐步上采样）
        scales = [1, 2, 4, 8, 4, 2, 1]  # 每个阶段的空间缩放因子
        
        # 配置哪些层使用checkpoint以节省内存
        use_checkpoint_attns = [False if i in no_checkpoint_attn_blocks else use_checkpoint_attn for i in
                                range(len(depths))]
        use_checkpoint_ffns = [False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn for i in
                               range(len(depths))]

        # 阶段1-7：U型结构的编码器和解码器
        # 阶段1-4：编码器（下采样），阶段5-7：解码器（上采样）
        for i in range(7):
            setattr(self, f'stage{i + 1}',
                    Stage(
                        in_dim=embed_dims[i - 1] if i > 0 else embed_dims[0],  # 第一个阶段使用embed_dims[0]
                        dim=embed_dims[i],
                        input_resolution=(img_size[0], img_size[1] // scales[i], img_size[2] // scales[i]),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        mul_attn_ratio=mul_attn_ratio,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],  # 该阶段的drop_path率
                        norm_layer=norm_layer,
                        pa_frames=pa_frames,
                        deformable_groups=deformable_groups,
                        reshape=reshapes[i],  # 下采样、上采样或保持尺寸
                        max_residue_magnitude=10 / scales[i],  # 根据尺度调整最大残差幅度
                        use_checkpoint_attn=use_checkpoint_attns[i],
                        use_checkpoint_ffn=use_checkpoint_ffns[i],
                        )
                    )

        # 阶段8：重建阶段，使用RTMSA模块（残差时间互相关自注意力）
        # RTMSA不使用互注意力，只使用自注意力，用于最终的特征精炼
        # 首先将维度从stage7的输出投影到stage8的输入维度
        self.stage8 = nn.ModuleList(
            [nn.Sequential(
                Rearrange('n c d h w ->  n d h w c'),  # 转换为 (B, D, H, W, C) 以便进行LayerNorm
                nn.LayerNorm(embed_dims[6]),  # 归一化
                nn.Linear(embed_dims[6], embed_dims[7]),  # 维度投影：从stage7的维度投影到stage8的维度
                Rearrange('n d h w c -> n c d h w')  # 转换回 (B, C, D, H, W)
            )]
        )
        # 添加RTMSA模块：用于最终的特征精炼和重建
        for i in range(7, len(depths)):
            self.stage8.append(
                RTMSA(dim=embed_dims[i],
                      input_resolution=img_size,
                      depth=depths[i],
                      num_heads=num_heads[i],
                      # 对于独立重建层（indep_reconsts），时间窗口大小为1（独立处理每一帧）
                      # 这样可以独立提取每一帧的特征，而不考虑时间依赖关系
                      # 对于其他层，使用完整的窗口大小
                      window_size=[1, window_size[1], window_size[2]] if i in indep_reconsts else window_size,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],  # 该阶段的drop_path率
                      norm_layer=norm_layer,
                      use_checkpoint_attn=use_checkpoint_attns[i],
                      use_checkpoint_ffn=use_checkpoint_ffns[i]
                      )
            )

        # 归一化和维度投影
        self.norm = norm_layer(embed_dims[-1])  # 最终归一化
        self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])  # 投影回初始维度

        # 重建模块：根据任务类型选择不同的重建方式
        if self.pa_frames:
            if self.upscale == 1:
                # 视频去模糊等任务：直接输出，不需要上采样
                self.conv_last = nn.Conv3d(embed_dims[0], out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            else:
                # 视频超分辨率：需要上采样
                num_feat = 64
                self.conv_before_upsample = nn.Sequential(
                    nn.Conv3d(embed_dims[0], num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                    nn.LeakyReLU(inplace=True))
                self.upsample = Upsample(upscale, num_feat)  # 上采样模块
                self.conv_last = nn.Conv3d(num_feat, out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        else:
            # 视频插帧等任务：使用2D卷积
            num_feat = 64
            self.linear_fuse = nn.Conv2d(embed_dims[0]*img_size[0], num_feat, kernel_size=1 , stride=1)
            self.conv_last = nn.Conv2d(num_feat, out_chans , kernel_size=7 , stride=1, padding=0)

    def init_weights(self, pretrained=None, strict=True):
        """初始化模型权重
        
        加载预训练权重（如果提供）。

        Args:
            pretrained (str, optional): 预训练权重路径。如果为None，则不加载预训练权重。默认为None
            strict (bool, optional): 是否严格加载预训练模型（要求所有键匹配）。默认为True
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

    def reflection_pad2d(self, x, pad=1):
        """反射填充（2D）
        
        对输入张量进行反射填充，支持任何数据类型（包括torch.bfloat16）。
        在H和W维度上进行反射填充。

        Args:
            x: 输入张量，形状为 (B, C, H, W)
            pad: 填充大小。默认为1

        Returns:
            x: 填充后的张量，形状为 (B, C, H+2*pad, W+2*pad)
        """
        # 在H维度上进行反射填充
        x = torch.cat([torch.flip(x[:, :, 1:pad+1, :], [2]), x, torch.flip(x[:, :, -pad-1:-1, :], [2])], 2)
        # 在W维度上进行反射填充
        x = torch.cat([torch.flip(x[:, :, :, 1:pad+1], [3]), x, torch.flip(x[:, :, :, -pad-1:-1], [3])], 3)
        return x

    def forward(self, x):
        """前向传播函数
        
        执行完整的视频恢复流程：
        1. 光流估计（如果启用并行变形对齐）：使用SpyNet估计相邻帧之间的光流
        2. 输入变形对齐（如果启用并行变形对齐）：使用光流将相邻帧对齐到当前帧
        3. 特征提取（通过U型网络结构）：使用Transformer提取多尺度特征
        4. 重建输出：根据任务类型（去模糊/超分辨率/插帧）进行不同的重建
        
        对于不同的任务类型，处理流程略有不同：
        - 视频去模糊/去噪：使用并行变形对齐，不需要上采样
        - 视频超分辨率：使用并行变形对齐，需要上采样
        - 视频插帧：不使用并行变形对齐，使用2D卷积处理

        Args:
            x: 输入视频，形状为 (N, D, C, H, W)，其中N为批次大小，D为帧数，C为通道数
               对于非盲去噪任务，C可能包含噪声水平图通道

        Returns:
            x: 恢复后的视频，形状为 (N, D, C, H, W) 或 (N, D, C, H*upscale, W*upscale)
        """
        # x: (N, D, C, H, W)

        # 主网络：使用并行变形对齐（PDA）的任务
        if self.pa_frames:
            # 获取噪声水平图（非盲去噪）
            # 对于非盲去噪，输入包含原始图像和噪声水平图，需要分离
            if self.nonblind_denoising:
                x, noise_level_map = x[:, :, :self.in_chans, :, :], x[:, :, self.in_chans:, :, :]

            x_lq = x.clone()  # 保存原始输入用于残差连接（全局残差连接）

            # 步骤1：计算光流：估计相邻帧之间的光流
            # 根据pa_frames参数，可能估计1帧、2帧或3帧间隔的光流
            flows_backward, flows_forward = self.get_flows(x)

            # 步骤2：变形输入：使用光流将相邻帧对齐到当前帧
            # 这是并行变形对齐的第一步：粗对齐（使用光流）
            # 返回对齐后的后向帧和前向帧，每个形状为 (B, D, C, H, W)
            x_backward, x_forward = self.get_aligned_image_2frames(x,  flows_backward[0], flows_forward[0])
            # 拼接原始帧、后向对齐帧和前向对齐帧
            # 通道维度从C变为3*C，提供更丰富的输入信息
            x = torch.cat([x, x_backward, x_forward], 2)

            # 拼接噪声水平图（非盲去噪）
            if self.nonblind_denoising:
                x = torch.cat([x, noise_level_map], 2)

            if self.upscale == 1:
                # 视频去模糊等任务：不需要上采样
                x = self.conv_first(x.transpose(1, 2))  # 转换为 (B, C, D, H, W)
                # 通过U型网络提取特征，然后投影回初始维度
                # 使用残差连接：x = x + conv_after_body(forward_features(x))
                x = x + self.conv_after_body(
                    self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
                x = self.conv_last(x).transpose(1, 2)  # 转换回 (B, D, C, H, W)
                # 残差连接：如果输入通道数多于输出通道数（如RGB+Spike -> RGB），
                # 只使用前out_chans个通道进行残差连接
                if x_lq.shape[2] != x.shape[2]:
                    return x + x_lq[:, :, :self.out_chans, :, :]
                return x + x_lq
            else:
                # 视频超分辨率：需要上采样
                x = self.conv_first(x.transpose(1, 2))  # 转换为 (B, C, D, H, W)
                # 通过U型网络提取特征，然后投影回初始维度
                # 使用残差连接：x = x + conv_after_body(forward_features(x))
                x = x + self.conv_after_body(
                    self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
                # 上采样并输出：先通过conv_before_upsample，然后上采样，最后通过conv_last
                x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)
                _, _, C, H, W = x.shape
                # 残差连接：将输入上采样到输出尺寸
                # 使用三线性插值上采样输入，然后与输出相加
                # 如果输入通道数多于输出通道数，只使用前out_chans个通道
                x_lq_upsampled = torch.nn.functional.interpolate(x_lq, size=(C, H, W), mode='trilinear', align_corners=False)
                if x_lq_upsampled.shape[2] != x.shape[2]:
                    x_lq_upsampled = x_lq_upsampled[:, :, :self.out_chans, :, :]
                return x + x_lq_upsampled
        else:
            # 视频插帧等任务：不使用并行变形对齐
            # 减去均值进行归一化
            x_mean = x.mean([1,3,4], keepdim=True)
            x = x - x_mean

            x = self.conv_first(x.transpose(1, 2))  # 转换为 (B, C, D, H, W)
            # 通过U型网络提取特征（不使用光流）
            x = x + self.conv_after_body(
                self.forward_features(x, [], []).transpose(1, 4)).transpose(1, 4)

            # 将时间维度展开，使用2D卷积处理
            x = torch.cat(torch.unbind(x , 2) , 1)  # (B, D*C, H, W)
            x = self.conv_last(self.reflection_pad2d(F.leaky_relu(self.linear_fuse(x), 0.2), pad=3))
            # 重新堆叠为时间序列
            x = torch.stack(torch.split(x, dim=1, split_size_or_sections=3), 1)  # (B, D, C, H, W)

            return x + x_mean  # 加回均值


    def get_flows(self, x):
        """获取光流
        
        根据pa_frames参数获取不同间隔的光流：
        - 2帧模式：获取(t,t+1)光流
        - 4帧模式：获取(t,t+1)和(t,t+2)光流
        - 6帧模式：获取(t,t+1)、(t,t+2)和(t,t+3)光流
        
        Args:
            x: 输入视频，形状为 (B, D, C, H, W)

        Returns:
            flows_backward: 后向光流列表，每个元素形状为 (B, D-?, 2, H//scale, W//scale)
            flows_forward: 前向光流列表，每个元素形状为 (B, D-?, 2, H//scale, W//scale)
        """

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
        """获取相邻两帧之间的光流（t和t+1）
        
        使用SpyNet网络估计相邻帧之间的光流，返回多尺度的光流。

        Args:
            x: 输入视频，形状为 (B, N, C, H, W)

        Returns:
            flows_backward: 后向光流列表（4个尺度），每个形状为 (B, N-1, 2, H//scale, W//scale)
            flows_forward: 前向光流列表（4个尺度），每个形状为 (B, N-1, 2, H//scale, W//scale)
        """
        b, n, c, h, w = x.size()
        
        # 对于RGB+Spike输入（c=4），只使用RGB通道（前3个）进行光流估计
        # SpyNet在RGB图像上预训练，期望3个通道
        if c > 3:
            x_rgb = x[:, :, :3, :, :]  # 只提取RGB通道
        else:
            x_rgb = x
        
        # 准备相邻帧对：x_1为前N-1帧，x_2为后N-1帧
        x_1 = x_rgb[:, :-1, :, :, :].reshape(-1, 3, h, w)
        x_2 = x_rgb[:, 1:, :, :, :].reshape(-1, 3, h, w)

        # 后向光流：从x_1到x_2（从t到t+1）
        flows_backward = self.spynet(x_1, x_2)
        # 将光流重塑为多尺度格式
        flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                          zip(flows_backward, range(4))]

        # 前向光流：从x_2到x_1（从t+1到t）
        flows_forward = self.spynet(x_2, x_1)
        # 将光流重塑为多尺度格式
        flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                         zip(flows_forward, range(4))]

        return flows_backward, flows_forward

    def get_flow_4frames(self, flows_forward, flows_backward):
        """从(t,t+1)和(t+1,t+2)光流推导(t,t+2)光流
        
        通过组合相邻帧的光流来推导间隔2帧的光流。这是光流组合的关键操作：
        要计算从帧t+2到帧t的光流，需要：
        1. 从帧t+2到帧t+1的光流（flow_n2）
        2. 从帧t+1到帧t的光流（flow_n1）
        3. 将flow_n1应用到flow_n2上，得到从t+2到t的组合光流
        
        光流组合公式：flow(t+2→t) = flow(t+1→t) + warp(flow(t+2→t+1), flow(t+1→t))
        其中warp表示使用光流进行变形操作。

        Args:
            flows_forward: 前向光流列表（从t到t+1），多尺度，每个元素形状为 (B, D-1, 2, H//scale, W//scale)
            flows_backward: 后向光流列表（从t+1到t），多尺度，每个元素形状为 (B, D-1, 2, H//scale, W//scale)

        Returns:
            flows_backward2: 后向光流列表（从t+2到t），每个形状为 (B, D-2, 2, H//scale, W//scale)
            flows_forward2: 前向光流列表（从t-2到t），每个形状为 (B, D-2, 2, H//scale, W//scale)
        """
        # 后向光流：从t+2到t（从未来帧到当前帧）
        d = flows_forward[0].shape[1]  # 帧数减1
        flows_backward2 = []
        # 对每个尺度的光流进行处理
        for flows in flows_backward:
            flow_list = []
            # 从后向前遍历：从最后一帧对开始
            for i in range(d - 1, 0, -1):
                flow_n1 = flows[:, i - 1, :, :, :]  # 从帧i+1到帧i的光流（1帧间隔）
                flow_n2 = flows[:, i, :, :, :]      # 从帧i+2到帧i+1的光流（1帧间隔）
                # 组合光流：从帧i+2到帧i = 从帧i+1到帧i + 变形后的从帧i+2到帧i+1
                # flow_warp将flow_n2按照flow_n1进行变形，然后相加得到组合光流
                # permute(0,2,3,1)将光流从(B,2,H,W)转换为(B,H,W,2)以匹配flow_warp的输入格式
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_backward2.append(torch.stack(flow_list, 1))

        # 前向光流：从t-2到t（从过去帧到当前帧）
        flows_forward2 = []
        # 对每个尺度的光流进行处理
        for flows in flows_forward:
            flow_list = []
            # 从前向后遍历：从第一帧对开始
            for i in range(1, d):
                flow_n1 = flows[:, i, :, :, :]      # 从帧i-1到帧i的光流（1帧间隔）
                flow_n2 = flows[:, i - 1, :, :, :]  # 从帧i-2到帧i-1的光流（1帧间隔）
                # 组合光流：从帧i-2到帧i = 从帧i-1到帧i + 变形后的从帧i-2到帧i-1
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_forward2.append(torch.stack(flow_list, 1))

        return flows_backward2, flows_forward2

    def get_flow_6frames(self, flows_forward, flows_backward, flows_forward2, flows_backward2):
        """从(t,t+2)和(t+2,t+3)光流推导(t,t+3)光流
        
        通过组合间隔2帧的光流来推导间隔3帧的光流。这是光流组合的进一步扩展：
        要计算从帧t+3到帧t的光流，需要：
        1. 从帧t+2到帧t的光流（flow_n1，2帧间隔，已通过get_flow_4frames计算）
        2. 从帧t+3到帧t+2的光流（flow_n2，1帧间隔）
        3. 将flow_n1应用到flow_n2上，得到从t+3到t的组合光流
        
        光流组合公式：flow(t+3→t) = flow(t+2→t) + warp(flow(t+3→t+2), flow(t+2→t))

        Args:
            flows_forward: 前向光流列表（从t到t+1），多尺度，每个元素形状为 (B, D-1, 2, H//scale, W//scale)
            flows_backward: 后向光流列表（从t+1到t），多尺度，每个元素形状为 (B, D-1, 2, H//scale, W//scale)
            flows_forward2: 前向光流列表（从t到t+2），多尺度，每个元素形状为 (B, D-2, 2, H//scale, W//scale)
            flows_backward2: 后向光流列表（从t+2到t），多尺度，每个元素形状为 (B, D-2, 2, H//scale, W//scale)

        Returns:
            flows_backward3: 后向光流列表（从t+3到t），每个形状为 (B, D-3, 2, H//scale, W//scale)
            flows_forward3: 前向光流列表（从t-3到t），每个形状为 (B, D-3, 2, H//scale, W//scale)
        """
        # 后向光流：从t+3到t（从未来帧到当前帧）
        d = flows_forward2[0].shape[1]  # 帧数减2
        flows_backward3 = []
        # 对每个尺度的光流进行处理，同时需要flows_backward和flows_backward2
        for flows, flows2 in zip(flows_backward, flows_backward2):
            flow_list = []
            # 从后向前遍历：从最后一帧对开始
            for i in range(d - 1, 0, -1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # 从帧i+2到帧i的光流（2帧间隔）
                flow_n2 = flows[:, i + 1, :, :, :]    # 从帧i+3到帧i+2的光流（1帧间隔）
                # 组合光流：从帧i+3到帧i = 从帧i+2到帧i + 变形后的从帧i+3到帧i+2
                # 注意：flows的索引是i+1，因为flows_backward的维度是D-1，而flows2的维度是D-2
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_backward3.append(torch.stack(flow_list, 1))

        # 前向光流：从t-3到t（从过去帧到当前帧）
        flows_forward3 = []
        # 对每个尺度的光流进行处理，同时需要flows_forward和flows_forward2
        for flows, flows2 in zip(flows_forward, flows_forward2):
            flow_list = []
            # 从前向后遍历：从第一帧对开始
            # 注意：起始索引是2，因为需要至少3帧才能计算3帧间隔的光流
            for i in range(2, d + 1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # 从帧i-2到帧i的光流（2帧间隔）
                flow_n2 = flows[:, i - 2, :, :, :]    # 从帧i-3到帧i-2的光流（1帧间隔）
                # 组合光流：从帧i-3到帧i = 从帧i-2到帧i + 变形后的从帧i-3到帧i-2
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_forward3.append(torch.stack(flow_list, 1))

        return flows_backward3, flows_forward3

    def get_aligned_image_2frames(self, x, flows_backward, flows_forward):
        """并行图像变形对齐（2帧模式）
        
        对每一帧，使用光流将相邻帧对齐到当前帧的位置。
        这是输入级的对齐，用于准备输入到第一个卷积层。

        Args:
            x: 输入图像，形状为 (B, D, C, H, W)
            flows_backward: 后向光流，形状为 (B, D-1, 2, H, W)
            flows_forward: 前向光流，形状为 (B, D-1, 2, H, W)

        Returns:
            [x_backward, x_forward]: 对齐后的后向和前向图像，每个形状为 (B, D, C*4, H, W)
        """
        # 后向对齐：从最后一帧向前处理
        n = x.size(1)
        # 最后一帧的对齐图像初始化为零，并重复4次（用于4个相邻帧）
        x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]  # 当前帧
            flow = flows_backward[:, i - 1, ...]  # 从帧i+1到帧i的光流
            # 使用光流将帧i+1变形对齐到帧i的位置，使用nearest4插值
            x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4'))

        # 前向对齐：从第一帧向后处理
        # 第一帧的对齐图像初始化为零，并重复4次
        x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]  # 当前帧
            flow = flows_forward[:, i, ...]  # 从帧i-1到帧i的光流
            # 使用光流将帧i-1变形对齐到帧i的位置
            x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4'))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def forward_features(self, x, flows_backward, flows_forward):
        """主网络特征提取（U型架构）
        
        这是VRT的核心特征提取网络，采用U型（U-Net）架构：
        - 编码路径：stage1 -> stage2 -> stage3 -> stage4（下采样，提取多尺度特征）
        - 瓶颈层：stage5（最深层特征）
        - 解码路径：stage6 -> stage7（上采样，融合多尺度特征）
        - 输出层：stage8（最终特征处理）
        
        每个stage使用不同尺度的光流进行并行变形对齐（PDA）：
        - stage1使用最粗尺度光流（flows[0::4]）
        - stage2使用次粗尺度光流（flows[1::4]）
        - stage3使用次细尺度光流（flows[2::4]）
        - stage4使用最细尺度光流（flows[3::4]）
        
        Args:
            x: 输入特征，形状为 (B, C, D, H, W)
            flows_backward: 后向光流列表（多尺度），每个元素形状为 (B, D-?, 2, H//scale, W//scale)
            flows_forward: 前向光流列表（多尺度），每个元素形状为 (B, D-?, 2, H//scale, W//scale)

        Returns:
            x: 提取的特征，形状为 (B, C, D, H, W)
        """
        # 编码路径：下采样并提取多尺度特征
        x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4])  # 最粗尺度光流
        x2 = self.stage2(x1, flows_backward[1::4], flows_forward[1::4])  # 次粗尺度光流
        x3 = self.stage3(x2, flows_backward[2::4], flows_forward[2::4])  # 次细尺度光流
        x4 = self.stage4(x3, flows_backward[3::4], flows_forward[3::4])  # 最细尺度光流
        
        # 瓶颈层：最深层特征提取
        x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4])
        
        # 解码路径：上采样并融合多尺度特征（带跳跃连接）
        x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4])  # 融合stage3特征
        x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4])  # 融合stage2特征
        x = x + x1  # 融合stage1特征（跳跃连接）

        # 输出层：最终特征处理
        # stage8包含：维度投影层 + 多个RTMSA模块
        # RTMSA模块不使用互注意力，只使用自注意力，用于最终的特征精炼
        for layer in self.stage8:
            x = layer(x)

        # 层归一化：先转换为 (B, D, H, W, C) 进行归一化，再转回 (B, C, D, H, W)
        # LayerNorm需要在最后一个维度上进行归一化，所以需要转置
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x