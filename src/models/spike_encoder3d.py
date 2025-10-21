from __future__ import annotations

from typing import List, Tuple
import time

import torch
import torch.nn as nn

from src.utils.timing_logger import log_timing
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out + identity
        out = self.act(out)
        return out


class SpikeEncoder3D(nn.Module):
    """
    3D Conv 残差金字塔：在时间和空间维度下采样，输出与 VRT 编码端各尺度对齐的 5D 特征列表。
    
    空间下采样模式匹配VRT编码端的4个尺度:
    Scale 1: 原始分辨率 1x
    Scale 2: 1/2x
    Scale 3: 1/4x
    Scale 4: 1/8x

    输入:  x: (B, T, K, H, W)
    输出:  List[Tensor]，长度为4，每个张量形状为 (B, C_i, T_i, H_i, W_i)
    """

    def __init__(
        self,
        in_bins: int,
        channels_per_scale: List[int],
        temporal_strides: List[int] | None = None,
        spatial_strides: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.in_bins = in_bins
        self.channels_per_scale = channels_per_scale
        # temporal_strides 控制各尺度之间沿时间维的步长（长度 = num_scales-1）。
        if temporal_strides is None:
            self.temporal_strides = [1] * (len(channels_per_scale) - 1)
        else:
            if len(temporal_strides) != len(channels_per_scale) - 1:
                raise ValueError(
                    f"temporal_strides length must be {len(channels_per_scale) - 1}, got {len(temporal_strides)}"
                )
            self.temporal_strides = temporal_strides
        
        # spatial_strides 控制各尺度之间空间维的步长，匹配VRT编码端架构
        # VRT 编码端: 1x -> 1/2x -> 1/4x -> 1/8x
        # 空间stride: [2, 2, 2]（3次下采样）
        if spatial_strides is None:
            # 默认匹配VRT编码端的4个尺度（需要3次下采样）
            self.spatial_strides = [2, 2, 2]
        else:
            if len(spatial_strides) != len(channels_per_scale) - 1:
                raise ValueError(
                    f"spatial_strides length must be {len(channels_per_scale) - 1}, got {len(spatial_strides)}"
                )
            self.spatial_strides = spatial_strides

        self.act = nn.ReLU(inplace=True)

        # 第一尺度投影与残差
        c0 = channels_per_scale[0]
        self.in_proj = nn.Conv3d(in_bins, c0, kernel_size=3, stride=1, padding=1)
        self.res0 = nn.Sequential(ResidualBlock3D(c0), ResidualBlock3D(c0))

        # 后续尺度：时间和空间维下采样；通道对齐到对应 C_i
        self.downs: nn.ModuleList = nn.ModuleList()
        self.residuals: nn.ModuleList = nn.ModuleList()
        for i in range(1, len(channels_per_scale)):
            cin = channels_per_scale[i - 1]
            cout = channels_per_scale[i]
            s_t = int(self.temporal_strides[i - 1])
            s_s = int(self.spatial_strides[i - 1])  # 空间步长
            if s_t < 1 or s_s < 1:
                raise ValueError("strides must be >=1")
            # 使用 (s_t, s_s, s_s) 的stride进行下采样
            self.downs.append(nn.Conv3d(cin, cout, kernel_size=(3, 3, 3), stride=(s_t, s_s, s_s), padding=1))
            self.residuals.append(nn.Sequential(ResidualBlock3D(cout), ResidualBlock3D(cout)))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        start_time = time.time()
        assert x.dim() == 5, "Expected input of shape (B, T, K, H, W)"
        
        # (B, T, K, H, W) -> (B, K, T, H, W)
        permute_start = time.time()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        permute_time = time.time() - permute_start
        log_timing("Spike编码器/输入维度转换", permute_time * 1000)

        feats: List[torch.Tensor] = []
        
        # 第一尺度处理
        scale0_start = time.time()
        out = self.act(self.in_proj(x))
        out = self.res0(out)
        feats.append(out)
        scale0_time = time.time() - scale0_start
        log_timing("Spike编码器/尺度0", scale0_time * 1000)

        # 后续尺度处理
        for i, (down, res) in enumerate(zip(self.downs, self.residuals), 1):
            scale_start = time.time()
            out = self.act(down(out))
            out = res(out)
            feats.append(out)
            scale_time = time.time() - scale_start
            log_timing(f"Spike编码器/尺度{i}", scale_time * 1000)

        total_time = time.time() - start_time
        log_timing("Spike编码器", total_time * 1000)
        return feats



