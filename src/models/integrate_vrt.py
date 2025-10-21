from __future__ import annotations

from typing import List
import types
import time

import torch
import torch.nn as nn

from src.utils.timing_logger import log_timing
from .spike_encoder3d import SpikeEncoder3D
from .spike_temporal_sa import SpikeTemporalSA
from .fusion.cross_attn_temporal import MultiScaleTemporalCrossAttnFuse
from einops import rearrange


class VRTWithSpike(nn.Module):
    """
    新版架构：Spike 与 RGB 各自完成时域建模后，再通过 Cross-Attention 融合
    
    流程：
    1. RGB → VRT 编码 + TMSA → Fr_i (各尺度特征)
    2. Spike → SpikeEncoder3D → Fs_i
    3. Spike → TemporalSA → Fs'_i (时间维 Self-Attention)
    4. Cross-Attention 融合：Ff_i = CrossAttn(Q=Fr_i, K/V=Fs'_i)
    5. Ff_i → VRT 解码端
    """

    def __init__(
        self,
        vrt_backbone: nn.Module,
        spike_bins: int,
        channels_per_scale: list[int] | None = None,
        temporal_strides: list[int] | None = None,
        spatial_strides: list[int] | None = None,
        tsa_heads: int = 4,
        tsa_dropout: float = 0.0,
        tsa_mlp_ratio: int = 2,
        tsa_chunk_cfg: dict | None = None,
        fuse_heads: int = 4,
        fuse_dropout: float = 0.0,
        fuse_mlp_ratio: int = 2,
        fuse_chunk_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.vrt = vrt_backbone
        
        # 确定每个尺度的通道数（默认：4个尺度，每个96通道）
        if channels_per_scale is None:
            channels_per_scale = [96] * 4
        if len(channels_per_scale) != 4:
            raise ValueError(f"channels_per_scale must have length 4, got {len(channels_per_scale)}")

        # Spike 编码器：输出多尺度特征 Fs_1..4
        self.spike_encoder = SpikeEncoder3D(
            in_bins=spike_bins, 
            channels_per_scale=channels_per_scale,
            temporal_strides=temporal_strides,
            spatial_strides=spatial_strides,
        )
        
        # Spike 时间维 Self-Attention：Fs_i → Fs'_i
        self.spike_temporal_sa = SpikeTemporalSA(
            channels_per_scale=channels_per_scale, 
            heads=tsa_heads,
            dropout=tsa_dropout,
            mlp_ratio=tsa_mlp_ratio,
            chunk_cfg=tsa_chunk_cfg,
        )
        
        # 时间维 Cross-Attention 融合：(Fr_i, Fs'_i) → Ff_i
        self.cross_attn_fuse = MultiScaleTemporalCrossAttnFuse(
            channels_per_scale=channels_per_scale, 
            heads=fuse_heads,
            dropout=fuse_dropout,
            mlp_ratio=fuse_mlp_ratio,
            chunk_cfg=fuse_chunk_cfg,
        )

    def _monkeypatch_forward_features(self, spike_feats_fused: List[torch.Tensor]):
        """
        Monkey-patch VRT 的 forward_features 方法，只在编码端的4个stage做融合
        
        符合开发指导 5.5 节的集成流程：
        1. VRT 编码端 Stage 1-4 完成 TMSA 后，得到 Fr_1..4
        2. 与 Spike 分支的 Fs'_1..4 做 Cross-Attention 融合，得到 Ff_1..4
        3. Ff_1..4 作为编码端输出，继续走 VRT 的解码流程（Stage 5-8）
        
        关键：只在编码端的4个尺度做融合
        """
        vrt = self.vrt
        cross_attn_fuse = self.cross_attn_fuse

        def forward_features_fused(self_vrt, x, flows_backward, flows_forward):  # type: ignore[override]
            # x: (B, C, D, H, W)
            B, C, D, H, W = x.shape

            def _fuse_after_stage(i: int, x_stage_out: torch.Tensor) -> torch.Tensor:
                """
                编码端 Stage 输出后，与对应的 Spike 特征做 Cross-Attention 融合
                
                Args:
                    i: Stage 索引 (0-3 对应 stage1-4)
                    x_stage_out: Stage 输出，形状 [B, C, D, H, W]
                
                Returns:
                    融合后的特征，形状 [B, C, D, H, W]
                """
                fuse_start = time.time()
                
                if i >= len(spike_feats_fused):
                    raise ValueError(f"Stage index {i} out of range (expected 0-{len(spike_feats_fused)-1})")
                
                sf = spike_feats_fused[i]  # Fs'_i, 形状 [B, C, T, H, W] (Spike编码器输出格式)
                
                # 将 VRT 输出转换为 [B, T, C, H, W] 格式（VRT 使用 D 表示时间维）
                convert_start = time.time()
                Fr = x_stage_out  # [B, C, D, H, W]
                Fr_btchw = Fr.permute(0, 2, 1, 3, 4)  # [B, C, D, H, W] -> [B, D, C, H, W] -> [B, T, C, H, W]
                
                # Spike 编码器输出格式为 [B, C, T, H, W]，需要转换为 [B, T, C, H, W]
                sf_btchw = sf.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
                convert_time = time.time() - convert_start
                
                # 检查形状匹配
                if sf_btchw.shape[0] != Fr_btchw.shape[0] or sf_btchw.shape[1] != Fr_btchw.shape[1]:
                    raise ValueError(
                        f"Stage {i} batch/time mismatch: Spike {sf_btchw.shape} vs RGB {Fr_btchw.shape}"
                    )
                
                # 空间尺寸对齐（如需要）
                interp_time = 0.0
                if sf_btchw.shape[3] != Fr_btchw.shape[3] or sf_btchw.shape[4] != Fr_btchw.shape[4]:
                    interp_start = time.time()
                    b, t, c, h, w = sf_btchw.shape
                    sf_btchw = sf_btchw.reshape(b * t, c, h, w)
                    sf_btchw = torch.nn.functional.interpolate(
                        sf_btchw, size=(Fr_btchw.shape[3], Fr_btchw.shape[4]), 
                        mode='bilinear', align_corners=False
                    )
                    sf_btchw = sf_btchw.reshape(b, t, c, Fr_btchw.shape[3], Fr_btchw.shape[4])
                    interp_time = time.time() - interp_start
                    log_timing(f"VRT融合/尺度{i+1}/空间插值", interp_time * 1000)
                
                # Cross-Attention 融合（单尺度）
                # cross_attn_fuse 是 MultiScale 版本，这里使用对应尺度的 fuse_block
                cross_attn_start = time.time()
                Ff_btchw = cross_attn_fuse.fuse_blocks[i](Fr_btchw, sf_btchw)  # [B, T, C, H, W]
                cross_attn_time = time.time() - cross_attn_start
                
                # 转换回 VRT 格式 [B, C, D, H, W]
                Ff = Ff_btchw.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, D, H, W]
                
                fuse_time = time.time() - fuse_start
                log_timing(f"VRT融合/尺度{i+1}/转换", convert_time * 1000)
                
                return Ff

            # ===== 编码阶段（带融合）=====
            
            # Stage 1: 原始分辨率
            stage1_start = time.time()
            x1 = self_vrt.stage1(x, flows_backward[0::4], flows_forward[0::4])
            stage1_time = time.time() - stage1_start
            log_timing("VRT处理/Stage1", stage1_time * 1000)
            x1 = _fuse_after_stage(0, x1)  # Ff_1

            # Stage 2: 1/2 分辨率
            stage2_start = time.time()
            x2 = self_vrt.stage2(x1, flows_backward[1::4], flows_forward[1::4])
            stage2_time = time.time() - stage2_start
            log_timing("VRT处理/Stage2", stage2_time * 1000)
            x2 = _fuse_after_stage(1, x2)  # Ff_2

            # Stage 3: 1/4 分辨率
            stage3_start = time.time()
            x3 = self_vrt.stage3(x2, flows_backward[2::4], flows_forward[2::4])
            stage3_time = time.time() - stage3_start
            log_timing("VRT处理/Stage3", stage3_time * 1000)
            x3 = _fuse_after_stage(2, x3)  # Ff_3

            # Stage 4: 1/8 分辨率
            stage4_start = time.time()
            x4 = self_vrt.stage4(x3, flows_backward[3::4], flows_forward[3::4])
            stage4_time = time.time() - stage4_start
            log_timing("VRT处理/Stage4", stage4_time * 1000)
            x4 = _fuse_after_stage(3, x4)  # Ff_4

            # ===== 瓶颈层（不融合）=====
            stage5_start = time.time()
            x = self_vrt.stage5(x4, flows_backward[2::4], flows_forward[2::4])
            stage5_time = time.time() - stage5_start
            log_timing("VRT处理/Stage5", stage5_time * 1000)

            # ===== 解码阶段（不融合）=====
            
            # Stage 6: 1/4 分辨率（跳连 x3）
            stage6_start = time.time()
            x = self_vrt.stage6(x + x3, flows_backward[1::4], flows_forward[1::4])
            stage6_time = time.time() - stage6_start
            log_timing("VRT处理/Stage6", stage6_time * 1000)

            # Stage 7: 1/2 分辨率（跳连 x2）
            stage7_start = time.time()
            x = self_vrt.stage7(x + x2, flows_backward[0::4], flows_forward[0::4])
            stage7_time = time.time() - stage7_start
            log_timing("VRT处理/Stage7", stage7_time * 1000)

            # ===== 最终跳连 + 重建层 =====
            # 跳连 x1
            x = x + x1

            # Stage 8: 重建层
            stage8_start = time.time()
            for layer in self_vrt.stage8:
                x = layer(x)
            stage8_time = time.time() - stage8_start
            log_timing("VRT处理/Stage8", stage8_time * 1000)

            # Norm
            norm_start = time.time()
            x = rearrange(x, 'n c d h w -> n d h w c')
            x = self_vrt.norm(x)
            x = rearrange(x, 'n d h w c -> n c d h w')
            norm_time = time.time() - norm_start
            log_timing("VRT处理/LayerNorm", norm_time * 1000)
            
            return x

        # Bind the method to the VRT instance
        self._orig_forward_features = vrt.forward_features  # type: ignore[attr-defined]
        vrt.forward_features = types.MethodType(forward_features_fused, vrt)  # type: ignore[assignment]

    def _restore_forward_features(self) -> None:
        if hasattr(self, "_orig_forward_features"):
            self.vrt.forward_features = self._orig_forward_features  # type: ignore[attr-defined]
            delattr(self, "_orig_forward_features")

    def forward(self, rgb_clip: torch.Tensor, spike_vox: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            rgb_clip: (B, T, 3, H, W) RGB 输入序列
            spike_vox: (B, T, K, H, W) 体素化的 Spike 输入
        
        Returns:
            (B, T, 3, H, W) 重建的清晰帧
        """
        total_start = time.time()
        
        # Spike 分支处理
        # 1. SpikeEncoder3D: (B,T,K,H,W) -> List[(B,C_i,T_i,H_i,W_i)]，长度为4
        encoder_start = time.time()
        spike_feats = self.spike_encoder(spike_vox)  # Fs_1..4
        encoder_time = time.time() - encoder_start
        
        # 2. Temporal Self-Attention: Fs_i -> Fs'_i
        tsa_start = time.time()
        spike_feats_fused = self.spike_temporal_sa(spike_feats)  # Fs'_1..4
        tsa_time = time.time() - tsa_start

        # Monkey-patch VRT 的 forward_features，在各 Stage 后做 Cross-Attention 融合
        vrt_start = time.time()
        self._monkeypatch_forward_features(spike_feats_fused)
        try:
            out = self.vrt(rgb_clip)
        finally:
            self._restore_forward_features()
        vrt_time = time.time() - vrt_start
        
        total_time = time.time() - total_start
        log_timing("前向传播总耗时", total_time * 1000)
        log_timing("VRT处理", vrt_time * 1000)
        
        return out
