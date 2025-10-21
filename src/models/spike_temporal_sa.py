# src/models/spike_temporal_sa.py
import logging
from typing import Dict
import time

import torch
import torch.nn as nn

from src.utils.attention import BaseChunkableAttention
from src.utils.timing_logger import log_timing

logger = logging.getLogger(__name__)


class SpikeTemporalSelfAttention(BaseChunkableAttention):
    """
    Spike branch temporal self-attention module.
    Performs self-attention along the time dimension T for each spatial location (h,w).
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        mlp_ratio: int = 2,
        chunk_cfg: Dict | None = None,
    ):
        super().__init__(chunk_cfg)
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        start_time = time.time()
        B, T, C, H, W = x.shape

        # Permute to [B, H, W, T, C] for temporal attention
        permute_start = time.time()
        x_bhwtc = x.permute(0, 3, 4, 1, 2).contiguous()
        out_bhwtc = x.new_zeros(B, H, W, T, C)
        permute_time = time.time() - permute_start

        h_chunk, w_chunk = self._get_chunks(B, H, W)
        self._log_chunk_info(h_chunk, w_chunk, B, H, W, T, C, self.__class__.__name__)

        # Process in chunks to manage memory
        chunk_count = 0
        chunk_total_time = 0.0
        attn_total_time = 0.0
        ffn_total_time = 0.0
        
        for hs in range(0, H, h_chunk):
            he = min(hs + h_chunk, H)
            for ws in range(0, W, w_chunk):
                we = min(ws + w_chunk, W)
                chunk_start = time.time()

                chunk = x_bhwtc[:, hs:he, ws:we, :, :].contiguous()
                h_, w_ = he - hs, we - ws
                flat = chunk.reshape(B * h_ * w_, T, C)

                # Self-Attention
                attn_start = time.time()
                y = self.ln1(flat)
                y, _ = self.attn(y, y, y, need_weights=False)
                flat = flat + y
                attn_time = time.time() - attn_start
                attn_total_time += attn_time

                # Feed-Forward
                ffn_start = time.time()
                y = self.mlp(self.ln2(flat))
                flat = flat + y
                ffn_time = time.time() - ffn_start
                ffn_total_time += ffn_time

                out_bhwtc[:, hs:he, ws:we, :, :] = flat.reshape(B, h_, w_, T, C)
                
                chunk_time = time.time() - chunk_start
                chunk_total_time += chunk_time
                chunk_count += 1

        # Permute back to [B, T, C, H, W]
        out = out_bhwtc.permute(0, 3, 4, 1, 2).contiguous()
        
        total_time = time.time() - start_time
        log_timing("Spike时间自注意力/维度转换", permute_time * 1000)
        log_timing("Spike时间自注意力/Self-Attention", attn_total_time * 1000)
        log_timing("Spike时间自注意力/FFN", ffn_total_time * 1000)
        log_timing("Spike时间自注意力/块处理", chunk_total_time * 1000)
        
        return out


class SpikeTemporalSA(nn.Module):
    """
    Multi-scale Spike Temporal Self-Attention.
    Creates a TemporalSelfAttentionBlock for each scale.
    """

    def __init__(
        self,
        channels_per_scale: list[int],
        heads: int = 4,
        dropout: float = 0.0,
        mlp_ratio: int = 2,
        chunk_cfg: Dict | None = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SpikeTemporalSelfAttention(
                    dim=c,
                    heads=heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    chunk_cfg=chunk_cfg,
                )
                for c in channels_per_scale
            ]
        )

    def forward(self, feats_list):
        start_time = time.time()
        # print(f"  [SpikeTemporalSA] 开始处理{len(feats_list)}个尺度的特征")
        
        outputs = []
        for i, (block, feat) in enumerate(zip(self.blocks, feats_list)):
            scale_start = time.time()
            # print(f"  [SpikeTemporalSA] 尺度{i}: 输入shape={feat.shape}")
            
            # Input format from SpikeEncoder3D: [B, C, T, H, W]
            # Expected format for attention block: [B, T, C, H, W]
            feat_btchw = feat.permute(0, 2, 1, 3, 4)

            # Temporal Self-Attention
            out_btchw = block(feat_btchw)

            # Convert back to VRT format: [B, C, T, H, W]
            out = out_btchw.permute(0, 2, 1, 3, 4)
            outputs.append(out)
            
            scale_time = time.time() - scale_start
            log_timing(f"Spike时间自注意力/尺度{i}", scale_time * 1000)
        
        total_time = time.time() - start_time
        log_timing("Spike时间自注意力", total_time * 1000)
        return outputs
