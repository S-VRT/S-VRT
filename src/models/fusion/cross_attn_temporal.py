# src/models/fusion/cross_attn_temporal.py
import logging
from typing import Dict
import time

import torch
import torch.nn as nn

from src.utils.attention import BaseChunkableAttention
from src.utils.timing_logger import log_timing

logger = logging.getLogger(__name__)


class TemporalCrossAttention(BaseChunkableAttention):
    """
    Chunked Temporal Cross-Attention.
    Q is from RGB, K/V are from Spike. Attention is along Time dimension.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        chunk_cfg: Dict | None = None,
    ):
        super().__init__(chunk_cfg)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout
        )

    def forward(self, q, k, v):  # q, k, v are [B, H, W, T, C]
        start_time = time.time()
        B, H, W, T, C = q.shape

        output = torch.zeros_like(q)
        h_chunk, w_chunk = self._get_chunks(B, H, W)
        self._log_chunk_info(
            h_chunk, w_chunk, B, H, W, T, C, self.__class__.__name__
        )

        chunk_count = 0
        attn_total_time = 0.0
        
        for h_start in range(0, H, h_chunk):
            h_end = min(h_start + h_chunk, H)
            for w_start in range(0, W, w_chunk):
                w_end = min(w_start + w_chunk, W)

                q_chunk = q[:, h_start:h_end, w_start:w_end, :, :]
                k_chunk = k[:, h_start:h_end, w_start:w_end, :, :]
                v_chunk = v[:, h_start:h_end, w_start:w_end, :, :]

                h_, w_ = h_end - h_start, w_end - w_start

                q_flat = q_chunk.reshape(B * h_ * w_, T, C)
                k_flat = k_chunk.reshape(B * h_ * w_, T, C)
                v_flat = v_chunk.reshape(B * h_ * w_, T, C)

                attn_start = time.time()
                attn_out, _ = self.attn(q_flat, k_flat, v_flat, need_weights=False)
                attn_time = time.time() - attn_start
                attn_total_time += attn_time

                attn_out = attn_out.view(B, h_, w_, T, C)
                output[:, h_start:h_end, w_start:w_end, :, :] = attn_out
                chunk_count += 1

        total_time = time.time() - start_time
        log_timing("VRT融合/Cross-Attention", attn_total_time * 1000)
        
        return output


class TemporalCrossAttnFuseBlock(nn.Module):
    """
    Temporal Cross-Attention fusion block with pre-normalization,
    chunked attention, and a feed-forward network.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        mlp_ratio: int = 2,
        chunk_cfg: Dict | None = None,
    ):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = TemporalCrossAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            chunk_cfg=chunk_cfg,
        )
        self.ln_ffn = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, Fr, Fs):  # Fr, Fs: [B, T, C, H, W]
        start_time = time.time()
        B, T, C, H, W = Fr.shape
        
        # Permute to [B, H, W, T, C] for spatial chunking
        permute_start = time.time()
        Fr_bhwtc = Fr.permute(0, 3, 4, 1, 2).contiguous()
        Fs_bhwtc = Fs.permute(0, 3, 4, 1, 2).contiguous()
        permute_time = time.time() - permute_start
        
        # Flatten for LayerNorm
        norm_start = time.time()
        Fr_flat = Fr_bhwtc.view(-1, C)
        Fs_flat = Fs_bhwtc.view(-1, C)
        
        Q = self.ln_q(Fr_flat).view(B, H, W, T, C)
        K = self.ln_kv(Fs_flat).view(B, H, W, T, C)
        V = K
        norm_time = time.time() - norm_start
        
        # Chunked Cross-Attention
        attn_start = time.time()
        Y = self.attn(Q, K, V)
        attn_time = time.time() - attn_start
        X = Fr_bhwtc + Y  # Add to original Fr, not normed Q
        
        # Feed-Forward
        ffn_start = time.time()
        X_flat = X.view(-1, C)
        Y_ffn = self.ffn(self.ln_ffn(X_flat))
        Y_ffn = Y_ffn.view(B, H, W, T, C)
        X = X + Y_ffn
        ffn_time = time.time() - ffn_start
        
        # Permute back to [B, T, C, H, W]
        output = X.permute(0, 3, 4, 1, 2).contiguous()
        
        total_time = time.time() - start_time
        log_timing("VRT融合/维度转换", permute_time * 1000)
        log_timing("VRT融合/LayerNorm", norm_time * 1000)
        log_timing("VRT融合/FFN", ffn_time * 1000)
        
        return output


class MultiScaleTemporalCrossAttnFuse(nn.Module):
    def __init__(
        self,
        channels_per_scale: list[int],
        heads: int,
        dropout: float = 0.0,
        mlp_ratio: int = 2,
        chunk_cfg: Dict | None = None,
    ):
        super().__init__()
        self.fuse_blocks = nn.ModuleList(
            [
                TemporalCrossAttnFuseBlock(
                    dim=channels,
                    heads=heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    chunk_cfg=chunk_cfg,
                )
                for channels in channels_per_scale
            ]
        )

    def forward(self, Fr_list, Fs_list):
        start_time = time.time()
        
        results = []
        for i, (fuse, Fr, Fs) in enumerate(zip(self.fuse_blocks, Fr_list, Fs_list)):
            scale_start = time.time()
            
            result = fuse(Fr, Fs)
            results.append(result)
            
            scale_time = time.time() - scale_start
            log_timing(f"VRT融合/尺度{i}", scale_time * 1000)
        
        total_time = time.time() - start_time
        log_timing("VRT融合", total_time * 1000)
        
        return results



