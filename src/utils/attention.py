import logging
import math
import time
from typing import Dict

import torch.nn as nn

logger = logging.getLogger(__name__)


def _derive_spatial_chunks(
    B: int,
    H: int,
    W: int,
    max_batch_tokens: int,
    chunk_size_soft_cap: int,
    shape: str = "square",
):
    """
    Derive (h_chunk, w_chunk) such that B*h_chunk*w_chunk <= max_batch_tokens,
    while being as large as possible, close to the target shape, and not exceeding chunk_size_soft_cap.
    """
    if B == 0:
        return min(H, chunk_size_soft_cap), min(W, chunk_size_soft_cap)

    # 1) Target tokens per chunk (defined by batch dimension: B*h*w)
    target_per_chunk = max(1, max_batch_tokens // B)

    # 2) Upper bound: area should not exceed the soft cap area
    soft_cap_area = chunk_size_soft_cap * chunk_size_soft_cap
    target_per_chunk = min(target_per_chunk, soft_cap_area)

    # 3) Determine (h,w) based on shape strategy
    if shape == "wide":
        # Wider (for W >> H)
        base_w = int(min(W, max(1, round(math.sqrt(target_per_chunk) * 1.25))))
        base_h = max(1, min(W, target_per_chunk // base_w))
    elif shape == "tall":
        # Taller (for H >> W)
        base_h = int(min(H, max(1, round(math.sqrt(target_per_chunk) * 1.25))))
        base_w = max(1, min(W, target_per_chunk // base_h))
    else:  # "square"
        side = int(max(1, math.isqrt(target_per_chunk)))
        base_h = min(H, side)
        base_w = min(W, side)

    # 4) Prevent zeros and re-constrain B*h*w <= max_batch_tokens
    base_h = max(1, base_h)
    base_w = max(1, base_w)
    while B * base_h * base_w > max_batch_tokens and (base_h > 1 or base_w > 1):
        if base_w >= base_h and base_w > 1:
            base_w -= 1
        elif base_h > 1:
            base_h -= 1
        else:
            break

    return max(1, base_h), max(1, base_w)


class BaseChunkableAttention(nn.Module):
    def __init__(self, chunk_cfg: Dict | None = None):
        super().__init__()
        self.chunk_cfg = chunk_cfg if chunk_cfg is not None else {}
        self.adaptive = self.chunk_cfg.get("ADAPTIVE_CHUNK", False)
        self.max_batch_tokens = self.chunk_cfg.get("MAX_BATCH_TOKENS", 49152)
        self.chunk_size_soft_cap = self.chunk_cfg.get("CHUNK_SIZE", 64)
        self.chunk_shape = self.chunk_cfg.get("CHUNK_SHAPE", "square").lower()
        self.last_log_time = 0
        self.log_interval = 300  # seconds

    def _get_chunks(self, B: int, H: int, W: int) -> tuple[int, int]:
        if self.adaptive:
            h_chunk, w_chunk = _derive_spatial_chunks(
                B=B,
                H=H,
                W=W,
                max_batch_tokens=self.max_batch_tokens,
                chunk_size_soft_cap=self.chunk_size_soft_cap,
                shape=self.chunk_shape,
            )
        else:
            h_chunk = min(H, self.chunk_size_soft_cap)
            w_chunk = min(W, self.chunk_size_soft_cap)
        return h_chunk, w_chunk

    def _log_chunk_info(self, h_chunk: int, w_chunk: int, B: int, H: int, W: int, T: int, C: int, class_name: str):
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            batch_tokens = B * h_chunk * w_chunk
            logger.info(
                f"[{class_name}] adaptive={self.adaptive} h_chunk={h_chunk} w_chunk={w_chunk} "
                f"B*chunk={batch_tokens} (cap={self.max_batch_tokens}) HxW={H}x{W} T={T} C={C}"
            )
            self.last_log_time = current_time
