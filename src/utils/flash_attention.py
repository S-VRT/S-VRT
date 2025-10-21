"""
Flash Attention 工具函数

提供统一的接口来检测和使用 Flash Attention。
支持多种实现：
1. PyTorch 2.0+ 内置的 scaled_dot_product_attention (SDPA)
2. xformers 的 memory_efficient_attention
3. 标准的 attention 实现作为 fallback
"""

import torch
import torch.nn.functional as F
from typing import Optional
import logging
import types

logger = logging.getLogger(__name__)

# 全局标志：是否已检测 Flash Attention 可用性
_FLASH_ATTN_AVAILABLE = None
_FLASH_ATTN_METHOD = None


def check_flash_attention_available():
    """
    检测 Flash Attention 是否可用
    
    Returns:
        tuple: (is_available: bool, method: str)
            method 可以是 'sdpa', 'xformers', 或 'none'
    """
    global _FLASH_ATTN_AVAILABLE, _FLASH_ATTN_METHOD
    
    if _FLASH_ATTN_AVAILABLE is not None:
        return _FLASH_ATTN_AVAILABLE, _FLASH_ATTN_METHOD
    
    # 检查 PyTorch 2.0+ 的 SDPA
    if hasattr(F, 'scaled_dot_product_attention'):
        try:
            # 测试是否真正可用（某些版本/设备上可能不支持）
            test_q = torch.randn(1, 1, 1, 8, device='cuda' if torch.cuda.is_available() else 'cpu')
            test_k = torch.randn(1, 1, 1, 8, device='cuda' if torch.cuda.is_available() else 'cpu')
            test_v = torch.randn(1, 1, 1, 8, device='cuda' if torch.cuda.is_available() else 'cpu')
            _ = F.scaled_dot_product_attention(test_q, test_k, test_v)
            _FLASH_ATTN_AVAILABLE = True
            _FLASH_ATTN_METHOD = 'sdpa'
            logger.info("Flash Attention enabled: using PyTorch SDPA (scaled_dot_product_attention)")
            return True, 'sdpa'
        except Exception as e:
            logger.debug(f"SDPA test failed: {e}")
    
    # 检查 xformers
    try:
        import xformers.ops as xops
        if hasattr(xops, 'memory_efficient_attention'):
            _FLASH_ATTN_AVAILABLE = True
            _FLASH_ATTN_METHOD = 'xformers'
            logger.info("Flash Attention enabled: using xformers.ops.memory_efficient_attention")
            return True, 'xformers'
    except ImportError:
        pass
    
    # 都不可用，使用标准实现
    _FLASH_ATTN_AVAILABLE = False
    _FLASH_ATTN_METHOD = 'none'
    logger.info("Flash Attention not available, using standard attention implementation")
    return False, 'none'


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    使用 Flash Attention 计算注意力
    
    Args:
        q: Query tensor, shape (B, num_heads, seq_len, head_dim)
        k: Key tensor, shape (B, num_heads, seq_len, head_dim)
        v: Value tensor, shape (B, num_heads, seq_len, head_dim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        scale: Optional scale factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor, shape (B, num_heads, seq_len, head_dim)
    """
    is_available, method = check_flash_attention_available()
    
    if method == 'sdpa':
        return _flash_attention_sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale)
    elif method == 'xformers':
        return _flash_attention_xformers(q, k, v, attn_mask, dropout_p, scale)
    else:
        return _standard_attention(q, k, v, attn_mask, dropout_p, scale)


def _flash_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
) -> torch.Tensor:
    """使用 PyTorch 的 scaled_dot_product_attention"""
    # SDPA 不支持同时使用 attn_mask 和 is_causal
    if attn_mask is not None:
        is_causal = False
    
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if q.requires_grad else 0.0,  # 推理时不使用 dropout
        is_causal=is_causal,
        scale=scale,
    )


def _flash_attention_xformers(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    scale: Optional[float],
) -> torch.Tensor:
    """使用 xformers 的 memory_efficient_attention"""
    import xformers.ops as xops
    
    # xformers 需要 (B, seq_len, num_heads, head_dim) 格式
    B, H, N, D = q.shape
    q = q.transpose(1, 2)  # (B, N, H, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    out = xops.memory_efficient_attention(
        q, k, v,
        attn_bias=attn_mask,
        p=dropout_p if q.requires_grad else 0.0,
        scale=scale,
    )
    
    # 转回 (B, H, N, D)
    out = out.transpose(1, 2)
    return out


def _standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    scale: Optional[float],
) -> torch.Tensor:
    """标准的注意力实现（作为 fallback）"""
    B, H, N, D = q.shape
    
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    
    # 计算注意力分数
    attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
    
    # 应用 mask
    if attn_mask is not None:
        # 处理不同形状的 mask
        if attn_mask.dim() == 2:  # (N, N)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        elif attn_mask.dim() == 3:  # (B, N, N) or (H, N, N)
            if attn_mask.shape[0] == B:
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N)
            else:
                attn_mask = attn_mask.unsqueeze(0)  # (1, H, N, N)
        
        # 将 True/1 位置设为 -inf（这些位置会被 mask 掉）
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        else:
            attn = attn + attn_mask
    
    # Softmax
    attn = F.softmax(attn, dim=-1)
    
    # Dropout
    if dropout_p > 0.0 and q.requires_grad:
        attn = F.dropout(attn, p=dropout_p)
    
    # 计算输出
    out = attn @ v  # (B, H, N, D)
    
    return out


def apply_flash_attention_to_vrt(vrt_model):
    """
    将 VRT 模型的 attention 替换为 Flash Attention
    
    Args:
        vrt_model: VRT 模型实例
    
    Returns:
        修改后的模型（原地修改）
    """
    is_available, method = check_flash_attention_available()
    
    if not is_available:
        logger.warning("Flash Attention not available, VRT will use standard attention")
        return vrt_model
    
    # 递归查找并替换所有 WindowAttention 模块
    # 计数有多少个 WindowAttention 模块被 patch
    patch_count = [0]  # 使用列表以便在闭包中修改
    
    def replace_attention_recursive(module):
        for name, child in module.named_children():
            if child.__class__.__name__ == 'WindowAttention':
                # 为 WindowAttention 添加 flash attention 方法
                _patch_window_attention(child, method)
                patch_count[0] += 1
            else:
                replace_attention_recursive(child)
    
    replace_attention_recursive(vrt_model)
    logger.info(f"VRT model patched with Flash Attention (method: {method}), patched {patch_count[0]} WindowAttention modules")
    
    return vrt_model


def _patch_window_attention(window_attn_module, method: str):
    """
    Monkey-patch WindowAttention 模块的 attention 方法
    
    Args:
        window_attn_module: WindowAttention 模块实例
        method: Flash Attention 方法 ('sdpa' 或 'xformers')
    """
    original_attention = window_attn_module.attention
    
    def flash_attention_forward(self, q, k, v, mask, x_shape, relative_position_encoding=True):
        """
        使用 Flash Attention 的版本
        
        Args:
            self: WindowAttention 实例（自动传入）
            q: (B_, nH, N, C) Query
            k: (B_, nH, N, C) Key
            v: (B_, nH, N, C) Value
            mask: (nW, N, N) or None
            x_shape: (B_, N, C)
            relative_position_encoding: bool
        """
        B_, N, C = x_shape
        nH = self.num_heads
        
        # 处理 relative position encoding
        attn_bias = None
        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)
            ].reshape(N, N, -1)  # (N, N, nH)
            attn_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # (1, nH, N, N)
        
        # 处理 window mask
        if mask is not None:
            nW = mask.shape[0]
            # mask: (nW, N, N) -> (B_//nW, nW, 1, N, N)
            mask_expanded = mask[:, :N, :N].unsqueeze(1).unsqueeze(0)  # (1, nW, 1, N, N)
            mask_expanded = mask_expanded.expand(B_ // nW, nW, nH, N, N)
            mask_expanded = mask_expanded.reshape(B_, nH, N, N)
            
            if attn_bias is not None:
                attn_bias = attn_bias + mask_expanded
            else:
                attn_bias = mask_expanded
        
        # 使用 Flash Attention
        scale = self.scale
        x = flash_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0, scale=scale)
        
        # Reshape
        x = x.transpose(1, 2).reshape(B_, N, C)
        
        return x
    
    # 替换 attention 方法
    window_attn_module.attention = flash_attention_forward.__get__(window_attn_module, window_attn_module.__class__)
    
    # 标记已 patch
    window_attn_module._flash_attention_patched = True

