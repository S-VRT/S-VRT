import pytest
import torch

from models.architectures.vrt.attention import WindowAttention
import models.architectures.vrt.attention as attention_module


def test_window_attention_forward():
    # small window: D=2,H=4,W=4 -> N=32
    window_size = (2, 4, 4)
    dim = 64
    num_heads = 8
    N = window_size[0] * window_size[1] * window_size[2]
    Bn = 2
    x = torch.randn(Bn, N, dim)
    attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, qkv_bias=True, mut_attn=False)
    out = attn(x, mask=None)
    assert out.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_window_attention_flash_vs_sdpa_parity():
    """互注意力路径：flash-attn 与 SDPA 数值接近（atol=5e-2，fp16 精度）"""
    try:
        from flash_attn import flash_attn_func as _
    except ImportError:
        pytest.skip("flash-attn not installed")

    window_size = (2, 4, 4)
    dim = 64
    num_heads = 8
    N = window_size[0] * window_size[1] * window_size[2]
    Bn = 2
    device = "cuda"
    dtype = torch.float16

    torch.manual_seed(42)
    x = torch.randn(Bn, N, dim, dtype=dtype, device=device)

    # mut_attn=False → 只走 self-attention 路径（relative_position_encoding=True）
    # 要测互注意力路径需要 mut_attn=True，但这里直接测 attention() 内部
    attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads,
                           qkv_bias=True, mut_attn=False, use_flash_attn=False)
    attn = attn.to(dtype=dtype, device=device)
    attn.eval()

    with torch.no_grad():
        # Build q/k/v once, then compare SDPA vs flash on the same branch
        qkv = attn.qkv_self(x).reshape(Bn, N, 3, num_heads, dim // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # SDPA path (force use_flash_attn=False)
        attn.use_flash_attn = False
        out_sdpa = attn.attention(q, k, v, mask=None, x_shape=(Bn, N, dim),
                                  relative_position_encoding=False)

        # flash path
        attn.use_flash_attn = True
        out_flash_raw = attn.attention(q, k, v, mask=None, x_shape=(Bn, N, dim),
                                       relative_position_encoding=False)

    # flash-attn vs SDPA for mutual-attention path: relaxed tolerance for fp16
    assert out_flash_raw.shape == (Bn, N, dim)
    assert out_sdpa.shape == (Bn, N, dim)
    assert torch.allclose(out_flash_raw.float(), out_sdpa.float(), atol=5e-2, rtol=1e-2), \
        f"max diff: {(out_flash_raw.float() - out_sdpa.float()).abs().max().item():.4f}"


def test_window_attention_uses_flash_only_for_unmasked_half_mutual_path(monkeypatch):
    calls = []

    def fake_flash_attn_func(q, k, v, softmax_scale=None):
        calls.append((q.shape, k.shape, v.shape, softmax_scale))
        return torch.zeros_like(q)

    monkeypatch.setattr(attention_module, "FLASH_ATTN_AVAILABLE", True)
    monkeypatch.setattr(attention_module, "flash_attn_func", fake_flash_attn_func)

    attn = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2,
                           mut_attn=True, use_flash_attn=True)
    q = torch.randn(2, 2, 4, 4, dtype=torch.float16)
    k = torch.randn(2, 2, 4, 4, dtype=torch.float16)
    v = torch.randn(2, 2, 4, 4, dtype=torch.float16)

    out = attn.attention(q, k, v, mask=None, x_shape=(2, 4, 8),
                         relative_position_encoding=False)

    assert out.shape == (2, 4, 8)
    assert len(calls) == 1


def test_window_attention_self_attention_keeps_sdpa_path_with_relative_bias(monkeypatch):
    def fail_flash_attn_func(*args, **kwargs):
        raise AssertionError("self-attention with relative position bias must not call flash_attn_func")

    monkeypatch.setattr(attention_module, "FLASH_ATTN_AVAILABLE", True)
    monkeypatch.setattr(attention_module, "flash_attn_func", fail_flash_attn_func)

    attn = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2,
                           mut_attn=False, use_flash_attn=True)
    q = torch.randn(2, 2, 8, 4, dtype=torch.float16)
    k = torch.randn(2, 2, 8, 4, dtype=torch.float16)
    v = torch.randn(2, 2, 8, 4, dtype=torch.float16)

    out = attn.attention(q, k, v, mask=None, x_shape=(2, 8, 8),
                         relative_position_encoding=True)

    assert out.shape == (2, 8, 8)
