import torch

from models.architectures.vrt.attention import WindowAttention


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


