import math
import copy
import pytest
import torch
import torch.nn as nn

from models.lora import LoRALinear, inject_lora, merge_lora


def test_lora_initial_output_equals_base():
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    lora = LoRALinear(copy.deepcopy(base), rank=4, alpha=8)
    x = torch.randn(2, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


class _MiniAttention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.qkv_self = nn.Linear(dim, dim * 3)
        self.qkv_mut = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.other_linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)


def test_inject_lora_hits_only_targets():
    m = _MiniAttention()
    replaced = inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    assert sorted(replaced) == ["proj", "qkv_mut", "qkv_self"]
    assert isinstance(m.qkv_self, LoRALinear)
    assert isinstance(m.qkv_mut, LoRALinear)
    assert isinstance(m.proj, LoRALinear)
    assert isinstance(m.other_linear, nn.Linear)
    assert not isinstance(m.other_linear, LoRALinear)


def test_inject_lora_is_idempotent():
    m = _MiniAttention()
    inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    replaced_second = inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    assert replaced_second == []
    assert isinstance(m.qkv_self, LoRALinear)
    assert isinstance(m.qkv_self.base, nn.Linear)
    assert not isinstance(m.qkv_self.base, LoRALinear)
