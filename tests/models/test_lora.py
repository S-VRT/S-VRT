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


def test_merge_lora_state_dict_matches_original():
    torch.manual_seed(0)
    original = _MiniAttention()
    original_keys = set(original.state_dict().keys())
    original_shapes = {k: v.shape for k, v in original.state_dict().items()}

    wrapped = copy.deepcopy(original)
    inject_lora(wrapped, ["qkv", "proj"], rank=4, alpha=8)
    # Perturb LoRA weights so merge is non-trivial
    for mod in wrapped.modules():
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_A.weight, std=0.01)
            nn.init.normal_(mod.lora_B.weight, std=0.01)

    merge_lora(wrapped)
    merged_sd = wrapped.state_dict()
    assert set(merged_sd.keys()) == original_keys
    for k in original_keys:
        assert merged_sd[k].shape == original_shapes[k], k


def test_merge_lora_forward_equivalence():
    torch.manual_seed(1)
    m = _MiniAttention()
    inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_A.weight, std=0.02)
            nn.init.normal_(mod.lora_B.weight, std=0.02)

    x = torch.randn(2, 8)
    with torch.no_grad():
        y_before = m.qkv_self(x)
    merged = copy.deepcopy(m)
    merge_lora(merged)
    with torch.no_grad():
        y_after = merged.qkv_self(x)
    assert torch.allclose(y_before, y_after, atol=1e-5), (y_before - y_after).abs().max()


def test_freeze_backbone_keeps_lora_trainable():
    from models.model_plain import freeze_backbone

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _MiniAttention()
            # simulate fusion adapter on model root
            self.fusion_adapter = nn.Linear(8, 3)

    m = _Wrapper()
    inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    freeze_backbone(m)

    # Base weights of LoRA-wrapped layers frozen
    assert m.attn.qkv_self.base.weight.requires_grad is False
    # LoRA adapters trainable
    assert m.attn.qkv_self.lora_A.weight.requires_grad is True
    assert m.attn.qkv_self.lora_B.weight.requires_grad is True
    # Fusion adapter stays trainable
    assert m.fusion_adapter.weight.requires_grad is True
    # Non-target Linear is frozen (backbone)
    assert m.attn.other_linear.weight.requires_grad is False
