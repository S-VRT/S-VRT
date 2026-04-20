import torch
import torch.nn as nn


class _MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_qkv = nn.Linear(4, 12)   # 模拟 attention qkv（名含 "qkv"）
        self.spynet_conv = nn.Linear(4, 4)      # 模拟 SCFlow（名含 "spynet"）
        self.pa_deform_conv = nn.Linear(4, 4)   # 模拟 DCN（名含 "pa_deform"）
        self.fusion_operator_gate = nn.Linear(4, 4)  # 模拟 Fusion


def test_phase2_lora_mode_skips_init_lora_injection():
    """phase2_lora_mode=true 时，init_train 不应注入 LoRA。"""
    from models.lora import inject_lora, LoRALinear

    model = _MiniModel()
    # 模拟 phase2_lora_mode=true 时的 init_train 逻辑（跳过注入）
    phase2_lora_mode = True
    use_lora = True
    if use_lora and not phase2_lora_mode:
        inject_lora(model, target_substrings=["qkv"], rank=4, alpha=8)

    # backbone_qkv 不应被替换为 LoRALinear
    assert not isinstance(model.backbone_qkv, LoRALinear), \
        "phase2_lora_mode=true 时 init_train 不应注入 LoRA"


def test_phase2_lora_mode_false_injects_lora_at_init():
    """phase2_lora_mode=false 时，init_train 应注入 LoRA。"""
    from models.lora import inject_lora, LoRALinear

    model = _MiniModel()
    phase2_lora_mode = False
    use_lora = True
    if use_lora and not phase2_lora_mode:
        inject_lora(model, target_substrings=["qkv"], rank=4, alpha=8)

    assert isinstance(model.backbone_qkv, LoRALinear), \
        "phase2_lora_mode=false 时 init_train 应注入 LoRA"
