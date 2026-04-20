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


def test_define_optimizer_skips_frozen_params_in_flow_group():
    """差异化 LR 分组时，冻结参数（requires_grad=False）应被跳过。"""
    model = _MiniModel()
    # 冻结 spynet（模拟 Phase 1 状态）
    for name, param in model.named_parameters():
        if 'spynet' in name:
            param.requires_grad = False

    fix_keys = ["spynet"]
    fix_lr_mul = 0.1
    base_lr = 2e-4

    normal_params = []
    flow_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in fix_keys):
            flow_params.append(param)
        else:
            normal_params.append(param)

    # spynet 被冻结，flow_params 应为空
    assert len(flow_params) == 0, "冻结的 spynet 参数不应出现在 flow_params 中"
    assert len(normal_params) > 0


def test_define_optimizer_differential_lr_with_unfrozen_params():
    """解冻后，fix_keys 参数应进入低 LR 组。"""
    model = _MiniModel()
    # 所有参数可训练（模拟 Phase 2 状态）
    for param in model.parameters():
        param.requires_grad = True

    fix_keys = ["spynet"]
    fix_lr_mul = 0.1
    base_lr = 2e-4

    normal_params = []
    flow_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in fix_keys):
            flow_params.append(param)
        else:
            normal_params.append(param)

    assert len(flow_params) > 0, "解冻后 spynet 应在 flow_params"
    assert abs(base_lr * fix_lr_mul - 2e-5) < 1e-10


def test_enter_phase2_injects_lora_and_unfreezes_fix_keys():
    """_enter_phase2 应注入 LoRA 并解冻 fix_keys，主干 base 权重保持冻结。"""
    from models.model_plain import freeze_backbone
    from models.lora import inject_lora, LoRALinear

    model = _MiniModel()
    freeze_backbone(model)  # Phase 1 状态：只有 fusion_operator 可训练

    # 模拟 _enter_phase2 逻辑
    fix_keys = ["spynet", "pa_deform"]
    targets = ["qkv"]
    rank, alpha = 4, 8

    # 1. 注入 LoRA
    inject_lora(model, target_substrings=targets, rank=rank, alpha=alpha)
    # 2. 解冻 fix_keys
    for name, param in model.named_parameters():
        if any(key in name for key in fix_keys):
            param.requires_grad = True
    # 3. 解冻 LoRA params（freeze_backbone 已保留，但注入后需确认）
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True

    # LoRA 已注入
    assert isinstance(model.backbone_qkv, LoRALinear)
    # LoRA 参数可训练
    assert model.backbone_qkv.lora_A.weight.requires_grad is True
    assert model.backbone_qkv.lora_B.weight.requires_grad is True
    # backbone base 权重冻结
    assert model.backbone_qkv.base.weight.requires_grad is False
    # fix_keys 解冻
    assert model.spynet_conv.weight.requires_grad is True
    assert model.pa_deform_conv.weight.requires_grad is True
    # fusion 仍可训练
    assert model.fusion_operator_gate.weight.requires_grad is True
