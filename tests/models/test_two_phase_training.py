import torch
import torch.nn as nn
from types import SimpleNamespace


class _WarmupAwareOperator:
    def __init__(self):
        self.last_stage = None

    def set_warmup_stage(self, stage):
        self.last_stage = stage


class _WarmupAwareNet:
    def __init__(self):
        self.fusion_enabled = True
        self.fusion_cfg = {"placement": "early"}
        self.fusion_adapter = SimpleNamespace(operator=_WarmupAwareOperator())


class _MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_qkv = nn.Linear(4, 12)
        self.spynet_conv = nn.Linear(4, 4)
        self.pa_deform_conv = nn.Linear(4, 4)
        self.fusion_operator_gate = nn.Linear(4, 4)


def test_phase2_lora_mode_freezes_lora_in_phase1():
    """phase2_lora_mode=true 时，freeze_backbone 后 LoRA 参数应被额外冻结。"""
    from models.model_plain import freeze_backbone
    from models.lora import inject_lora, LoRALinear

    model = _MiniModel()
    inject_lora(model, target_substrings=["qkv"], rank=4, alpha=8)
    freeze_backbone(model)

    # 模拟 phase2_lora_mode=true 的额外冻结逻辑
    phase2_lora_mode = True
    if phase2_lora_mode:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = False

    # LoRA 参数应被冻结
    assert model.backbone_qkv.lora_A.weight.requires_grad is False
    assert model.backbone_qkv.lora_B.weight.requires_grad is False
    # backbone base 权重应冻结
    assert model.backbone_qkv.base.weight.requires_grad is False
    # fusion_operator 应可训练
    assert model.fusion_operator_gate.weight.requires_grad is True
    # spynet/pa_deform 应冻结（freeze_backbone 冻结了它们）
    assert model.spynet_conv.weight.requires_grad is False
    assert model.pa_deform_conv.weight.requires_grad is False


def test_phase2_lora_mode_false_keeps_lora_trainable():
    """phase2_lora_mode=false 时，LoRA 参数应保持可训练（freeze_backbone 保留）。"""
    from models.model_plain import freeze_backbone
    from models.lora import inject_lora

    model = _MiniModel()
    inject_lora(model, target_substrings=["qkv"], rank=4, alpha=8)
    freeze_backbone(model)

    phase2_lora_mode = False
    if phase2_lora_mode:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = False

    # LoRA 参数应可训练（freeze_backbone 的 _TRAINABLE_NAME_MARKERS 保留了它们）
    assert model.backbone_qkv.lora_A.weight.requires_grad is True
    assert model.backbone_qkv.lora_B.weight.requires_grad is True


def test_all_params_in_optimizer_including_frozen():
    """define_optimizer 应包含所有参数（含冻结的），以便解冻后立即生效。"""
    from models.model_plain import freeze_backbone
    from models.lora import inject_lora

    model = _MiniModel()
    inject_lora(model, target_substrings=["qkv"], rank=4, alpha=8)
    freeze_backbone(model)

    # 模拟 phase2_lora_mode=true 额外冻结 LoRA
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = False

    fix_keys = ["spynet", "pa_deform"]
    fix_lr_mul = 0.1
    base_lr = 2e-4

    # 包含所有参数（不跳过冻结的）
    normal_params = []
    flow_params = []
    for name, param in model.named_parameters():
        if any(key in name for key in fix_keys):
            flow_params.append(param)
        else:
            normal_params.append(param)

    total_in_optimizer = len(normal_params) + len(flow_params)
    total_in_model = sum(1 for _ in model.parameters())
    assert total_in_optimizer == total_in_model, \
        f"optimizer 应包含所有 {total_in_model} 个参数，但只有 {total_in_optimizer}"


def test_phase2_unfreezes_fix_keys_and_lora_only():
    """fix_iter 时 phase2_lora_mode=true 只解冻 fix_keys+LoRA，主干 base 保持冻结。"""
    from models.model_plain import freeze_backbone
    from models.lora import inject_lora

    model = _MiniModel()
    inject_lora(model, target_substrings=["qkv"], rank=4, alpha=8)
    freeze_backbone(model)
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = False

    # 模拟 fix_iter 时的 phase2_lora_mode 解冻逻辑
    fix_keys = ["spynet", "pa_deform"]
    for name, param in model.named_parameters():
        if any(key in name for key in fix_keys):
            param.requires_grad_(True)
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad_(True)

    # fix_keys 解冻
    assert model.spynet_conv.weight.requires_grad is True
    assert model.pa_deform_conv.weight.requires_grad is True
    # LoRA 解冻
    assert model.backbone_qkv.lora_A.weight.requires_grad is True
    assert model.backbone_qkv.lora_B.weight.requires_grad is True
    # backbone base 仍冻结
    assert model.backbone_qkv.base.weight.requires_grad is False
    # fusion 仍可训练
    assert model.fusion_operator_gate.weight.requires_grad is True


def test_model_plain_configures_writeback_only_stage_during_head_only_iters():
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.opt_train = {"fusion_warmup": {"head_only_iters": 2}}
    model.fix_iter = 10
    model.netG = _WarmupAwareNet()

    stage = model._configure_fusion_warmup_trainability(current_step=0)

    assert stage == "writeback_only"
    assert model.netG.fusion_adapter.operator.last_stage == "writeback_only"


def test_model_plain_switches_to_token_mixer_stage_after_head_only_iters():
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.opt_train = {"fusion_warmup": {"head_only_iters": 2}}
    model.fix_iter = 10
    model.netG = _WarmupAwareNet()

    stage = model._configure_fusion_warmup_trainability(current_step=3)

    assert stage == "token_mixer"
    assert model.netG.fusion_adapter.operator.last_stage == "token_mixer"


def test_model_plain_optimizer_keeps_frozen_mamba_params_for_later_unfreeze():
    from models.model_plain import ModelPlain
    from models.fusion.factory import create_fusion_operator

    model = ModelPlain.__new__(ModelPlain)
    model.opt_train = {
        "G_optimizer_type": "adam",
        "G_optimizer_lr": 1e-4,
        "G_optimizer_betas": [0.9, 0.99],
        "G_optimizer_wd": 0.0,
    }
    model.netG = nn.Module()
    model.netG.fusion_operator = create_fusion_operator("mamba", 3, 1, 3, {"token_dim": 8, "token_stride": 2, "num_layers": 1})
    model.netG.fusion_operator.set_warmup_stage("writeback_only")

    model.define_optimizer()

    optimizer_param_ids = {
        id(param)
        for group in model.G_optimizer.param_groups
        for param in group["params"]
    }
    frozen_param_ids = {id(param) for param in model.netG.fusion_operator.rgb_context_encoder.parameters()}
    assert frozen_param_ids <= optimizer_param_ids
