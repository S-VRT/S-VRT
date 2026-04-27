import math
import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lora import LoRALinear, inject_lora, merge_lora, merged_state_dict
from models.fusion.factory import create_fusion_operator


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


def test_inject_lora_skips_mamba_fusion_operator_proj_layers():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {
            "token_dim": 16,
            "token_stride": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "num_layers": 1,
        },
    )

    replaced = inject_lora(op, ["qkv", "proj"], rank=4, alpha=8)

    assert replaced == []
    block = op.mamba_token_mixer[0]
    if getattr(block, "mamba", None) is not None:
        assert isinstance(block.mamba.in_proj, nn.Linear)
        assert isinstance(block.mamba.out_proj, nn.Linear)
        assert isinstance(block.mamba.x_proj, nn.Linear)
        assert isinstance(block.mamba.dt_proj, nn.Linear)


def test_inject_lora_only_targets_vrt_attention_modules_in_mixed_model():
    class _MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _MiniAttention()
            self.fusion_operator = create_fusion_operator(
                "mamba",
                3,
                1,
                3,
                {
                    "token_dim": 16,
                    "token_stride": 2,
                    "d_state": 16,
                    "d_conv": 4,
                    "expand": 2,
                    "num_layers": 1,
                },
            )

    model = _MixedModel()

    replaced = inject_lora(model, ["qkv", "proj"], rank=4, alpha=8)

    assert sorted(replaced) == ["attn.proj", "attn.qkv_mut", "attn.qkv_self"]
    assert isinstance(model.attn.qkv_self, LoRALinear)
    assert isinstance(model.attn.qkv_mut, LoRALinear)
    assert isinstance(model.attn.proj, LoRALinear)


def test_inject_lora_skips_pa_deform_and_spynet_proj_layers():
    class _DcnStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.value_proj = nn.Linear(8, 8)
            self.output_proj = nn.Linear(8, 8)

    class _MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _MiniAttention()
            self.pa_deform = nn.Module()
            self.pa_deform.dcn = _DcnStub()
            self.spynet = nn.Module()
            self.spynet.proj = nn.Linear(8, 8)

    model = _MixedModel()

    replaced = inject_lora(model, ["qkv", "proj"], rank=4, alpha=8)

    assert sorted(replaced) == ["attn.proj", "attn.qkv_mut", "attn.qkv_self"]
    assert isinstance(model.pa_deform.dcn.value_proj, nn.Linear)
    assert isinstance(model.pa_deform.dcn.output_proj, nn.Linear)
    assert not isinstance(model.pa_deform.dcn.value_proj, LoRALinear)
    assert not isinstance(model.pa_deform.dcn.output_proj, LoRALinear)
    assert isinstance(model.spynet.proj, nn.Linear)
    assert not isinstance(model.spynet.proj, LoRALinear)


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


def test_init_train_injects_lora(monkeypatch, tmp_path):
    """When train.use_lora=True, init_train should inject LoRA adapters into netG."""
    from models.model_plain import ModelPlain
    from models.architectures.vrt.attention import WindowAttention

    opt = {
        "train": {
            "use_lora": True,
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_target_modules": ["qkv", "proj"],
            "freeze_backbone": True,
            "E_decay": 0,
            "G_lossfn_type": "l1",
            "G_lossfn_weight": 1.0,
            "G_optimizer_type": "adam",
            "G_optimizer_lr": 1e-4,
            "G_optimizer_betas": [0.9, 0.99],
            "G_optimizer_wd": 0,
            "G_optimizer_reuse": False,
            "G_optimizer_clipgrad": None,
            "G_scheduler_type": "CosineAnnealingWarmRestarts",
            "G_scheduler_periods": 100,
            "G_scheduler_restart_weights": 1,
            "G_scheduler_eta_min": 1e-7,
            "G_regularizer_orthstep": None,
            "G_regularizer_clipstep": None,
            "G_param_strict": False,
        },
        "path": {"pretrained_netG": None, "pretrained_netE": None,
                 "pretrained_optimizerG": None},
        "rank": 0,
        "dist": False,
    }

    model = ModelPlain.__new__(ModelPlain)
    model.opt = opt
    model.opt_train = opt["train"]
    model.device = torch.device("cpu")
    model.schedulers = []
    model.netG = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)

    class _DummyTimer:
        def timer(self, *a, **k):
            from contextlib import nullcontext
            return nullcontext()
    model.timer = _DummyTimer()

    # Stub out calls init_train makes that we don't want to exercise here
    monkeypatch.setattr(model, "load", lambda: None)
    monkeypatch.setattr(model, "load_optimizers", lambda: None)
    monkeypatch.setattr(model, "define_loss", lambda: None)
    monkeypatch.setattr(model, "define_scheduler", lambda: None)
    monkeypatch.setattr(model, "get_bare_model", lambda net: net)

    model.init_train()

    assert isinstance(model.netG.qkv_self, LoRALinear)
    assert isinstance(model.netG.proj, LoRALinear)
    assert model.netG.qkv_self.base.weight.requires_grad is False
    assert model.netG.qkv_self.lora_A.weight.requires_grad is True


def test_init_train_strict_loads_lora_checkpoint_after_injection(monkeypatch, tmp_path):
    """LoRA checkpoints saved during training must strict-load on resume."""
    from models.model_plain import ModelPlain
    from models.architectures.vrt.attention import WindowAttention

    source = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)
    inject_lora(source, ["qkv", "proj"], rank=2, alpha=4)
    with torch.no_grad():
        source.qkv_self.lora_A.weight.fill_(0.25)
        source.qkv_self.lora_B.weight.fill_(0.5)
    ckpt_path = tmp_path / "10_G.pth"
    torch.save({"params": source.state_dict()}, ckpt_path)

    opt = {
        "train": {
            "use_lora": True,
            "lora_rank": 2,
            "lora_alpha": 4,
            "lora_target_modules": ["qkv", "proj"],
            "freeze_backbone": False,
            "E_decay": 0,
            "G_optimizer_reuse": False,
            "G_param_strict": True,
        },
        "path": {
            "pretrained_netG": str(ckpt_path),
            "pretrained_netE": None,
            "pretrained_optimizerG": None,
        },
        "rank": 0,
        "dist": False,
    }

    model = ModelPlain.__new__(ModelPlain)
    model.opt = opt
    model.opt_train = opt["train"]
    model.device = torch.device("cpu")
    model.schedulers = []
    model.netG = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)

    monkeypatch.setattr(model, "define_loss", lambda: None)
    monkeypatch.setattr(model, "define_optimizer", lambda: None)
    monkeypatch.setattr(model, "load_optimizers", lambda: None)
    monkeypatch.setattr(model, "define_scheduler", lambda: None)
    monkeypatch.setattr(model, "get_bare_model", lambda net: net)

    model.init_train()

    assert isinstance(model.netG.qkv_self, LoRALinear)
    assert torch.allclose(model.netG.qkv_self.lora_A.weight, source.qkv_self.lora_A.weight)
    assert torch.allclose(model.netG.qkv_self.lora_B.weight, source.qkv_self.lora_B.weight)


def test_init_train_strict_loads_plain_checkpoint_before_lora_injection(monkeypatch, tmp_path):
    """Plain pretrained checkpoints should still load before LoRA wrapping."""
    from models.model_plain import ModelPlain
    from models.architectures.vrt.attention import WindowAttention

    source = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)
    with torch.no_grad():
        source.qkv_self.weight.fill_(0.125)
    ckpt_path = tmp_path / "plain_G.pth"
    torch.save({"params": source.state_dict()}, ckpt_path)

    opt = {
        "train": {
            "use_lora": True,
            "lora_rank": 2,
            "lora_alpha": 4,
            "lora_target_modules": ["qkv", "proj"],
            "freeze_backbone": False,
            "E_decay": 0,
            "G_optimizer_reuse": False,
            "G_param_strict": True,
        },
        "path": {
            "pretrained_netG": str(ckpt_path),
            "pretrained_netE": None,
            "pretrained_optimizerG": None,
        },
        "rank": 0,
        "dist": False,
    }

    model = ModelPlain.__new__(ModelPlain)
    model.opt = opt
    model.opt_train = opt["train"]
    model.device = torch.device("cpu")
    model.schedulers = []
    model.netG = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)

    monkeypatch.setattr(model, "define_loss", lambda: None)
    monkeypatch.setattr(model, "define_optimizer", lambda: None)
    monkeypatch.setattr(model, "load_optimizers", lambda: None)
    monkeypatch.setattr(model, "define_scheduler", lambda: None)
    monkeypatch.setattr(model, "get_bare_model", lambda net: net)

    model.init_train()

    assert isinstance(model.netG.qkv_self, LoRALinear)
    assert torch.allclose(model.netG.qkv_self.base.weight, source.qkv_self.weight)


def test_save_merged_writes_fused_ckpt(tmp_path):
    from models.model_plain import ModelPlain
    from models.architectures.vrt.attention import WindowAttention

    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"train": {"use_lora": True, "E_decay": 0}, "rank": 0, "dist": False}
    model.opt_train = model.opt["train"]
    model.save_dir = str(tmp_path)
    net = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)
    inject_lora(net, ["qkv", "proj"], rank=4, alpha=8)
    # Perturb LoRA so merge is non-trivial
    for mod in net.modules():
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_A.weight, std=0.02)
            nn.init.normal_(mod.lora_B.weight, std=0.02)
    model.netG = net

    # Reference structure without LoRA
    ref = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)
    ref_keys = set(ref.state_dict().keys())

    model.save_merged(iter_label=12345)

    merged_path = tmp_path / "12345_G_merged.pth"
    assert merged_path.exists()
    sd = torch.load(merged_path, map_location="cpu", weights_only=True)
    assert set(sd.keys()) == ref_keys
    # netG in memory untouched (still has LoRA)
    assert isinstance(model.netG.qkv_self, LoRALinear)


def test_save_merged_noop_without_lora(tmp_path):
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"train": {"use_lora": False, "E_decay": 0}, "rank": 0, "dist": False}
    model.opt_train = model.opt["train"]
    model.save_dir = str(tmp_path)
    model.netG = nn.Linear(4, 4)

    model.save_merged(iter_label=1)

    assert list(tmp_path.iterdir()) == []


def test_merged_state_dict_avoids_deepcopy_for_non_leaf_runtime_tensor():
    class _RuntimeTensorWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _MiniAttention()
            self.runtime_tensor = None

        def forward(self, x):
            y = self.attn.qkv_self(x)
            self.runtime_tensor = y + 1
            return y

    torch.manual_seed(0)
    model = _RuntimeTensorWrapper()
    inject_lora(model, ["qkv", "proj"], rank=4, alpha=8)
    x = torch.randn(2, 8, requires_grad=True)
    model(x)

    with pytest.raises(RuntimeError, match="deepcopy protocol"):
        copy.deepcopy(model)

    exported = merged_state_dict(model)
    reference = _RuntimeTensorWrapper().state_dict()
    assert set(exported.keys()) == set(reference.keys()) - {"runtime_tensor"}
    assert all(".lora_A." not in key and ".lora_B." not in key for key in exported)


# ============================================================================
# Extended acceptance tests
# ============================================================================


# ---------- LoRALinear mathematical correctness ----------


def test_lora_rejects_non_positive_rank():
    base = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        LoRALinear(base, rank=0, alpha=8)
    with pytest.raises(ValueError):
        LoRALinear(base, rank=-1, alpha=8)


def test_lora_scaling_value_matches_alpha_over_rank():
    lora = LoRALinear(nn.Linear(4, 4), rank=8, alpha=16)
    assert lora.scaling == pytest.approx(2.0)
    lora2 = LoRALinear(nn.Linear(4, 4), rank=4, alpha=4)
    assert lora2.scaling == pytest.approx(1.0)


def test_lora_preserves_base_bias_in_forward():
    torch.manual_seed(0)
    base = nn.Linear(6, 10, bias=True)
    lora = LoRALinear(copy.deepcopy(base), rank=2, alpha=4)
    x = torch.randn(3, 6)
    # With B=0, output must equal base(x) including bias term
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_non_square_shapes():
    torch.manual_seed(0)
    base = nn.Linear(7, 13)
    lora = LoRALinear(copy.deepcopy(base), rank=3, alpha=6)
    assert lora.lora_A.weight.shape == (3, 7)
    assert lora.lora_B.weight.shape == (13, 3)
    x = torch.randn(4, 7)
    assert lora(x).shape == (4, 13)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_forward_handles_3d_batches():
    """Attention inputs arrive as [B_, N, C]; LoRA must broadcast correctly."""
    torch.manual_seed(0)
    base = nn.Linear(8, 24)
    lora = LoRALinear(copy.deepcopy(base), rank=4, alpha=8)
    nn.init.normal_(lora.lora_A.weight, std=0.01)
    nn.init.normal_(lora.lora_B.weight, std=0.01)
    x = torch.randn(5, 11, 8)  # batched sequence of tokens
    y = lora(x)
    assert y.shape == (5, 11, 24)
    # Manual compute matches
    expected = base(x) + (lora.lora_B(lora.lora_A(x)) * lora.scaling)
    assert torch.allclose(y, expected, atol=1e-6)


def test_lora_exposes_linear_weight_for_direct_matmul_callers():
    torch.manual_seed(0)
    base = nn.Linear(8, 12)
    lora = LoRALinear(copy.deepcopy(base), rank=4, alpha=8)
    nn.init.normal_(lora.lora_A.weight, std=0.01)
    nn.init.normal_(lora.lora_B.weight, std=0.01)

    x = torch.randn(3, 8)
    expected = lora(x)
    actual = x @ lora.weight.t()
    if lora.bias is not None:
        actual = actual + lora.bias

    assert torch.allclose(actual, expected, atol=1e-6)


def test_lora_dtype_preserved():
    base = nn.Linear(8, 8).to(torch.float64)
    lora = LoRALinear(base, rank=4, alpha=8)
    # Match base dtype explicitly (factory does this in inject_lora)
    lora = lora.to(torch.float64)
    x = torch.randn(2, 8, dtype=torch.float64)
    y = lora(x)
    assert y.dtype == torch.float64


def test_lora_gradient_only_flows_through_adapters():
    """With base frozen, only lora_A/B should accumulate gradients."""
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    lora = LoRALinear(copy.deepcopy(base), rank=4, alpha=8)
    # Perturb B so gradient path through LoRA is non-zero
    nn.init.normal_(lora.lora_B.weight, std=0.02)
    for p in lora.base.parameters():
        p.requires_grad = False

    x = torch.randn(2, 8)
    target = torch.randn(2, 8)
    loss = F.mse_loss(lora(x), target)
    loss.backward()

    assert lora.base.weight.grad is None
    assert lora.base.bias.grad is None
    assert lora.lora_A.weight.grad is not None
    assert lora.lora_B.weight.grad is not None
    assert lora.lora_A.weight.grad.abs().sum().item() > 0
    assert lora.lora_B.weight.grad.abs().sum().item() > 0


# ---------- inject_lora edge cases ----------


def test_inject_lora_empty_targets_is_noop():
    m = _MiniAttention()
    replaced = inject_lora(m, [], rank=4, alpha=8)
    assert replaced == []
    assert isinstance(m.qkv_self, nn.Linear)
    assert not isinstance(m.qkv_self, LoRALinear)


def test_inject_lora_no_matches():
    class _Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo = nn.Linear(4, 4)
            self.bar = nn.Linear(4, 4)

    m = _Plain()
    replaced = inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    assert replaced == []
    assert isinstance(m.foo, nn.Linear) and not isinstance(m.foo, LoRALinear)


def test_inject_lora_traverses_nested_modules():
    class _Block(nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.attn = _MiniAttention(dim)

    class _Net(nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.block0 = _Block(dim)
            self.block1 = _Block(dim)
            self.head = nn.Linear(dim, dim)

    m = _Net()
    replaced = inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    # Two blocks × (qkv_self + qkv_mut + proj) = 6 replacements
    assert len(replaced) == 6
    # Parent path preserved in dotted name
    assert any(r.startswith("block0.") for r in replaced)
    assert any(r.startswith("block1.") for r in replaced)
    assert isinstance(m.block0.attn.qkv_self, LoRALinear)
    assert isinstance(m.block1.attn.proj, LoRALinear)
    # head not touched (no "qkv"/"proj" substring)
    assert not isinstance(m.head, LoRALinear)


def test_inject_lora_preserves_base_weight_identity():
    """The original Linear weight tensor must remain as lora.base.weight
    (so pretrained weights aren't re-initialized when wrapping)."""
    m = _MiniAttention()
    original_weight_id = id(m.qkv_self.weight)
    original_weight_data = m.qkv_self.weight.data.clone()
    inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    assert id(m.qkv_self.base.weight) == original_weight_id
    assert torch.equal(m.qkv_self.base.weight.data, original_weight_data)


# ---------- merge_lora numerical correctness ----------


def test_merge_lora_exact_weight_equation():
    """Merged weight must equal base.weight + (alpha/rank) * B @ A — exactly."""
    torch.manual_seed(0)
    base = nn.Linear(5, 7, bias=True)
    base_w = base.weight.data.clone()
    base_b = base.bias.data.clone()
    lora = LoRALinear(copy.deepcopy(base), rank=2, alpha=4)
    nn.init.normal_(lora.lora_A.weight, std=0.3)
    nn.init.normal_(lora.lora_B.weight, std=0.3)

    fused = lora.merged_linear()
    expected_w = base_w + (lora.lora_B.weight @ lora.lora_A.weight) * lora.scaling
    assert torch.allclose(fused.weight.data, expected_w, atol=1e-7)
    # Bias should be unchanged (LoRA adds no bias)
    assert torch.allclose(fused.bias.data, base_b, atol=1e-7)


def test_merge_lora_noop_when_b_zero():
    """Freshly-injected LoRA (B=0) merges to the unchanged base weight."""
    torch.manual_seed(0)
    base = nn.Linear(6, 6)
    base_w = base.weight.data.clone()
    lora = LoRALinear(copy.deepcopy(base), rank=3, alpha=6)
    fused = lora.merged_linear()
    assert torch.allclose(fused.weight.data, base_w, atol=1e-7)


def test_merge_lora_preserves_bias_flag():
    base_no_bias = nn.Linear(4, 4, bias=False)
    lora = LoRALinear(base_no_bias, rank=2, alpha=4)
    fused = lora.merged_linear()
    assert fused.bias is None


def test_merge_lora_is_reinjectable():
    """After merging, the model should accept a fresh injection (for iterative training)."""
    m = _MiniAttention()
    inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    nn.init.normal_(m.qkv_self.lora_B.weight, std=0.02)
    merge_lora(m)
    assert isinstance(m.qkv_self, nn.Linear)
    # Re-inject; should wrap the merged Linear (which contains the prior delta)
    replaced = inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    assert len(replaced) == 3
    assert isinstance(m.qkv_self, LoRALinear)


# ---------- Gradient + optimizer integration ----------


def test_optimizer_only_receives_trainable_params_after_injection():
    """Verify the ModelPlain.define_optimizer filter interacts correctly with LoRA."""
    from models.model_plain import freeze_backbone

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _MiniAttention()
            self.fusion_adapter = nn.Linear(8, 3)

    m = _Wrapper()
    inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    freeze_backbone(m)

    trainable = [(k, v) for k, v in m.named_parameters() if v.requires_grad]
    trainable_names = {k for k, _ in trainable}

    # Expected trainable set: every lora_A/B weight, fusion_adapter, nothing else
    expected_prefixes = (
        "attn.qkv_self.lora_A", "attn.qkv_self.lora_B",
        "attn.qkv_mut.lora_A", "attn.qkv_mut.lora_B",
        "attn.proj.lora_A", "attn.proj.lora_B",
        "fusion_adapter.",
    )
    for name in trainable_names:
        assert name.startswith(expected_prefixes), f"Unexpected trainable: {name}"
    # And the backbone Linear base weights must NOT appear
    for name in trainable_names:
        assert ".base.weight" not in name
        assert ".base.bias" not in name
    # other_linear is backbone → frozen
    assert "attn.other_linear.weight" not in trainable_names


def test_stage_a_freeze_backbone_unchanged_without_lora():
    """Regression: freeze_backbone without LoRA still behaves the same as before."""
    from models.model_plain import freeze_backbone

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.fusion_adapter = nn.Linear(4, 3)
            self.fusion_operator = nn.Linear(3, 3)

    m = _Wrapper()
    freeze_backbone(m)
    assert m.backbone.weight.requires_grad is False
    assert m.fusion_adapter.weight.requires_grad is True
    assert m.fusion_operator.weight.requires_grad is True


# ---------- End-to-end on real VRT attention ----------


def test_merge_lora_preserves_vrt_attention_forward():
    """On real VRT WindowAttention, forward before/after merge must agree."""
    from models.architectures.vrt.attention import WindowAttention

    torch.manual_seed(0)
    dim, num_heads = 8, 2
    window = (2, 2, 2)
    attn = WindowAttention(dim=dim, window_size=window, num_heads=num_heads, mut_attn=True)
    attn.eval()
    inject_lora(attn, ["qkv", "proj"], rank=2, alpha=4)
    for mod in attn.modules():
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_A.weight, std=0.02)
            nn.init.normal_(mod.lora_B.weight, std=0.02)

    N = window[0] * window[1] * window[2]
    x = torch.randn(3, N, dim)
    with torch.no_grad():
        y_before = attn(x, mask=None)
    merged = copy.deepcopy(attn)
    merge_lora(merged)
    with torch.no_grad():
        y_after = merged(x, mask=None)
    assert torch.allclose(y_before, y_after, atol=1e-5), (y_before - y_after).abs().max()


def test_merged_ckpt_loads_into_clean_vrt_attention_strict():
    """Merged state_dict must load strict=True into a fresh non-LoRA WindowAttention."""
    from models.architectures.vrt.attention import WindowAttention

    torch.manual_seed(0)
    dim, num_heads = 8, 2
    window = (2, 2, 2)
    attn = WindowAttention(dim=dim, window_size=window, num_heads=num_heads, mut_attn=True)
    inject_lora(attn, ["qkv", "proj"], rank=2, alpha=4)
    for mod in attn.modules():
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_A.weight, std=0.02)
            nn.init.normal_(mod.lora_B.weight, std=0.02)
    merged = copy.deepcopy(attn)
    merge_lora(merged)
    sd = merged.state_dict()

    clean = WindowAttention(dim=dim, window_size=window, num_heads=num_heads, mut_attn=True)
    missing, unexpected = clean.load_state_dict(sd, strict=True)
    assert missing == [] and unexpected == []


# ---------- save_merged additional scenarios ----------


def test_save_merged_non_rank0_is_noop(tmp_path):
    from models.model_plain import ModelPlain
    from models.architectures.vrt.attention import WindowAttention

    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"train": {"use_lora": True, "E_decay": 0}, "rank": 1, "dist": True}
    model.opt_train = model.opt["train"]
    model.save_dir = str(tmp_path)
    net = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)
    inject_lora(net, ["qkv", "proj"], rank=2, alpha=4)
    model.netG = net

    model.save_merged(iter_label=99)
    assert list(tmp_path.iterdir()) == []


def test_save_merged_writes_E_branch(tmp_path):
    """When E_decay > 0, save_merged should also emit {iter}_E_merged.pth."""
    from models.model_plain import ModelPlain
    from models.architectures.vrt.attention import WindowAttention

    model = ModelPlain.__new__(ModelPlain)
    model.opt = {
        "train": {"use_lora": True, "E_decay": 0.999},
        "rank": 0, "dist": False,
    }
    model.opt_train = model.opt["train"]
    model.save_dir = str(tmp_path)

    def _make_net():
        net = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)
        inject_lora(net, ["qkv", "proj"], rank=2, alpha=4)
        for mod in net.modules():
            if isinstance(mod, LoRALinear):
                nn.init.normal_(mod.lora_B.weight, std=0.01)
        return net

    model.netG = _make_net()
    model.netE = _make_net()

    model.save_merged(iter_label=42)
    assert (tmp_path / "42_G_merged.pth").exists()
    assert (tmp_path / "42_E_merged.pth").exists()
    sd_e = torch.load(tmp_path / "42_E_merged.pth", map_location="cpu", weights_only=True)
    # Must be structurally a plain VRT attention (no lora_A/B keys)
    assert not any("lora_A" in k or "lora_B" in k for k in sd_e.keys())


def test_save_merged_in_memory_model_untouched(tmp_path):
    """save_merged must not mutate the in-memory model (training continues after)."""
    from models.model_plain import ModelPlain
    from models.architectures.vrt.attention import WindowAttention

    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"train": {"use_lora": True, "E_decay": 0}, "rank": 0, "dist": False}
    model.opt_train = model.opt["train"]
    model.save_dir = str(tmp_path)

    net = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)
    inject_lora(net, ["qkv", "proj"], rank=2, alpha=4)
    pre_A = net.qkv_self.lora_A.weight.data.clone()
    pre_B = net.qkv_self.lora_B.weight.data.clone()
    model.netG = net

    model.save_merged(iter_label=1)

    # netG still has LoRA wrappers with original adapter weights
    assert isinstance(model.netG.qkv_self, LoRALinear)
    assert torch.equal(model.netG.qkv_self.lora_A.weight.data, pre_A)
    assert torch.equal(model.netG.qkv_self.lora_B.weight.data, pre_B)


# ---------- Config-driven sanity ----------


def test_inject_lora_respects_rank_in_param_shapes():
    """Parameter count of A/B must match the declared rank precisely."""
    m = _MiniAttention(dim=8)
    inject_lora(m, ["qkv", "proj"], rank=5, alpha=10)
    # qkv_self: in=8, out=24
    assert m.qkv_self.lora_A.weight.shape == (5, 8)
    assert m.qkv_self.lora_B.weight.shape == (24, 5)
    # proj: in=8, out=8
    assert m.proj.lora_A.weight.shape == (5, 8)
    assert m.proj.lora_B.weight.shape == (8, 5)
