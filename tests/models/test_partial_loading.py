import tempfile

import pytest
import torch
import torch.nn as nn

from models.model_base import ModelBase
from models.model_plain import freeze_backbone
from models.architectures.vrt.vrt import VRT


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)


class _DummyModelBase(ModelBase):
    def __init__(self):
        opt = {
            "path": {"models": "."},
            "dist": False,
            "is_train": False,
        }
        super().__init__(opt)


def test_load_network_partial_matches_key_and_shape():
    model_base = _DummyModelBase()
    net = _TinyNet()
    original_head_weight = net.head.weight.detach().clone()

    checkpoint = {
        "params": {
            "linear.weight": torch.full_like(net.linear.weight, 3.0),
            "linear.bias": torch.full_like(net.linear.bias, 2.0),
            "head.weight": torch.randn(3, 4),  # shape mismatch, should be skipped
            "missing.weight": torch.randn(1),
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        torch.save(checkpoint, tmp.name)
        model_base.load_network_partial(tmp.name, net, param_key="params")

    assert torch.allclose(net.linear.weight, torch.full_like(net.linear.weight, 3.0))
    assert torch.allclose(net.linear.bias, torch.full_like(net.linear.bias, 2.0))
    assert torch.allclose(net.head.weight, original_head_weight)


def test_freeze_backbone_freezes_non_fusion_params():
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 7,
            },
            "output_mode": "restoration",
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            },
        },
        "train": {
            "freeze_backbone": True,
        },
    }
    model = VRT(
        upscale=1,
        in_chans=7,
        out_chans=3,
        img_size=[2, 8, 8],
        window_size=[2, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=opt,
    )

    freeze_backbone(model)

    for name, param in model.named_parameters():
        if "fusion_adapter" in name or "fusion_operator" in name:
            assert param.requires_grad, f"Fusion param {name} should be trainable"
        else:
            assert not param.requires_grad, f"Backbone param {name} should be frozen"
