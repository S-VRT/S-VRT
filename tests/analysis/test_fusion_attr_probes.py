import torch
from torch import nn

from scripts.analysis.fusion_attr.probes import FusionProbe, find_fusion_adapter, reduce_operator_explanations
from models.fusion.operators.gated import GatedFusionOperator


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_adapter = nn.Identity()

    def forward(self, x):
        return self.fusion_adapter(x)


def test_find_fusion_adapter_prefers_named_attribute():
    net = TinyNet()
    assert find_fusion_adapter(net) is net.fusion_adapter


def test_fusion_probe_captures_inputs_and_output():
    module = nn.Identity()
    probe = FusionProbe(module)
    probe.attach()
    x = torch.randn(1, 2, 3, 4, 4)
    y = module(x)
    probe.close()
    record = probe.record
    assert record is not None
    assert torch.equal(record.output, y)
    assert torch.equal(record.inputs[0], x)
    assert record.module_name == "Identity"


def test_gated_operator_explain_exports_effective_update():
    op = GatedFusionOperator(rgb_chans=3, spike_chans=2, out_chans=3, operator_params={"hidden_chans": 4})
    rgb = torch.randn(1, 3, 6, 6)
    spike = torch.randn(1, 2, 6, 6)
    _ = op(rgb, spike)
    maps = op.explain()
    assert set(maps) == {"gate", "correction", "effective_update"}
    assert maps["gate"].shape == rgb.shape
    assert maps["effective_update"].shape == rgb.shape


def test_reduce_operator_explanations_converts_tensors_to_2d_maps():
    explanations = {
        "gate": torch.ones(1, 3, 4, 5),
        "effective_update": torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5),
    }
    reduced = reduce_operator_explanations(explanations)
    assert reduced["gate_mean"].shape == (4, 5)
    assert reduced["effective_update"].shape == (4, 5)
    assert reduced["gate_mean"].max().item() == 1.0


from models.fusion.operators.mamba import MambaFusionOperator


def test_mamba_operator_explain_exports_effective_update_without_ssm_dependency():
    op = MambaFusionOperator(
        rgb_chans=3,
        spike_chans=1,
        out_chans=3,
        operator_params={
            "token_dim": 8,
            "token_stride": 2,
            "num_layers": 0,
            "enable_diagnostics": True,
        },
    )
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)

    _ = op(rgb, spike)

    maps = op.explain()
    assert set(maps) >= {"gate", "delta", "effective_update", "token_energy"}
    assert maps["effective_update"].shape == rgb.shape
    assert maps["gate"].shape == rgb.shape


def test_reduce_operator_explanations_accepts_mamba_specific_maps():
    explanations = {
        "gate": torch.ones(1, 2, 3, 4, 5),
        "delta": torch.ones(1, 2, 3, 4, 5) * 2,
        "effective_update": torch.ones(1, 2, 3, 4, 5) * 3,
        "token_energy": torch.ones(1, 2, 1, 4, 5) * 4,
    }

    reduced = reduce_operator_explanations(explanations)

    assert reduced["gate_mean"].shape == (4, 5)
    assert reduced["delta"].shape == (4, 5)
    assert reduced["effective_update"].shape == (4, 5)
    assert reduced["token_energy"].shape == (4, 5)
