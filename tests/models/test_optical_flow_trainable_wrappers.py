import torch
import torch.nn as nn

from models.optical_flow.scflow.wrapper import SCFlowWrapper
from models.optical_flow.spynet import SpyNetWrapper


class _TinySCFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, spk1, spk2, flow_init, dt=10):
        del flow_init, dt
        base = (spk1 - spk2).mean(dim=1, keepdim=True) * self.scale
        flow = base.repeat(1, 2, 1, 1)
        flows = [flow, flow[:, :, ::2, ::2], flow[:, :, ::4, ::4], flow[:, :, ::8, ::8]]
        return flows, {}


class _TinySpyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, ref, supp):
        base = (ref - supp).mean(dim=1, keepdim=True) * self.scale
        return base.repeat(1, 2, 1, 1)


def _make_scflow_wrapper():
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    nn.Module.__init__(wrapper)
    wrapper.input_type = "spike"
    wrapper.device = torch.device("cpu")
    wrapper.dt = 10
    wrapper.model = _TinySCFlow()
    return wrapper


def _make_spynet_wrapper():
    wrapper = SpyNetWrapper.__new__(SpyNetWrapper)
    nn.Module.__init__(wrapper)
    wrapper.input_type = "rgb"
    wrapper.device = torch.device("cpu")
    wrapper.return_levels = [5]
    wrapper.model = _TinySpyNet()
    return wrapper


def test_scflow_wrapper_tracks_gradients_when_phase2_unfreezes_flow():
    wrapper = _make_scflow_wrapper()
    spk1 = torch.randn(1, 25, 16, 16)
    spk2 = torch.randn(1, 25, 16, 16)

    for param in wrapper.parameters():
        param.requires_grad_(False)
    frozen_flow = wrapper(spk1, spk2)[0]
    assert frozen_flow.requires_grad is False

    for param in wrapper.parameters():
        param.requires_grad_(True)
    trainable_flow = wrapper(spk1, spk2)[0]
    assert trainable_flow.requires_grad is True

    trainable_flow.mean().backward()
    assert wrapper.model.scale.grad is not None


def test_spynet_wrapper_tracks_gradients_when_phase2_unfreezes_flow():
    wrapper = _make_spynet_wrapper()
    frame1 = torch.randn(1, 3, 16, 16)
    frame2 = torch.randn(1, 3, 16, 16)

    for param in wrapper.parameters():
        param.requires_grad_(False)
    frozen_flow = wrapper(frame1, frame2)
    assert frozen_flow.requires_grad is False

    for param in wrapper.parameters():
        param.requires_grad_(True)
    trainable_flow = wrapper(frame1, frame2)
    assert trainable_flow.requires_grad is True

    trainable_flow.mean().backward()
    assert wrapper.model.scale.grad is not None
