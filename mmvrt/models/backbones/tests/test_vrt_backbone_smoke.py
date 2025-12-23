"""Smoke tests for VRTBackbone wrapper focusing on channel adaptation and forward."""
import torch
import torch.nn as nn

from mmvrt.models.backbones.vrt_backbone import VRTBackbone


class DummyNet(nn.Module):
    """Minimal stand-in for the legacy VRT network to test wrapper behavior."""

    def __init__(self, in_channels: int):
        super().__init__()
        # mimic conv_first attribute used by wrapper to determine expected channels
        class _C:
            pass

        self.conv_first = _C()
        setattr(self.conv_first, "in_channels", in_channels)

    def forward(self, x):
        # return input directly to make assertions easier
        return x


def test_vrt_backbone_padding_and_slicing():
    # Create uninitialized VRTBackbone instance (avoid heavy legacy init)
    bb = VRTBackbone.__new__(VRTBackbone)

    # Case A: expected channels larger than input -> padding
    expected_c = 12
    bb.net = DummyNet(in_channels=expected_c)
    x = torch.rand((1, 5, 3, 16, 16))  # C=3 < expected_c -> should be padded to 12
    out = bb.forward(x)
    assert out.shape[2] == expected_c

    # Case B: input channels greater than expected -> slicing
    expected_c2 = 6
    bb.net = DummyNet(in_channels=expected_c2)
    x2 = torch.rand((1, 5, 10, 16, 16))  # C=10 > expected_c2 -> should be sliced to 6
    out2 = bb.forward(x2)
    assert out2.shape[2] == expected_c2


if __name__ == "__main__":
    test_vrt_backbone_padding_and_slicing()
    print("VRTBackbone smoke tests passed")


