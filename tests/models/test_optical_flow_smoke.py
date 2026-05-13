import torch
from torch import nn

from models.optical_flow import create_optical_flow
from models.optical_flow.spynet import SpyNetWrapper


def _make_frames(device='cpu', h=64, w=64):
    # RAFT expects 0-255 image range; use floats to be safe
    frame1 = (torch.rand(1, 3, h, w, device=device) * 255.0).float()
    frame2 = (torch.rand(1, 3, h, w, device=device) * 255.0).float()
    return frame1, frame2


def test_optical_flow_backends_forward_shapes_cpu():
    device = 'cpu'
    frame1, frame2 = _make_frames(device=device)

    for backend in ('spynet', 'sea_raft'):
        of = create_optical_flow(module=backend, checkpoint=None, device=device)
        out = of(frame1, frame2)
        # The optical-flow stack accepts either a multiscale list/tuple or a single final-flow tensor.
        if isinstance(out, (list, tuple)):
            final = out[-1] if len(out) > 1 else out[0]
        else:
            final = out
        assert final.shape[0] == 1 and final.shape[1] == 2
        assert torch.isfinite(final).all()


def test_spynet_wrapper_preserves_vrt_rgb_input_contract():
    """SpyNetWrapper should be a thin adapter for VRT RGB [0,1] tensors."""
    torch.manual_seed(7)
    wrapper = SpyNetWrapper(checkpoint=None, device='cpu', return_levels=[2, 3, 4, 5])
    wrapper.eval()
    frame1 = torch.rand(2, 3, 64, 64)
    frame2 = torch.rand(2, 3, 64, 64)

    with torch.no_grad():
        expected = wrapper.model(frame1, frame2)
        actual = wrapper(frame1, frame2)

    max_abs = max(float((exp - got).abs().max()) for exp, got in zip(expected, actual))
    mean_abs = sum(float((exp - got).abs().mean()) for exp, got in zip(expected, actual)) / len(expected)
    assert max_abs <= 1e-6 and mean_abs <= 1e-7, (
        "SpyNetWrapper changed VRT RGB inputs before calling the KAIR-derived SpyNet core; "
        f"observed max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}."
    )


def test_spynet_wrapper_passes_vrt_rgb_tensor_to_core_without_reformatting(monkeypatch):
    """Observe the exact tensor SpyNetWrapper sends into the KAIR-derived core."""
    wrapper = SpyNetWrapper(checkpoint=None, device='cpu', return_levels=[5])
    frame1 = torch.zeros(1, 3, 4, 4)
    frame2 = torch.zeros(1, 3, 4, 4)
    frame1[:, 0].fill_(0.10)  # R
    frame1[:, 1].fill_(0.20)  # G
    frame1[:, 2].fill_(0.30)  # B
    frame2.copy_(frame1)
    captured = {}

    class FakeCore(nn.Module):
        def forward(self, ref, supp):
            captured["ref"] = ref.detach().clone()
            captured["supp"] = supp.detach().clone()
            return [torch.zeros(ref.size(0), 2, ref.size(2), ref.size(3), device=ref.device)]

    monkeypatch.setattr(wrapper, "model", FakeCore())

    wrapper(frame1, frame2)

    assert torch.allclose(captured["ref"], frame1), (
        "SpyNetWrapper should pass VRT RGB [0,1] tensors unchanged into SpyNet core; "
        f"observed first-pixel={captured['ref'][0, :, 0, 0].tolist()}."
    )
