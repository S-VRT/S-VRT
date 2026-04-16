import torch

from models.optical_flow import create_optical_flow


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
