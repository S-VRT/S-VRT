import torch

from models.architectures.vrt.stages import Stage


def test_stage_forward_2frames():
    b = 1
    in_dim = 8
    dim = 8
    d = 2
    h = 4
    w = 4
    # Create stage
    stage = Stage(in_dim=in_dim, dim=dim, input_resolution=(d, h, w), depth=2, num_heads=2, window_size=(2, 4, 4), pa_frames=2, reshape='none')
    x = torch.randn(b, in_dim, d, h, w)
    # flows_backward and flows_forward: lists with one tensor each for this simple test
    flows_backward = [torch.zeros(b, d - 1, 2, h, w)]
    flows_forward = [torch.zeros(b, d - 1, 2, h, w)]
    out = stage(x, flows_backward, flows_forward)
    # output should have same shape as input (b, dim, d, h, w)
    assert out.shape == (b, dim, d, h, w)


