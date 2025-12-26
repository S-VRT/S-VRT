import math
import torch


def extract_rgb(x, channels=3):
    """Return up to the first `channels` feature channels (compatibility helper)."""
    return x[:, :, :min(channels, x.size(2)), :, :]


def compute_flows_2frames(spynet, x):
    """Compute flows between consecutive frames in x using a SpyNet-like callable.

    Args:
        spynet: SpyNet instance or callable that accepts (ref, supp) and returns a list of multiscale flows.
        x: tensor (b, n, c, h, w)

    Returns:
        flows_backward, flows_forward: lists of flow tensors at multiple scales
    """
    b, n, c, h, w = x.size()
    # SpyNet is pretrained on RGB only; strip any auxiliary channels (e.g., Spike) before flow.
    x_flow = extract_rgb(x)
    c_flow = x_flow.size(2)

    x_1 = x_flow[:, :-1, :, :, :].reshape(-1, c_flow, h, w)
    x_2 = x_flow[:, 1:, :, :, :].reshape(-1, c_flow, h, w)

    # backward
    flows_backward_raw = spynet(x_1, x_2)
    if not isinstance(flows_backward_raw, (list, tuple)):
        flows_backward_raw = [flows_backward_raw]
    flows_backward = []
    for i, flow in enumerate(flows_backward_raw):
        # reshape to (b, n-1, 2, H_i, W_i). Use enumerate index as the scale index.
        H_i = h // (2 ** i) if (h // (2 ** i)) > 0 else 1
        W_i = w // (2 ** i) if (w // (2 ** i)) > 0 else 1
        flows_backward.append(flow.view(b, n - 1, 2, H_i, W_i))

    # forward
    flows_forward_raw = spynet(x_2, x_1)
    if not isinstance(flows_forward_raw, (list, tuple)):
        flows_forward_raw = [flows_forward_raw]
    flows_forward = []
    for i, flow in enumerate(flows_forward_raw):
        H_i = h // (2 ** i) if (h // (2 ** i)) > 0 else 1
        W_i = w // (2 ** i) if (w // (2 ** i)) > 0 else 1
        flows_forward.append(flow.view(b, n - 1, 2, H_i, W_i))

    return flows_backward, flows_forward


__all__ = ["extract_rgb", "compute_flows_2frames"]


