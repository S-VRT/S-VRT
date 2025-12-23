"""Flow estimation and composition utilities to mirror VRT.get_flows behavior."""
from typing import Tuple, List
import torch
import math

from mmvrt.models.motion.spynet import SpyNet
from mmvrt.models.motion.flow_ops import flow_warp
from mmvrt.models.motion.flow_composition import get_flow_4frames, get_flow_6frames


def get_flow_2frames(x: torch.Tensor, spynet: SpyNet) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Estimate flows between consecutive frames using SpyNet.

    Args:
        x: (B, N, C, H, W)
        spynet: SpyNet instance

    Returns:
        flows_backward, flows_forward: lists of per-scale flow tensors
    """
    b, n, c, h, w = x.size()
    # SpyNet expects RGB only; extract first 3 channels
    x_flow = x[:, :, :3, :, :]
    c_flow = x_flow.size(2)
    x_1 = x_flow[:, :-1].reshape(-1, c_flow, h, w)
    x_2 = x_flow[:, 1:].reshape(-1, c_flow, h, w)

    # backward: spynet(x1, x2)
    flows_backward = spynet(x_1, x_2)
    flows_backward = [flow.view(b, n - 1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_backward, range(len(flows_backward)))]

    # forward: spynet(x2, x1)
    flows_forward = spynet(x_2, x_1)
    flows_forward = [flow.view(b, n - 1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in zip(flows_forward, range(len(flows_forward)))]

    return flows_backward, flows_forward


def compute_flows(x: torch.Tensor, pa_frames: int = 2, spynet: SpyNet = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Compute multi-scale flows depending on pa_frames (2/4/6).

    Args:
        x: (B, N, C, H, W)
        pa_frames: int (2,4,6)
        spynet: optional SpyNet instance; if None, a fresh one will be created.

    Returns:
        flows_backward, flows_forward: concatenated lists of flows per scale.
    """
    if spynet is None:
        spynet = SpyNet()

    if pa_frames == 2:
        flows_backward, flows_forward = get_flow_2frames(x, spynet)
    elif pa_frames == 4:
        fb2, ff2 = get_flow_2frames(x, spynet)
        fb4, ff4 = get_flow_4frames(ff2, fb2)
        flows_backward = fb2 + fb4
        flows_forward = ff2 + ff4
    elif pa_frames == 6:
        fb2, ff2 = get_flow_2frames(x, spynet)
        fb4, ff4 = get_flow_4frames(ff2, fb2)
        fb6, ff6 = get_flow_6frames(ff2, fb2, ff4, fb4)
        flows_backward = fb2 + fb4 + fb6
        flows_forward = ff2 + ff4 + ff6
    else:
        raise ValueError("Unsupported pa_frames value: must be 2,4,6")

    return flows_backward, flows_forward


