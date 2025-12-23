"""Flow composition utilities migrated from legacy network_vrt.py.

Provides helpers to compose multi-frame flows (t->t+2, t->t+3) from lower-order flows.
"""
from typing import List, Tuple
import torch


def get_flow_4frames(flows_forward: List[torch.Tensor], flows_backward: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Get flow between t and t+2 from (t,t+1) and (t+1,t+2).

    Args:
        flows_forward: list of flow tensors (each tensor shape: (B, D, 2, H_i, W_i))
        flows_backward: list of flow tensors (same shapes)

    Returns:
        flows_backward2, flows_forward2: lists of composed flows
    """
    flows_backward2 = []
    for flows in flows_backward:
        d = flows.shape[1]
        flow_list = []
        for i in range(d - 1, 0, -1):
            flow_n1 = flows[:, i - 1, :, :, :]  # flow from i+1 to i
            flow_n2 = flows[:, i, :, :, :]  # flow from i+2 to i+1
            flow_list.insert(0, flow_n1 + _flow_warp(flow_n2, flow_n1))
        flows_backward2.append(torch.stack(flow_list, 1))

    flows_forward2 = []
    for flows in flows_forward:
        d = flows.shape[1]
        flow_list = []
        for i in range(1, d):
            flow_n1 = flows[:, i, :, :, :]  # flow from i-1 to i
            flow_n2 = flows[:, i - 1, :, :, :]  # flow from i-2 to i-1
            flow_list.append(flow_n1 + _flow_warp(flow_n2, flow_n1))
        flows_forward2.append(torch.stack(flow_list, 1))

    return flows_backward2, flows_forward2


def get_flow_6frames(flows_forward: List[torch.Tensor], flows_backward: List[torch.Tensor],
                     flows_forward2: List[torch.Tensor], flows_backward2: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Get flow between t and t+3 from (t,t+2) and (t+2,t+3).

    Args:
        flows_forward: list of flow tensors for 2-frame flows
        flows_backward: list of flow tensors for 2-frame flows
        flows_forward2: list of flow tensors for 4-frame composed flows
        flows_backward2: list of flow tensors for 4-frame composed flows

    Returns:
        flows_backward3, flows_forward3
    """
    flows_backward3 = []
    for flows, flows2 in zip(flows_backward, flows_backward2):
        d = flows2.shape[1]
        flow_list = []
        for i in range(d - 1, 0, -1):
            flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i+2 to i
            flow_n2 = flows[:, i + 1, :, :, :]  # flow from i+3 to i+2
            flow_list.insert(0, flow_n1 + _flow_warp(flow_n2, flow_n1))
        flows_backward3.append(torch.stack(flow_list, 1))

    flows_forward3 = []
    for flows, flows2 in zip(flows_forward, flows_forward2):
        d = flows2.shape[1]
        flow_list = []
        for i in range(2, d + 1):
            flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i-2 to i
            flow_n2 = flows[:, i - 2, :, :, :]  # flow from i-3 to i-2
            flow_list.append(flow_n1 + _flow_warp(flow_n2, flow_n1))
        flows_forward3.append(torch.stack(flow_list, 1))

    return flows_backward3, flows_forward3


def _flow_warp(flow_src: torch.Tensor, flow_ref: torch.Tensor) -> torch.Tensor:
    """Warp flow_src by flow_ref using simple grid-sample-based approximation.
    This mirrors legacy `flow_warp` usage but operates on flow tensors; keep it small
    and self-contained for smoke tests.
    """
    # flow_src, flow_ref: (B, 2, H, W) ; we will warp flow_src by flow_ref using sampling
    # Convert flow_ref (B, 2, H, W) to sampling grid (B, H, W, 2)
    B, C, H, W = flow_src.shape
    # create mesh grid
    yy, xx = torch.meshgrid(torch.arange(0, H, dtype=flow_src.dtype, device=flow_src.device),
                            torch.arange(0, W, dtype=flow_src.dtype, device=flow_src.device),
                            indexing='ij')
    grid = torch.stack((xx, yy), 2).float()  # H, W, 2
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # B,H,W,2
    vgrid = grid + flow_ref.permute(0, 2, 3, 1)
    # normalize to [-1,1]
    vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    # sample flow_src (need to permute to B, C, H, W for grid_sample)
    sampled = torch.nn.functional.grid_sample(flow_src, vgrid_scaled, mode='bilinear', padding_mode='zeros', align_corners=True)
    return sampled


