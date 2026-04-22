import os
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional

from models.blocks.basic import BasicModule
from models.utils.flow import flow_warp
from .base import OpticalFlowModule


class SpyNet(nn.Module):
    """SpyNet architecture (moved from architectures/vrt/warp.py).

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """
    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if not os.path.exists(load_path):
                import requests
                url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading SpyNet pretrained model from {url}')
                os.makedirs(os.path.dirname(load_path), exist_ok=True)
                open(load_path, 'wb').write(r.content)

            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        # tensor_input is already RGB [0,1] from the new preprocessing
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow_h = max(1, int(math.floor(ref[0].size(2) / 2.0)))
        flow_w = max(1, int(math.floor(ref[0].size(3) / 2.0)))
        flow = ref[0].new_zeros([ref[0].size(0), 2, flow_h, flow_w])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            target_h, target_w = ref[level].size(2), ref[level].size(3)
            if upsampled_flow.size(2) != target_h or upsampled_flow.size(3) != target_w:
                upsampled_flow = F.interpolate(input=upsampled_flow, size=(target_h, target_w), mode='bilinear', align_corners=True)

            warped = flow_warp(supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border')
            flow = self.basic_module[level](torch.cat([ref[level], warped, upsampled_flow], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2 ** (5 - level)
                flow_out = F.interpolate(input=flow, size=(h // scale, w // scale), mode='bilinear', align_corners=False)
                flow_out[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
                flow_out[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list


class SpyNetWrapper(OpticalFlowModule):
    """Thin adapter around the SpyNet implementation to match the OpticalFlowModule API."""
    def __init__(self, checkpoint: Optional[str] = None, device: str = 'cpu', return_levels=None):
        super().__init__()
        self.device = torch.device(device)
        self.return_levels = return_levels if return_levels is not None else [5]
        self.model = SpyNet(checkpoint, self.return_levels)
        self.model.to(self.device)
        self.model.eval()

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor):
        # Determine runtime device from model parameters (DDP or outer code may move model after construction)
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = self.device
        frame1 = frame1.to(device)
        frame2 = frame2.to(device)

        # Preprocess: BGR [0,1] -> RGB normalized for SpyNet
        frame1, frame2 = self.preprocess_frames(frame1, frame2, 'rgb_norm')

        with torch.set_grad_enabled(self._should_track_gradients()):
            out = self.model(frame1, frame2)
        # SpyNet returns flows in [low_res, ..., high_res] order, which matches VRT expectations
        return out

    def load_checkpoint(self, path: str) -> None:
        if path:
            state = torch.load(path, map_location=lambda storage, loc: storage)
            if isinstance(state, dict) and 'params' in state:
                self.model.load_state_dict(state['params'])
            else:
                self.model.load_state_dict(state)


