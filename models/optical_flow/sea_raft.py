import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Optional

from .base import OpticalFlowModule

# -------------------------
# Minimal utils (from vendor)
# -------------------------
def coords_grid(batch, ht, wd, device=None):
    N = batch
    # create a simple coords grid used by RAFT init functions
    # return a grid of (x, y) coordinates shaped (N, 2, ht, wd)
    y = torch.arange(ht, device=device).view(1, ht, 1).expand(N, ht, wd)
    x = torch.arange(wd, device=device).view(1, 1, wd).expand(N, ht, wd)
    coords = torch.stack([x, y], dim=3).float()            # (N, ht, wd, 2)
    coords = coords.permute(0, 3, 1, 2).contiguous()       # (N, 2, ht, wd)
    return coords


class InputPadder:
    def __init__(self, shape):
        # noop pad for our integrated/ported version
        self.shape = shape

    def pad(self, *imgs):
        return imgs

    def unpad(self, img):
        return img

# -------------------------
# Layer / small NN utils
# -------------------------
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    def __init__(self, dim, output_dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * output_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * output_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = self.final(inp + x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.bn3 = norm_layer(planes)
            self.downsample = nn.Sequential(conv1x1(in_planes, planes, stride=stride), self.bn3)

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


# -------------------------
# Extractor (ResNetFPN) (minimal)
# -------------------------
class ResNetFPN(nn.Module):
    def __init__(self, args, input_dim=3, output_dim=256, ratio=1.0, norm_layer=nn.BatchNorm2d, init_weight=False):
        super().__init__()
        block = BasicBlock
        block_dims = list(args.block_dims)
        initial_dim = args.initial_dim
        self.init_weight = init_weight
        self.input_dim = input_dim
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        if args.pretrain == 'resnet34':
            n_block = [3, 4, 6]
        elif args.pretrain == 'resnet18':
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError
        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])
        self.layer3 = self._make_layer(block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2])
        self.final_conv = conv1x1(block_dims[2], output_dim)
        self._init_weights(args)

    def _init_weights(self, args):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.init_weight:
            from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
            if args.pretrain == 'resnet18':
                pretrained_dict = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
            else:
                pretrained_dict = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if self.input_dim == 6:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        pretrained_dict[k] = torch.cat((v, v), dim=1)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
        output = self.final_conv(x)
        return output


# -------------------------
# CorrBlock (minimal port)
# -------------------------
def coords_feature(fmap, b, x, y):
    H, W = fmap.shape[2:]
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    b = b.long()
    x = torch.clamp(x, 0, W - 1).long()
    y = torch.clamp(y, 0, H - 1).long()
    res = fmap[b, :, y, x] * mask.float().unsqueeze(1)
    return res


def bilinear_sampling(fmap, coords):
    # fmap: (M, C, H, W)
    # coords: either (K, 3) with [b, x, y] per-row, or (M, ky, kx, 2) with per-sample grids (x,y)
    device = fmap.device
    if coords.ndim == 4 and coords.shape[-1] == 2:
        # coords shaped (M, ky, kx, 2) -> flatten to (M*ky*kx, 2) and prepend batch indices
        M, ky, kx, _ = coords.shape
        coords_flat = coords.reshape(M * ky * kx, 2)
        # create batch index for each point
        b_idx = torch.arange(M, device=device).view(M, 1).expand(M, ky * kx).reshape(-1)
        coords = torch.cat([b_idx.unsqueeze(1).float(), coords_flat], dim=1)
    else:
        coords = coords.view(-1, coords.shape[-1])

    offset = (coords - coords.floor()).to(device)
    # coords format now: [b, x, y]
    dx, dy = offset[:, 1].unsqueeze(1), offset[:, 2].unsqueeze(1)
    b = coords[:, 0].long()
    x0, y0 = coords[:, 1].floor().long(), coords[:, 2].floor().long()
    x1, y1 = x0 + 1, y0 + 1
    f00 = (1 - dy) * (1 - dx) * coords_feature(fmap, b, x0, y0)
    f01 = (1 - dy) * dx * coords_feature(fmap, b, x0, y1)
    f10 = dy * (1 - dx) * coords_feature(fmap, b, x1, y0)
    f11 = dy * dx * coords_feature(fmap, b, x1, y1)
    return f00 + f01 + f10 + f11


def bilinear_sampling_corr(corr, idx1, idx2):
    M, n_points = idx2.shape[:2]
    idx1 = idx1.unsqueeze(1).repeat(1, n_points, 1).view(-1, 3)
    idx2 = idx2.view(-1, 3)
    device = corr.device
    offset = idx2 - idx2.floor()
    dx, dy = offset[:, 1], offset[:, 2]
    b = idx2[:, 0].long()
    x0, y0 = idx2[:, 1].floor(), idx2[:, 2].floor()
    x1, y1 = x0 + 1, y0 + 1
    f00 = (1 - dy) * (1 - dx) * coords_feature(corr, b, x0, y0)
    res = f00
    return res.view(M, n_points)


class CorrBlock:
    def __init__(self, fmap1, fmap2, args):
        self.num_levels = args.corr_levels
        self.radius = args.corr_radius
        self.args = args
        self.corr_pyramid = []
        for i in range(self.num_levels):
            corr = CorrBlock.corr(fmap1, fmap2, 1)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
            fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode='bilinear', align_corners=False)
            self.corr_pyramid.append(corr)

    def __call__(self, coords, dilation=None):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        if dilation is None:
            dilation = torch.ones(batch, 1, h1, w1, device=coords.device)
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            device = coords.device
            dx = torch.linspace(-r, r, 2 * r + 1, device=device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            delta_lvl = delta_lvl * dilation.view(batch * h1 * w1, 1, 1, 1)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampling(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2, num_head):
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]
        fmap1 = fmap1.view(batch, num_head, dim // num_head, h1 * w1)
        fmap2 = fmap2.view(batch, num_head, dim // num_head, h2 * w2)
        corr = fmap1.transpose(2, 3) @ fmap2
        corr = corr.reshape(batch, num_head, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)
        return corr / torch.sqrt(torch.tensor(dim).float())


# -------------------------
# Update block (minimal)
# -------------------------
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class BasicMotionEncoder(nn.Module):
    def __init__(self, args, dim=128):
        super().__init__()
        cor_planes = args.corr_channel
        self.convc1 = nn.Conv2d(cor_planes, dim * 2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim * 2, dim + dim // 2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hdim=128, cdim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, dim=cdim)
        self.refine = []
        for i in range(args.num_blocks):
            self.refine.append(ConvNextBlock(2 * cdim + hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net


# -------------------------
# RAFT core (consolidated)
# -------------------------
from typing import Dict, Any

class RAFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.output_dim = args.dim * 2
        self.args.corr_levels = 4
        self.args.corr_radius = args.radius
        self.args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2
        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)
        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)
        self.upsample_weight = nn.Sequential(nn.Conv2d(args.dim, args.dim * 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0))
        self.flow_head = nn.Sequential(nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(2 * args.dim, 6, 3, padding=1))
        if args.iters > 0:
            self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d, init_weight=True)
            self.update_block = BasicUpdateBlock(args, hdim=args.dim, cdim=args.dim)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords2 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False) -> Dict[str, Any]:
        N, _, H, W = image1.shape
        if iters is None:
            iters = self.args.iters
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H, W, device=image1.device)
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        flow_predictions = []
        info_predictions = []
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H // 8, W // 8, device=image1.device)
        cnet = self.cnet(torch.cat([image1, image2], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)
        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)
        if self.args.iters > 0:
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)
        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=image1.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)
        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])
        if test_mode == False:
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.args.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.args.var_max
                    var_min = self.args.var_min
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)
            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': None}


# -------------------------
# SEA-RAFT adapter (final, self-contained)
# -------------------------
class SeaRaftWrapper(OpticalFlowModule):
    """Self-contained SEA-RAFT adapter (no runtime dependency on _vendor or SEA-RAFT-main)."""
    def __init__(self, checkpoint: Optional[str] = None, device: str = 'cpu', **kwargs):
        super().__init__()
        self.device = torch.device(device)
        args = SimpleNamespace()
        args.dim = kwargs.get('dim', 64)
        args.radius = kwargs.get('radius', 4)
        args.iters = kwargs.get('iters', 12)
        args.num_blocks = kwargs.get('num_blocks', 2)
        args.block_dims = kwargs.get('block_dims', [64, 128, 256])
        args.initial_dim = kwargs.get('initial_dim', 64)
        args.pretrain = kwargs.get('pretrain', 'resnet18')
        args.use_var = kwargs.get('use_var', False)
        args.var_max = kwargs.get('var_max', 0)
        args.var_min = kwargs.get('var_min', 0)

        # Build and prepare model
        self.model = RAFT(args)
        self.model.to(self.device)
        self.model.eval()

        if checkpoint:
            try:
                state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
                if isinstance(state, dict) and 'state_dict' in state:
                    self.model.load_state_dict(state['state_dict'])
                elif isinstance(state, dict) and 'params' in state:
                    self.model.load_state_dict(state['params'])
                else:
                    self.model.load_state_dict(state)
            except Exception:
                pass

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor):
        # Determine runtime device from model parameters (DDP or outer code may move model after construction)
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = self.device
        frame1 = frame1.to(device)
        frame2 = frame2.to(device)

        # Preprocess: BGR [0,1] -> RGB [0,255] for SeaRaft
        frame1, frame2 = self.preprocess_frames(frame1, frame2, 'rgb_255')

        # RAFT requires minimum size (8x8 after 1/8 downsampling, plus 4 levels of 0.5x downsampling in CorrBlock)
        # So minimum input size should be 8 * 2^4 = 8 * 16 = 128 to avoid 0-sized tensors
        _, _, H, W = frame1.shape
        min_size = 128  # 8 * 2^4
        if H < min_size or W < min_size:
            # For very small inputs, pad to minimum size
            pad_h = max(0, min_size - H)
            pad_w = max(0, min_size - W)
            frame1 = torch.nn.functional.pad(frame1, (0, pad_w, 0, pad_h), mode='replicate')
            frame2 = torch.nn.functional.pad(frame2, (0, pad_w, 0, pad_h), mode='replicate')
            padded = True
            orig_h, orig_w = H, W
        else:
            padded = False

        with torch.no_grad():
            out = self.model(frame1, frame2, test_mode=True)

        # If we padded, we need to unpad the output flows
        if padded:
            if isinstance(out, dict) and 'final' in out:
                out['final'] = out['final'][:, :, :orig_h, :orig_w]
            if isinstance(out, dict) and 'flow' in out and out['flow']:
                for i in range(len(out['flow'])):
                    out['flow'][i] = out['flow'][i][:, :, :orig_h, :orig_w]
            if isinstance(out, dict) and 'info' in out and out['info']:
                for i in range(len(out['info'])):
                    out['info'][i] = out['info'][i][:, :, :orig_h, :orig_w]
        # Normalize RAFT outputs into the same multi-scale list format SpyNet provides:
        # expected: list of 4 tensors for scales [1, 1/2, 1/4, 1/8] with shapes (N,2,H_s,W_s)
        if isinstance(out, dict):
            # try to obtain final full-resolution flow
            final_flow = None
            if 'final' in out and out['final'] is not None:
                final_flow = out['final']
            elif 'flow' in out and out['flow']:
                # take last prediction as final
                final_flow = out['flow'][-1]
            if final_flow is not None:
                N, C, H_full, W_full = final_flow.shape
                flows_multi = []
                # Build in same order as SpyNet: [fine, medium, coarse, coarsest]
                for i in range(4):  # 0, 1, 2, 3
                    if i == 0:
                        flow_i = final_flow  # full resolution
                    else:
                        H_i = H_full // (2 ** i)
                        W_i = W_full // (2 ** i)
                        flow_i = torch.nn.functional.interpolate(final_flow, size=(H_i, W_i), mode='bilinear', align_corners=False)

                    # Post-process flow for robustness (especially important for RAFT's sharper predictions)
                    flow_i = self.postprocess_flow(flow_i, max_displacement=min(H_full, W_full) * 0.1)  # clip to 10% of image size
                    flows_multi.append(flow_i)
                return flows_multi
        # Fallback: return original output (best-effort)
        return out

    def load_checkpoint(self, path: str) -> None:
        if not path:
            return
        state = torch.load(path, map_location=lambda storage, loc: storage)
        if isinstance(state, dict) and 'state_dict' in state:
            self.model.load_state_dict(state['state_dict'])
        elif isinstance(state, dict) and 'params' in state:
            self.model.load_state_dict(state['params'])
        else:
            self.model.load_state_dict(state)
