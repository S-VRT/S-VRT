import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from typing import List, Optional

from mmvrt.registry import MODELS


class VGGFeatureExtractor(nn.Module):
    """VGG feature extractor for perceptual loss.

    Mirrors the behavior in the original project: extracts intermediate feature
    maps from VGG19 and optionally normalizes input.
    """

    def __init__(self, feature_layer: List[int] = [2, 7, 16, 25, 34], use_input_norm: bool = True, use_range_norm: bool = False):
        super().__init__()
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        model = torchvision.models.vgg19(pretrained=True)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer) - 1):
                self.features.add_module('child' + str(i),
                                         nn.Sequential(*list(model.features.children())[(feature_layer[i] + 1):(feature_layer[i + 1] + 1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        # Freeze parameters
        for _, v in self.features.named_parameters():
            v.requires_grad = False

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

    def forward(self, x: torch.Tensor):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            outputs = []
            for child_model in self.features.children():
                x = child_model(x)
                outputs.append(x.clone())
            return outputs
        else:
            return self.features(x)


@MODELS.register_module()
class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""

    def __init__(self,
                 feature_layer: List[int] = [2, 7, 16, 25, 34],
                 weights: Optional[List[float]] = None,
                 lossfn_type: str = 'l1',
                 use_input_norm: bool = True,
                 use_range_norm: bool = False):
        super().__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.weights = weights or [0.1, 0.1, 1.0, 1.0, 1.0]
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()

    def forward(self, x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        x_vgg = self.vgg(x)
        gt_vgg = self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            for i in range(len(x_vgg)):
                loss = loss + self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss = self.lossfn(x_vgg, gt_vgg.detach())
        return loss


@MODELS.register_module()
class GANLoss(nn.Module):
    """Flexible GAN loss wrapper (gan, ragan, lsgan, wgan, softplusgan)."""

    def __init__(self, gan_type: str = 'gan', real_label_val: float = 1.0, fake_label_val: float = 0.0):
        super().__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type in ('gan', 'ragan'):
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            # WGAN loss handled as a custom function below
            self.loss = None
        elif self.gan_type == 'softplusgan':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN type [{self.gan_type}] is not supported')

    def get_target_label(self, input: torch.Tensor, target_is_real: bool):
        if self.gan_type in ('wgan', 'softplusgan'):
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input: torch.Tensor, target_is_real: bool):
        if self.gan_type == 'wgan':
            return -1 * input.mean() if target_is_real else input.mean()
        if self.gan_type == 'softplusgan':
            return F.softplus(-input).mean() if target_is_real else F.softplus(input).mean()
        target_label = self.get_target_label(input, target_is_real)
        return self.loss(input, target_label)


@MODELS.register_module()
class TVLoss(nn.Module):
    """Total variation loss."""

    def __init__(self, tv_loss_weight: float = 1.0):
        super().__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t: torch.Tensor) -> int:
        return t.size(1) * t.size(2) * t.size(3)


