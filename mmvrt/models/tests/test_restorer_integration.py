"""Integration smoke tests for BaseRestorer with backbone wrapper."""
import torch
import torch.nn as nn

from mmvrt.models.restorers.base_restorer import BaseRestorer
from mmvrt.structures.data_sample import RestorationDataSample


class IdentityBackbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # simply return input tensor unchanged
        return x


def test_restorer_tensor_mode():
    bb = IdentityBackbone()
    restorer = BaseRestorer(backbone=bb, head=None, data_preprocessor=None, loss_module=None)
    inp = torch.rand((1, 5, 3, 16, 16), dtype=torch.float32)
    out = restorer.forward(inp, mode='tensor')
    assert isinstance(out, torch.Tensor)
    assert out.shape == inp.shape


def test_restorer_loss_and_predict_modes():
    bb = IdentityBackbone()
    restorer = BaseRestorer(backbone=bb, head=None, data_preprocessor=None, loss_module=None)
    inp = torch.rand((1, 5, 3, 16, 16), dtype=torch.float32)
    gt = torch.rand((1, 5, 3, 16, 16), dtype=torch.float32)
    ds = RestorationDataSample(inputs=inp, gt=gt, pred=None, metainfo={})

    # loss mode should return dict with 'loss'
    loss_out = restorer.forward(inp, data_samples=[ds], mode='loss')
    assert isinstance(loss_out, dict) and 'loss' in loss_out

    # predict mode should return list of RestorationDataSample
    pred_out = restorer.forward(inp, data_samples=[ds], mode='predict')
    assert isinstance(pred_out, list)
    assert hasattr(pred_out[0], 'pred')


if __name__ == "__main__":
    test_restorer_tensor_mode()
    test_restorer_loss_and_predict_modes()
    print("restorer integration smoke tests passed")


