"""PSNR metric for restoration tasks."""

import torch
import torch.nn.functional as F
from typing import Optional

try:
    from mmengine.evaluator import BaseMetric
    _USE_MMENGINE = True
except ImportError:
    _USE_MMENGINE = False
    BaseMetric = object

from mmvrt.registry import METRICS


@METRICS.register_module()
class PSNR(BaseMetric if _USE_MMENGINE else object):
    """Peak Signal-to-Noise Ratio metric.
    
    Args:
        max_val (float): Maximum pixel value. Default: 1.0.
        collect_device (str): Device for collecting results. Default: 'cpu'.
        prefix (str): Prefix for metric name. Default: 'PSNR'.
    """
    
    def __init__(
        self,
        max_val: float = 1.0,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        **kwargs
    ):
        if _USE_MMENGINE:
            super().__init__(collect_device=collect_device, prefix=prefix, **kwargs)
        else:
            self.collect_device = collect_device
            self.prefix = prefix or 'PSNR'
        self.max_val = max_val
    
    def process(self, data_batch, data_samples):
        """Process a batch of data samples.
        
        Args:
            data_batch: Input data batch.
            data_samples: Model predictions (list of DataSamples or dict).
        """
        if not _USE_MMENGINE:
            # Fallback implementation
            return
        
        from mmvrt.structures import RestorationDataSample
        
        for data_sample in data_samples:
            if isinstance(data_sample, RestorationDataSample):
                pred = data_sample.pred
                gt = data_sample.gt
            elif isinstance(data_sample, dict):
                pred = data_sample.get('pred') or data_sample.get('pred_img')
                gt = data_sample.get('gt') or data_sample.get('gt_img')
            else:
                continue
            
            if pred is None or gt is None:
                continue
            
            # Compute PSNR
            psnr_value = self._compute_psnr(pred, gt)
            
            # Store result
            result = {f'{self.prefix}': psnr_value.item() if isinstance(psnr_value, torch.Tensor) else psnr_value}
            self.results.append(result)
    
    def compute_metrics(self, results):
        """Compute metrics from results.
        
        Args:
            results: List of result dicts.
        
        Returns:
            Dict with metric name and average value.
        """
        if not results:
            return {f'{self.prefix}': 0.0}
        
        metric_key = f'{self.prefix}'
        values = [r[metric_key] for r in results if metric_key in r]
        
        if not values:
            return {f'{self.prefix}': 0.0}
        
        avg_value = sum(values) / len(values)
        return {f'{self.prefix}': avg_value}
    
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute PSNR between prediction and target.
        
        Args:
            pred: Prediction tensor (B, T, C, H, W) or (B, C, H, W) or (C, H, W).
            target: Target tensor (same shape as pred).
        
        Returns:
            PSNR value (scalar tensor or float).
        """
        # Ensure same shape
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        # Flatten spatial and temporal dimensions
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Handle different dimensions
        if mse.dim() == 5:  # (B, T, C, H, W)
            mse = mse.flatten(1).mean(1)  # (B, T*C*H*W) -> (B,)
        elif mse.dim() == 4:  # (B, C, H, W)
            mse = mse.flatten(1).mean(1)  # (B, C*H*W) -> (B,)
        elif mse.dim() == 3:  # (C, H, W)
            mse = mse.flatten().mean()  # scalar
        else:
            mse = mse.mean()
        
        # Compute PSNR
        if isinstance(mse, torch.Tensor):
            psnr = 20 * torch.log10(torch.tensor(self.max_val, device=mse.device, dtype=mse.dtype)) - \
                   10 * torch.log10(mse + 1e-8)
        else:
            psnr = 20 * torch.log10(self.max_val) - 10 * torch.log10(mse + 1e-8)
        
        return psnr


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Compute PSNR between prediction and target (standalone function).
    
    Args:
        pred: Prediction tensor.
        target: Target tensor.
        max_val: Maximum pixel value. Default: 1.0.
    
    Returns:
        PSNR value.
    """
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(1)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse + 1e-8)

