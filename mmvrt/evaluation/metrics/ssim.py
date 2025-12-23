"""SSIM metric for restoration tasks."""

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
class SSIM(BaseMetric if _USE_MMENGINE else object):
    """Structural Similarity Index Metric.
    
    Args:
        window_size (int): Window size for SSIM computation. Default: 7.
        c1 (float): Constant for luminance. Default: 0.01**2.
        c2 (float): Constant for contrast. Default: 0.03**2.
        collect_device (str): Device for collecting results. Default: 'cpu'.
        prefix (str): Prefix for metric name. Default: 'SSIM'.
    """
    
    def __init__(
        self,
        window_size: int = 7,
        c1: float = 0.01**2,
        c2: float = 0.03**2,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        **kwargs
    ):
        if _USE_MMENGINE:
            super().__init__(collect_device=collect_device, prefix=prefix, **kwargs)
        else:
            self.collect_device = collect_device
            self.prefix = prefix or 'SSIM'
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2
    
    def process(self, data_batch, data_samples):
        """Process a batch of data samples.
        
        Args:
            data_batch: Input data batch.
            data_samples: Model predictions (list of DataSamples or dict).
        """
        if not _USE_MMENGINE:
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
            
            # Compute SSIM
            ssim_value = self._compute_ssim(pred, gt)
            
            # Store result
            result = {f'{self.prefix}': ssim_value.item() if isinstance(ssim_value, torch.Tensor) else ssim_value}
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
    
    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between prediction and target.
        
        Args:
            pred: Prediction tensor (B, T, C, H, W) or (B, C, H, W).
            target: Target tensor (same shape as pred).
        
        Returns:
            SSIM value (scalar tensor or float).
        """
        # Ensure same shape
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        # Handle video (B, T, C, H, W) -> (B*T, C, H, W)
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred = pred.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)
        
        # Compute SSIM for 2D images
        ssim_value = self._ssim_2d(pred, target)
        
        # Average over batch
        if ssim_value.dim() > 0:
            ssim_value = ssim_value.mean()
        
        return ssim_value
    
    def _ssim_2d(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM for 2D images.
        
        Args:
            x: Input tensor (B, C, H, W).
            y: Target tensor (B, C, H, W).
        
        Returns:
            SSIM values (B,).
        """
        mu_x = F.avg_pool2d(x, self.window_size, 1, self.window_size // 2)
        mu_y = F.avg_pool2d(y, self.window_size, 1, self.window_size // 2)
        sigma_x = F.avg_pool2d(x * x, self.window_size, 1, self.window_size // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, self.window_size, 1, self.window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, self.window_size, 1, self.window_size // 2) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)) / (
            (mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2)
        )
        return ssim_map.mean([1, 2, 3])


def ssim_video(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Compute SSIM for video (standalone function).
    
    Args:
        pred: Prediction tensor (B, T, C, H, W).
        target: Target tensor (B, T, C, H, W).
        window_size: Window size for SSIM. Default: 7.
    
    Returns:
        SSIM values (B,).
    """
    b, t, c, h, w = pred.shape
    pred = pred.reshape(b * t, c, h, w)
    target = target.reshape(b * t, c, h, w)
    
    mu_x = F.avg_pool2d(pred, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, 1, window_size // 2)
    sigma_x = F.avg_pool2d(pred * pred, window_size, 1, window_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, window_size, 1, window_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean([1, 2, 3]).view(b, t).mean(1)

