"""Base restorer following MMEngine BaseModel pattern.

This defines the unified forward protocol for restoration tasks.
"""

from typing import Dict, Optional, Union, Tuple
import torch
from torch import Tensor, nn

try:
    from mmengine.model import BaseModel
    _USE_MMENGINE = True
except ImportError:
    _USE_MMENGINE = False
    BaseModel = nn.Module

from mmvrt.structures import RestorationDataSample


class BaseRestorer(BaseModel if _USE_MMENGINE else nn.Module):
    """Base class for restoration models.
    
    This follows MMEngine BaseModel pattern, providing unified forward interface
    that distinguishes between 'loss', 'predict', and 'tensor' modes.
    
    Attributes:
        data_preprocessor: Optional data preprocessor for normalization/concatenation
        backbone: Main feature extraction backbone
        head: Optional head for final output reconstruction
        loss_module: Optional loss module for training
    """
    
    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        data_preprocessor: Optional[nn.Module] = None,
        loss_module: Optional[nn.Module] = None,
        init_cfg: Optional[Dict] = None,
    ):
        if _USE_MMENGINE:
            super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        else:
            super().__init__()
            self.data_preprocessor = data_preprocessor
        
        self.backbone = backbone
        self.head = head
        self.loss_module = loss_module
    
    def forward(
        self,
        inputs: Union[Tensor, Dict[str, Tensor], RestorationDataSample],
        data_samples: Optional[Union[RestorationDataSample, list]] = None,
        mode: str = 'tensor',
        **kwargs
    ) -> Union[Dict[str, Tensor], list, Tensor]:
        """Forward function.
        
        Args:
            inputs: Input tensor or dict or DataSample. Shape (B, T, C, H, W) for video.
            data_samples: Optional ground truth DataSamples (for loss mode).
            mode: Forward mode. One of 'loss', 'predict', 'tensor'.
                - 'loss': Compute loss for training. Returns dict with 'loss' key.
                - 'predict': Inference mode. Returns list of DataSamples with predictions.
                - 'tensor': Raw tensor output. Returns tensor or dict of tensors.
            **kwargs: Additional arguments.
        
        Returns:
            - If mode='loss': Dict with 'loss' key and optional other metrics.
            - If mode='predict': List of RestorationDataSample with pred filled.
            - If mode='tensor': Tensor or dict of tensors.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be one of ['loss', 'predict', 'tensor']")
    
    def _forward(self, inputs: Union[Tensor, Dict[str, Tensor]], **kwargs) -> Tensor:
        """Raw forward pass (tensor mode).
        
        Args:
            inputs: Input tensor or dict. Shape (B, T, C, H, W).
            **kwargs: Additional arguments.
        
        Returns:
            Output tensor. Shape (B, T, C_out, H_out, W_out).
        """
        # Preprocess if data_preprocessor exists
        if self.data_preprocessor is not None:
            if isinstance(inputs, dict):
                inputs = self.data_preprocessor(inputs)
            else:
                # Wrap in dict for preprocessor
                inputs = self.data_preprocessor({'inputs': inputs})
                if isinstance(inputs, dict):
                    inputs = inputs.get('inputs', inputs)
        
        # Extract input tensor
        if isinstance(inputs, dict):
            x = inputs.get('inputs') or inputs.get('lq') or inputs.get('L')
        else:
            x = inputs
        
        # Backbone forward
        if self.backbone is not None:
            x = self.backbone(x)
        
        # Head forward (if exists)
        if self.head is not None:
            x = self.head(x)
        
        return x
    
    def loss(
        self,
        inputs: Union[Tensor, Dict[str, Tensor], RestorationDataSample],
        data_samples: Union[RestorationDataSample, list],
        **kwargs
    ) -> Dict[str, Tensor]:
        """Compute loss for training.
        
        Args:
            inputs: Input tensor/dict/DataSample.
            data_samples: Ground truth DataSamples.
            **kwargs: Additional arguments.
        
        Returns:
            Dict with 'loss' key and optional other metrics.
        """
        # Get predictions
        pred = self._forward(inputs, **kwargs)
        
        # Extract GT
        if isinstance(data_samples, list):
            gt = torch.stack([ds.gt for ds in data_samples if ds.gt is not None])
        elif isinstance(data_samples, RestorationDataSample):
            gt = data_samples.gt
        else:
            raise TypeError(f"data_samples must be RestorationDataSample or list, got {type(data_samples)}")
        
        # Compute loss
        if self.loss_module is not None:
            loss_dict = self.loss_module(pred, gt)
            if isinstance(loss_dict, dict):
                return loss_dict
            else:
                return {'loss': loss_dict}
        else:
            # Fallback: L1 loss
            return {'loss': torch.nn.functional.l1_loss(pred, gt)}
    
    def predict(
        self,
        inputs: Union[Tensor, Dict[str, Tensor], RestorationDataSample],
        data_samples: Optional[Union[RestorationDataSample, list]] = None,
        **kwargs
    ) -> list:
        """Predict mode: inference with DataSample output.
        
        Args:
            inputs: Input tensor/dict/DataSample.
            data_samples: Optional input DataSamples (for metainfo).
            **kwargs: Additional arguments.
        
        Returns:
            List of RestorationDataSample with pred filled.
        """
        # Get predictions
        pred = self._forward(inputs, **kwargs)
        
        # Convert to DataSamples
        if isinstance(data_samples, list):
            result = []
            for i, ds in enumerate(data_samples):
                new_ds = RestorationDataSample(
                    inputs=ds.inputs if hasattr(ds, 'inputs') else None,
                    gt=ds.gt if hasattr(ds, 'gt') else None,
                    pred=pred[i] if pred.dim() > 0 else pred,
                    metainfo=ds.metainfo if hasattr(ds, 'metainfo') else {}
                )
                result.append(new_ds)
            return result
        elif isinstance(data_samples, RestorationDataSample):
            return [RestorationDataSample(
                inputs=data_samples.inputs,
                gt=data_samples.gt,
                pred=pred,
                metainfo=data_samples.metainfo
            )]
        else:
            # Create new DataSample from inputs
            if isinstance(inputs, RestorationDataSample):
                return [RestorationDataSample(
                    inputs=inputs.inputs,
                    gt=inputs.gt,
                    pred=pred,
                    metainfo=inputs.metainfo
                )]
            else:
                # Minimal DataSample
                return [RestorationDataSample(
                    inputs=inputs if isinstance(inputs, Tensor) else None,
                    pred=pred,
                    metainfo={}
                )]

