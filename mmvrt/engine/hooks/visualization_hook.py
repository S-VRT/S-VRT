"""Visualization hook for restoration tasks.

This hook periodically saves comparison videos/frames during training/validation.
"""

from typing import Any, Optional
from pathlib import Path
import torch
import numpy as np

try:
    from mmengine.hooks import Hook
    from mmengine.runner import Runner
    _USE_MMENGINE = True
except ImportError:
    # Fallback for compatibility
    _USE_MMENGINE = False
    Hook = object
    Runner = Any

from mmvrt.registry import HOOKS
from mmvrt.structures import RestorationDataSample


@HOOKS.register_module()
class VisualizationHook(Hook if _USE_MMENGINE else object):
    """Hook for saving visualization outputs during training/validation.
    
    This hook saves comparison images/videos showing:
    - Input (blurred/low-quality)
    - Prediction
    - Ground truth (if available)
    
    Args:
        interval (int): Save interval (in iterations). Default: 1000.
        save_dir (str): Directory to save visualizations. Default: 'visualizations'.
        max_samples (int): Maximum number of samples to save per interval. Default: 4.
        save_video (bool): Whether to save video clips. Default: True.
        save_frames (bool): Whether to save individual frames. Default: True.
    """
    
    priority = 50  # Run after other hooks
    
    def __init__(
        self,
        interval: int = 1000,
        save_dir: str = 'visualizations',
        max_samples: int = 4,
        save_video: bool = True,
        save_frames: bool = True,
    ):
        if _USE_MMENGINE:
            super().__init__()
        self.interval = interval
        self.save_dir = Path(save_dir)
        self.max_samples = max_samples
        self.save_video = save_video
        self.save_frames = save_frames
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Any = None,
        outputs: Any = None,
    ) -> None:
        """Save visualizations after validation iteration.
        
        Args:
            runner: MMEngine Runner instance.
            batch_idx: Current batch index.
            data_batch: Input data batch.
            outputs: Model outputs.
        """
        if batch_idx % self.interval != 0:
            return
        
        # Only save on rank 0
        if hasattr(runner, 'rank') and runner.rank != 0:
            return
        
        # Extract predictions and ground truth
        if isinstance(outputs, list):
            # Assume list of DataSamples
            for i, output in enumerate(outputs[:self.max_samples]):
                if isinstance(output, RestorationDataSample):
                    self._save_sample(output, batch_idx, i)
        elif isinstance(outputs, dict):
            # Assume dict format
            pred = outputs.get('pred') or outputs.get('pred_img')
            gt = outputs.get('gt') or outputs.get('gt_img')
            inputs = outputs.get('inputs') or outputs.get('lq')
            if pred is not None:
                self._save_from_tensors(inputs, pred, gt, batch_idx)
    
    def _save_sample(self, sample: RestorationDataSample, batch_idx: int, sample_idx: int):
        """Save a single DataSample.
        
        Args:
            sample: RestorationDataSample instance.
            batch_idx: Batch index.
            sample_idx: Sample index within batch.
        """
        if sample.pred is None:
            return
        
        # Convert tensors to numpy
        pred = self._tensor_to_numpy(sample.pred)
        inputs = self._tensor_to_numpy(sample.inputs) if sample.inputs is not None else None
        gt = self._tensor_to_numpy(sample.gt) if sample.gt is not None else None
        
        # Save frames
        if self.save_frames:
            self._save_frames(inputs, pred, gt, batch_idx, sample_idx)
        
        # Save video
        if self.save_video:
            self._save_video(inputs, pred, gt, batch_idx, sample_idx)
    
    def _save_from_tensors(
        self,
        inputs: Optional[torch.Tensor],
        pred: torch.Tensor,
        gt: Optional[torch.Tensor],
        batch_idx: int,
        sample_idx: int = 0
    ):
        """Save from raw tensors.
        
        Args:
            inputs: Input tensor (B, T, C, H, W) or (T, C, H, W).
            pred: Prediction tensor (B, T, C, H, W) or (T, C, H, W).
            gt: Ground truth tensor (B, T, C, H, W) or (T, C, H, W).
            batch_idx: Batch index.
            sample_idx: Sample index.
        """
        pred_np = self._tensor_to_numpy(pred)
        inputs_np = self._tensor_to_numpy(inputs) if inputs is not None else None
        gt_np = self._tensor_to_numpy(gt) if gt is not None else None
        
        if self.save_frames:
            self._save_frames(inputs_np, pred_np, gt_np, batch_idx, sample_idx)
        
        if self.save_video:
            self._save_video(inputs_np, pred_np, gt_np, batch_idx, sample_idx)
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array.
        
        Args:
            tensor: Input tensor (B, T, C, H, W) or (T, C, H, W) or (C, H, W).
        
        Returns:
            Numpy array (T, H, W, C) or (H, W, C).
        """
        if tensor is None:
            return None
        
        # Detach and move to CPU
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
        
        # Handle different dimensions
        if tensor.dim() == 5:  # (B, T, C, H, W)
            tensor = tensor[0]  # Take first batch
        if tensor.dim() == 4:  # (T, C, H, W)
            tensor = tensor.permute(0, 2, 3, 1)  # (T, H, W, C)
        elif tensor.dim() == 3:  # (C, H, W)
            tensor = tensor.permute(1, 2, 0)  # (H, W, C)
        
        # Convert to numpy
        arr = tensor.numpy()
        
        # Denormalize if needed (assume ImageNet normalization)
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            # Clamp to [0, 1] range
            arr = np.clip(arr, 0, 1)
            # Convert to uint8
            arr = (arr * 255).astype(np.uint8)
        
        return arr
    
    def _save_frames(
        self,
        inputs: Optional[np.ndarray],
        pred: np.ndarray,
        gt: Optional[np.ndarray],
        batch_idx: int,
        sample_idx: int
    ):
        """Save individual frames.
        
        Args:
            inputs: Input frames (T, H, W, C) or (H, W, C).
            pred: Prediction frames (T, H, W, C) or (H, W, C).
            gt: Ground truth frames (T, H, W, C) or (H, W, C).
            batch_idx: Batch index.
            sample_idx: Sample index.
        """
        try:
            import cv2
        except ImportError:
            print("Warning: OpenCV not available, skipping frame saving")
            return
        
        # Create save directory
        frame_dir = self.save_dir / f"iter_{batch_idx:08d}" / f"sample_{sample_idx}" / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle single frame vs sequence
        if pred.ndim == 3:  # (H, W, C)
            pred = pred[np.newaxis, ...]  # (1, H, W, C)
            if inputs is not None and inputs.ndim == 3:
                inputs = inputs[np.newaxis, ...]
            if gt is not None and gt.ndim == 3:
                gt = gt[np.newaxis, ...]
        
        # Save middle frame
        mid_idx = pred.shape[0] // 2
        cv2.imwrite(str(frame_dir / "pred_mid.png"), pred[mid_idx])
        if inputs is not None:
            # Extract RGB channels only (first 3)
            inputs_rgb = inputs[mid_idx, :, :, :3] if inputs.shape[-1] >= 3 else inputs[mid_idx]
            cv2.imwrite(str(frame_dir / "input_mid.png"), inputs_rgb)
        if gt is not None:
            cv2.imwrite(str(frame_dir / "gt_mid.png"), gt[mid_idx])
    
    def _save_video(
        self,
        inputs: Optional[np.ndarray],
        pred: np.ndarray,
        gt: Optional[np.ndarray],
        batch_idx: int,
        sample_idx: int
    ):
        """Save video clips.
        
        Args:
            inputs: Input frames (T, H, W, C) or (H, W, C).
            pred: Prediction frames (T, H, W, C) or (H, W, C).
            gt: Ground truth frames (T, H, W, C) or (H, W, C).
            batch_idx: Batch index.
            sample_idx: Sample index.
        """
        try:
            import cv2
        except ImportError:
            print("Warning: OpenCV not available, skipping video saving")
            return
        
        # Create save directory
        video_dir = self.save_dir / f"iter_{batch_idx:08d}" / f"sample_{sample_idx}"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle single frame vs sequence
        if pred.ndim == 3:  # (H, W, C)
            pred = pred[np.newaxis, ...]
            if inputs is not None and inputs.ndim == 3:
                inputs = inputs[np.newaxis, ...]
            if gt is not None and gt.ndim == 3:
                gt = gt[np.newaxis, ...]
        
        # Get video properties
        T, H, W, C = pred.shape
        fps = 10  # Default FPS
        
        # Save prediction video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        pred_path = video_dir / "pred.mp4"
        pred_writer = cv2.VideoWriter(str(pred_path), fourcc, fps, (W, H))
        for t in range(T):
            pred_writer.write(pred[t])
        pred_writer.release()
        
        # Save input video (if available)
        if inputs is not None:
            inputs_rgb = inputs[:, :, :, :3] if inputs.shape[-1] >= 3 else inputs
            input_path = video_dir / "input.mp4"
            input_writer = cv2.VideoWriter(str(input_path), fourcc, fps, (W, H))
            for t in range(min(T, inputs_rgb.shape[0])):
                input_writer.write(inputs_rgb[t])
            input_writer.release()
        
        # Save GT video (if available)
        if gt is not None:
            gt_path = video_dir / "gt.mp4"
            gt_writer = cv2.VideoWriter(str(gt_path), fourcc, fps, (W, H))
            for t in range(min(T, gt.shape[0])):
                gt_writer.write(gt[t])
            gt_writer.release()

