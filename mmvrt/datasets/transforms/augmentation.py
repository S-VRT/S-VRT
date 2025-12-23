"""Augmentation transforms for restoration data."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from mmvrt.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomCrop:
    """Random crop for paired RGB and spike data.
    
    Crops RGB frames, GT frames, and spike voxels together with corresponding locations.
    
    Args:
        gt_patch_size: GT patch size (height/width)
        scale: Scale factor between GT and LQ
    """
    
    def __init__(
        self,
        gt_patch_size: int = 256,
        scale: int = 1,
    ):
        self.gt_patch_size = gt_patch_size
        self.scale = scale
        self.lq_patch_size = gt_patch_size // scale
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random crop.
        
        Args:
            results: Dictionary containing 'lq', 'gt', and optionally 'spike_voxel'
        
        Returns:
            Dictionary with cropped data
        """
        # Get reference frame for size checking
        if 'lq' in results:
            if isinstance(results['lq'], list):
                ref_frame = results['lq'][0]
            else:
                ref_frame = results['lq']
        elif 'gt' in results:
            if isinstance(results['gt'], list):
                ref_frame = results['gt'][0]
            else:
                ref_frame = results['gt']
        else:
            return results
        
        # Determine input type
        if isinstance(ref_frame, torch.Tensor):
            input_type = 'Tensor'
            if ref_frame.ndim == 4:  # (C, H, W) or (T, C, H, W)
                h, w = ref_frame.shape[-2:]
            else:  # (H, W, C)
                h, w = ref_frame.shape[:2]
        else:
            input_type = 'Numpy'
            if ref_frame.ndim == 3:  # (H, W, C)
                h, w = ref_frame.shape[:2]
            else:  # (C, H, W)
                h, w = ref_frame.shape[-2:]
        
        # Check if crop is possible
        if h < self.lq_patch_size or w < self.lq_patch_size:
            # Cannot crop, return as-is
            return results
        
        # Randomly choose top and left coordinates for LQ patch
        top = random.randint(0, h - self.lq_patch_size)
        left = random.randint(0, w - self.lq_patch_size)
        
        # Crop LQ frames
        if 'lq' in results:
            results['lq'] = self._crop_frames(results['lq'], top, left, input_type, is_lq=True)
        
        # Crop GT frames
        if 'gt' in results:
            top_gt = int(top * self.scale)
            left_gt = int(left * self.scale)
            results['gt'] = self._crop_frames(results['gt'], top_gt, left_gt, input_type, is_lq=False)
        
        # Crop spike voxels (resize to match cropped RGB size)
        if 'spike_voxel' in results:
            cropped_h = self.lq_patch_size
            cropped_w = self.lq_patch_size
            results['spike_voxel'] = self._crop_spike_voxels(
                results['spike_voxel'], 
                top, 
                left, 
                cropped_h, 
                cropped_w,
                input_type
            )
        
        return results
    
    def _crop_frames(
        self, 
        frames: Union[List, np.ndarray, torch.Tensor], 
        top: int, 
        left: int, 
        input_type: str,
        is_lq: bool
    ) -> Union[List, np.ndarray, torch.Tensor]:
        """Crop frames (RGB or GT)."""
        patch_size = self.lq_patch_size if is_lq else self.gt_patch_size
        
        if isinstance(frames, list):
            cropped = []
            for frame in frames:
                cropped.append(self._crop_single_frame(frame, top, left, patch_size, input_type))
            return cropped
        else:
            return self._crop_single_frame(frames, top, left, patch_size, input_type)
    
    def _crop_single_frame(
        self,
        frame: Union[np.ndarray, torch.Tensor],
        top: int,
        left: int,
        patch_size: int,
        input_type: str
    ) -> Union[np.ndarray, torch.Tensor]:
        """Crop a single frame."""
        if input_type == 'Tensor':
            if frame.ndim == 4:  # (T, C, H, W) or (C, H, W) with batch
                return frame[..., top:top + patch_size, left:left + patch_size]
            elif frame.ndim == 3:  # (C, H, W)
                return frame[:, top:top + patch_size, left:left + patch_size]
            else:  # (H, W, C) - should not happen for Tensor
                return frame[top:top + patch_size, left:left + patch_size, ...]
        else:  # Numpy
            if frame.ndim == 3:
                if frame.shape[2] in [1, 3, 4]:  # (H, W, C)
                    return frame[top:top + patch_size, left:left + patch_size, ...]
                else:  # (C, H, W)
                    return frame[:, top:top + patch_size, left:left + patch_size]
            else:  # (H, W)
                return frame[top:top + patch_size, left:left + patch_size]
    
    def _crop_spike_voxels(
        self,
        spike_voxels: Union[List, np.ndarray],
        top: int,
        left: int,
        target_h: int,
        target_w: int,
        input_type: str
    ) -> Union[List, np.ndarray]:
        """Crop spike voxels to match RGB crop location.

        NOTE: This transform only crops (augmentation). Spatial resizing / scaling
        between spike and RGB resolutions is handled by the data preprocessor.
        """
        if isinstance(spike_voxels, list):
            cropped = []
            for voxel in spike_voxels:
                cropped.append(self._crop_single_voxel(voxel, top, left, target_h, target_w))
            return cropped
        else:
            return self._crop_single_voxel(spike_voxels, top, left, target_h, target_w)
    
    def _resize_spike_voxel(
        self,
        spike_voxel: np.ndarray,
        target_h: int,
        target_w: int
    ) -> np.ndarray:
        """(DEPRECATED) kept for compatibility; transforms should not resize.

        The transform pipeline no longer performs resizing — the preprocessor is
        responsible for spatial alignment between modalities. For safety we
        provide a simple crop fallback if callers still call this.
        """
        # Fallback: crop center region of spike_voxel if dimensions differ.
        if spike_voxel is None:
            return spike_voxel
        if spike_voxel.ndim == 3:
            # Normalize to (C, H, W)
            if spike_voxel.shape[0] < spike_voxel.shape[-1]:
                # likely (C, H, W)
                c, h, w = spike_voxel.shape
                top = max(0, (h - target_h) // 2)
                left = max(0, (w - target_w) // 2)
                return spike_voxel[:, top:top + target_h, left:left + target_w]
            else:
                # (H, W, C)
                h, w, c = spike_voxel.shape
                top = max(0, (h - target_h) // 2)
                left = max(0, (w - target_w) // 2)
                cropped = spike_voxel[top:top + target_h, left:left + target_w, :]
                # return as (C, H, W)
                return np.transpose(cropped, (2, 0, 1))
        else:
            # 2D array - crop center
            h, w = spike_voxel.shape
            top = max(0, (h - target_h) // 2)
            left = max(0, (w - target_w) // 2)
            return spike_voxel[top:top + target_h, left:left + target_w]

    def _crop_single_voxel(
        self,
        spike_voxel: np.ndarray,
        top: int,
        left: int,
        target_h: int,
        target_w: int
    ) -> np.ndarray:
        """Crop a single spike voxel without resizing."""
        if spike_voxel is None:
            return spike_voxel
        if spike_voxel.ndim == 3:
            # Support both (C, H, W) and (H, W, C)
            if spike_voxel.shape[0] <= spike_voxel.shape[2]:
                # (C, H, W)
                c, h, w = spike_voxel.shape
                top = max(0, min(top, h - 1))
                left = max(0, min(left, w - 1))
                return spike_voxel[:, top:top + target_h, left:left + target_w]
            else:
                # (H, W, C)
                h, w, c = spike_voxel.shape
                top = max(0, min(top, h - 1))
                left = max(0, min(left, w - 1))
                cropped = spike_voxel[top:top + target_h, left:left + target_w, :]
                return np.transpose(cropped, (2, 0, 1))
        else:
            # 2D array
            h, w = spike_voxel.shape
            top = max(0, min(top, h - 1))
            left = max(0, min(left, w - 1))
            return spike_voxel[top:top + target_h, left:left + target_w]


@TRANSFORMS.register_module()
class RandomFlip:
    """Random flip augmentation (horizontal, vertical, rotation).
    
    Args:
        hflip: Whether to apply horizontal flip
        vflip: Whether to apply vertical flip (rotation)
        rot90: Whether to apply 90-degree rotation
    """
    
    def __init__(
        self,
        hflip: bool = True,
        vflip: bool = False,
        rot90: bool = False,
    ):
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random flip.
        
        Args:
            results: Dictionary containing 'lq', 'gt', and optionally 'spike_voxel'
        
        Returns:
            Dictionary with flipped data
        """
        # Determine flip operations
        do_hflip = self.hflip and random.random() < 0.5
        do_vflip = self.vflip and random.random() < 0.5
        do_rot90 = self.rot90 and random.random() < 0.5
        
        if not (do_hflip or do_vflip or do_rot90):
            return results
        
        # Flip LQ frames
        if 'lq' in results:
            results['lq'] = self._flip_frames(results['lq'], do_hflip, do_vflip, do_rot90)
        
        # Flip GT frames
        if 'gt' in results:
            results['gt'] = self._flip_frames(results['gt'], do_hflip, do_vflip, do_rot90)
        
        # Flip spike voxels
        if 'spike_voxel' in results:
            results['spike_voxel'] = self._flip_spike_voxels(
                results['spike_voxel'], 
                do_hflip, 
                do_vflip, 
                do_rot90
            )
        
        return results
    
    def _flip_frames(
        self,
        frames: Union[List, np.ndarray, torch.Tensor],
        hflip: bool,
        vflip: bool,
        rot90: bool
    ) -> Union[List, np.ndarray, torch.Tensor]:
        """Flip frames."""
        if isinstance(frames, list):
            flipped = []
            for frame in frames:
                flipped.append(self._flip_single_frame(frame, hflip, vflip, rot90))
            return flipped
        else:
            return self._flip_single_frame(frames, hflip, vflip, rot90)
    
    def _flip_single_frame(
        self,
        frame: Union[np.ndarray, torch.Tensor],
        hflip: bool,
        vflip: bool,
        rot90: bool
    ) -> Union[np.ndarray, torch.Tensor]:
        """Flip a single frame."""
        if isinstance(frame, torch.Tensor):
            # Tensor format: (C, H, W) or (T, C, H, W)
            if hflip:
                frame = torch.flip(frame, dims=[-1])  # Flip width
            if vflip:
                frame = torch.flip(frame, dims=[-2])  # Flip height
            if rot90:
                frame = frame.transpose(-2, -1)  # Transpose H and W
            return frame
        else:
            # Numpy format: (H, W, C) or (C, H, W)
            if frame.ndim == 3:
                if frame.shape[2] in [1, 3, 4]:  # (H, W, C)
                    if hflip:
                        frame = cv2.flip(frame, 1)
                    if vflip:
                        frame = cv2.flip(frame, 0)
                    if rot90:
                        frame = frame.transpose(1, 0, 2)
                else:  # (C, H, W)
                    if hflip:
                        frame = np.flip(frame, axis=2)
                    if vflip:
                        frame = np.flip(frame, axis=1)
                    if rot90:
                        frame = frame.transpose(0, 2, 1)
            else:  # (H, W)
                if hflip:
                    frame = cv2.flip(frame, 1)
                if vflip:
                    frame = cv2.flip(frame, 0)
                if rot90:
                    frame = frame.transpose(1, 0)
            return frame
    
    def _flip_spike_voxels(
        self,
        spike_voxels: Union[List, np.ndarray],
        hflip: bool,
        vflip: bool,
        rot90: bool
    ) -> Union[List, np.ndarray]:
        """Flip spike voxels."""
        if isinstance(spike_voxels, list):
            flipped = []
            for voxel in spike_voxels:
                flipped.append(self._flip_single_voxel(voxel, hflip, vflip, rot90))
            return flipped
        else:
            return self._flip_single_voxel(spike_voxels, hflip, vflip, rot90)
    
    def _flip_single_voxel(
        self,
        voxel: np.ndarray,
        hflip: bool,
        vflip: bool,
        rot90: bool
    ) -> np.ndarray:
        """Flip a single spike voxel."""
        # Voxel format: (C, H, W)
        if voxel.ndim == 3:
            if hflip:
                voxel = np.flip(voxel, axis=2)  # Flip width
            if vflip:
                voxel = np.flip(voxel, axis=1)  # Flip height
            if rot90:
                voxel = voxel.transpose(0, 2, 1)  # Transpose H and W
        return voxel

