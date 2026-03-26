import torch
import torch.nn as nn


class OpticalFlowModule(nn.Module):
    """Adapter base class for optical flow backends.

    Required methods:
      - forward(frame1: Tensor, frame2: Tensor) -> Tensor  # returns flow or list of multiscale flows
      - load_checkpoint(path: str) -> None
    """
    def __init__(self):
        super().__init__()
        self.input_type = 'rgb'  # Default to 'rgb', can be overridden by subclasses (e.g., 'spike')

    @staticmethod
    def preprocess_frames(frame1: torch.Tensor, frame2: torch.Tensor, target_format='rgb'):
        """Convert frames from BGR 0-1 (OpenCV format) to target format.

        Args:
            frame1, frame2: tensors with shape (..., 3, H, W), values in [0, 1] as BGR
            target_format: 'rgb' for RGB [0,1], 'rgb_255' for RGB [0,255], 'rgb_norm' for RGB normalized

        Returns:
            Preprocessed frame1, frame2
        """
        # Convert BGR to RGB
        frame1_rgb = frame1.flip(-3)  # flip channel dimension: BGR -> RGB
        frame2_rgb = frame2.flip(-3)

        if target_format == 'rgb':
            # RGB [0, 1] - for SpyNet with ImageNet normalization
            return frame1_rgb, frame2_rgb
        elif target_format == 'rgb_255':
            # RGB [0, 255] - for SeaRaft
            return frame1_rgb * 255.0, frame2_rgb * 255.0
        elif target_format == 'rgb_norm':
            # RGB with ImageNet normalization - for SpyNet
            mean = torch.tensor([0.485, 0.456, 0.406], device=frame1.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=frame1.device).view(1, 3, 1, 1)
            frame1_norm = (frame1_rgb - mean) / std
            frame2_norm = (frame2_rgb - mean) / std
            return frame1_norm, frame2_norm
        else:
            raise ValueError(f"Unknown target_format: {target_format}")

    @staticmethod
    def postprocess_flow(flow: torch.Tensor, max_displacement=None, smoothing_kernel_size=None):
        """Post-process optical flow for robustness.

        Args:
            flow: tensor with shape (..., 2, H, W)
            max_displacement: maximum allowed displacement (clip outliers)
            smoothing_kernel_size: kernel size for median filtering (reduce noise)

        Returns:
            Processed flow tensor
        """
        processed_flow = flow.clone()

        # Clip displacement to prevent extreme outliers
        if max_displacement is not None:
            displacement = torch.sqrt(processed_flow[:, 0:1, :, :] ** 2 + processed_flow[:, 1:2, :, :] ** 2)
            mask = displacement > max_displacement
            scale = max_displacement / (displacement + 1e-8)
            scale = torch.clamp(scale, max=1.0)
            processed_flow[:, 0:1, :, :] *= scale
            processed_flow[:, 1:2, :, :] *= scale

        # Optional median filtering for noise reduction
        if smoothing_kernel_size is not None and smoothing_kernel_size > 1:
            # Simple box filtering as approximation (median would be better but slower)
            padding = smoothing_kernel_size // 2
            processed_flow = torch.nn.functional.conv2d(
                torch.nn.functional.pad(processed_flow, (padding, padding, padding, padding), mode='replicate'),
                torch.ones_like(processed_flow[:, :1, :smoothing_kernel_size, :smoothing_kernel_size]) / (smoothing_kernel_size ** 2),
                groups=2
            )

        return processed_flow

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError


