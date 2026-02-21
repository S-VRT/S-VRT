import numpy as np
import torch
from .spike_utils import middleTFP

class MiddleTFPReconstructor:
    def __init__(self, spike_h=360, spike_w=640, spike_len=88, center=44, **kwargs):
        """
        MiddleTFP Reconstructor for Spike data.
        Args:
            spike_h (int): Spike camera height.
            spike_w (int): Spike camera width.
            spike_len (int): Total length of the spike sequence used for reconstruction.
            center (int): The center index for reconstruction.
        """
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.spike_len = spike_len
        self.center = center

    def reconstruct(self, spike):
        """
        Args:
            spike (np.ndarray or torch.Tensor): Spike data in shape [T, H, W]
        Returns:
            np.ndarray: Reconstructed image in shape [H, W], normalized to [0, 1]
        """
        if isinstance(spike, torch.Tensor):
            spike_np = spike.cpu().numpy()
        else:
            spike_np = spike

        # Ensure we are working with correct dimensions
        if spike_np.ndim == 4: # [B, T, H, W]
            spike_np = spike_np[0]
            
        recon = middleTFP(spike_np, self.center)
        
        # Normalize to [0, 1] as seen in test.py
        recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-6)
        
        return recon.astype(np.float32)

    def __call__(self, spike):
        return self.reconstruct(spike)
