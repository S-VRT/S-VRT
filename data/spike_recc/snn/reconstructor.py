import os
import torch
import numpy as np
from ..middle_tfp.spike_utils import middleTFP

# Import SNNResidualEnhancer from the unified location
try:
    from models.architectures.vrt.snn import SNNResidualEnhancer
except ImportError:
    # Fallback: define locally if import fails (for backward compatibility)
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate, utils
    
    class SNNResidualEnhancer(nn.Module):
        def __init__(self, beta=0.9):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.lif1  = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.lif2  = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.conv3 = nn.Conv2d(32, 1, 3, padding=1)

        def forward(self, spike_seq):
            """
            spike_seq: [B, T, H, W]
            """
            utils.reset(self)
            mem1 = mem2 = None
            for t in range(spike_seq.shape[1]):
                x = spike_seq[:, t:t+1]  # [B,1,H,W]
                x = self.conv1(x)
                x, mem1 = self.lif1(x, mem1)
                x = self.conv2(x)
                x, mem2 = self.lif2(x, mem2)
            return self.conv3(x)

class SNNReconstructor:
    def __init__(self, checkpoint_path, spike_win=8, center=44, device='cpu', **kwargs):
        """
        SNN Reconstructor that enhances TFP result with SNN residuals.
        """
        self.device = torch.device(device)
        self.spike_win = spike_win
        self.center = center
        
        self.model = SNNResidualEnhancer().to(self.device)
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.eval()
            print(f"[SNNReconstructor] Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"[SNNReconstructor] Warning: Checkpoint {checkpoint_path} not found. Using random init.")

    def reconstruct(self, spike):
        """
        Args:
            spike (np.ndarray): Spike data [T, H, W]
        Returns:
            np.ndarray: Reconstructed image [H, W], normalized [0, 1]
        """
        # 1. Prepare TFP Base
        tfp = middleTFP(spike, self.center)
        tfp_min, tfp_max = tfp.min(), tfp.max()
        tfp = (tfp - tfp_min) / (tfp_max - tfp_min + 1e-6)
        
        # 2. Extract Spike Window
        half = self.spike_win // 2
        l = max(self.center - half, 0)
        r = min(self.center + half, spike.shape[0])
        spike_window = spike[l:r]
        
        if spike_window.shape[0] < self.spike_win:
            spike_window = np.pad(spike_window, ((0, self.spike_win - spike_window.shape[0]), (0,0), (0,0)))
            
        # 3. SNN Inference
        with torch.no_grad():
            spk_tensor = torch.from_numpy(spike_window).float().unsqueeze(0).to(self.device) # [1, T, H, W]
            res = self.model(spk_tensor) # [1, 1, H, W]
            res = res.squeeze().cpu().numpy()
            
        # 4. Combine
        out = np.clip(tfp + res, 0, 1)
        return out.astype(np.float32)

    def __call__(self, spike):
        return self.reconstruct(spike)
