import torch
from typing import Optional, List
from ..base import OpticalFlowModule
from .models.scflow import SCFlow

class SCFlowWrapper(OpticalFlowModule):
    """
    Wrapper for SCFlow that accepts pre-encoded spike sequences.
    Expected input shape: (B, 25, H, W) where 25 is the spike sequence length.
    """
    def __init__(self, checkpoint: Optional[str] = None, device: str = 'cpu', batch_norm: bool = False, dt: int = 10, **kwargs):
        super().__init__()
        self.input_type = 'spike'
        self.device = torch.device(device)
        self.dt = dt
        
        # Initialize model
        # Note: scflow factory expects a dict with 'state_dict' or None
        data = None
        if checkpoint:
            try:
                state = torch.load(checkpoint, map_location='cpu')
                if 'state_dict' in state:
                    data = state
                else:
                    data = {'state_dict': state}
            except Exception as e:
                print(f"Warning: Failed to load SCFlow checkpoint from {checkpoint}: {e}")
        
        from .models.scflow import scflow
        self.model = scflow(data=data, batchNorm=batch_norm)
        self.model.to(self.device)
        self.model.eval()


    def _validate_spike_pair(self, spk1: torch.Tensor, spk2: torch.Tensor) -> None:
        for name, tensor in (("spk1", spk1), ("spk2", spk2)):
            if tensor.ndim != 4:
                raise ValueError(
                    f"SCFlow expects {name} ndim=4 [B,25,H,W], got shape {tuple(tensor.shape)}"
                )
            if tensor.size(1) != 25:
                raise ValueError(
                    f"SCFlow expects {name} channels=25, got {tensor.size(1)} with shape {tuple(tensor.shape)}"
                )

    def forward(self, spk1: torch.Tensor, spk2: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for SCFlow.
        Args:
            spk1, spk2: Spike sequences of shape (B, 25, H, W)
        Returns:
            List of 4 flows: [full_res, 1/2_res, 1/4_res, 1/8_res]
        """
        self._validate_spike_pair(spk1, spk2)

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = self.device

        spk1 = spk1.to(device=device, dtype=torch.float32)
        spk2 = spk2.to(device=device, dtype=torch.float32)

        b, _, h, w = spk1.shape
        flow_init = torch.zeros(b, 2, h, w, device=device, dtype=torch.float32)

        with torch.autocast("cuda", enabled=False), torch.set_grad_enabled(self._should_track_gradients()):
            flows, _ = self.model(spk1, spk2, flow_init, dt=self.dt)

        return flows[:4]

    def load_checkpoint(self, path: str) -> None:
        if not path:
            return
        state = torch.load(path, map_location='cpu')
        if 'state_dict' in state:
            self.model.load_state_dict(state['state_dict'])
        else:
            self.model.load_state_dict(state)
