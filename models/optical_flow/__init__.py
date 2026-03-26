from typing import Optional

from .base import OpticalFlowModule
from .spynet import SpyNetWrapper
from .sea_raft import SeaRaftWrapper
from .scflow.wrapper import SCFlowWrapper


def create_optical_flow(module: str = 'spynet',
                        checkpoint: Optional[str] = None,
                        device: str = 'cpu',
                        **kwargs) -> OpticalFlowModule:
    """Factory that returns an OpticalFlowModule instance.

    Supported modules: 'spynet', 'sea_raft', 'scflow'.
    """
    module_name = (module or 'spynet').lower()
    if module_name in ('spynet', 'spy'):
        model = SpyNetWrapper(checkpoint=checkpoint, device=device, **kwargs)
        return model
    elif module_name in ('sea_raft', 'searaft'):
        model = SeaRaftWrapper(checkpoint=checkpoint, device=device, **kwargs)
        return model
    elif module_name in ('scflow', 'spike_flow'):
        model = SCFlowWrapper(checkpoint=checkpoint, device=device, **kwargs)
        return model
    else:
        raise ValueError(f'Unknown optical flow module: {module}')