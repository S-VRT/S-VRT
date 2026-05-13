from models.flows import compute_flows_2frames
from .vrt import VRT
from models.registry import register_model

# Register default VRT builder
@register_model('VRT')
def build_vrt(**kwargs):
    return VRT(**kwargs)

# Optional SNN functionality (requires extra dependency: snntorch)
try:
    from .snn import (
        SNNResidualEnhancer,
        GoProSpikeSNNDataset,
        train as train_snn,
        test as test_snn,
        evaluate as evaluate_snn,
        sobel_l1_loss,
        edge_loss,
    )
    _SNN_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name != 'snntorch':
        raise
    _SNN_AVAILABLE = False

__all__ = [
    'compute_flows_2frames',
    'VRT',
    'build_vrt',
]

if _SNN_AVAILABLE:
    __all__.extend([
        'SNNResidualEnhancer',
        'GoProSpikeSNNDataset',
        'train_snn',
        'test_snn',
        'evaluate_snn',
        'sobel_l1_loss',
        'edge_loss',
    ])


