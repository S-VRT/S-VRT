from models.flows import compute_flows_2frames
from .vrt import VRT
from models.registry import register_model

# Register default VRT builder
@register_model('VRT')
def build_vrt(**kwargs):
    return VRT(**kwargs)

__all__ = ['compute_flows_2frames', 'VRT', 'build_vrt']


