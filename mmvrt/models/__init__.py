"""Model components package.

This package aggregates commonly used submodules so callers can import:
    from mmvrt.models import backbones, restorers, losses, data_preprocessors

Keep this file lightweight — submodules live in their own files to follow
the OpenMMLab-style modular layout.
"""

from . import backbones  # noqa: F401
from . import restorers  # noqa: F401
from . import heads      # noqa: F401
from . import losses     # noqa: F401
from . import layers     # noqa: F401
from . import motion     # noqa: F401
from . import data_preprocessors  # noqa: F401

__all__ = [
    'backbones',
    'restorers',
    'heads',
    'losses',
    'layers',
    'motion',
    'data_preprocessors',
]

"""Model components (restorer/backbone/loss/...)."""

