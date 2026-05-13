import os
import sys

# Add the current directory to Python path so DCNv4 can find its extensions
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .functions import DCNv4Function, FlashDeformAttnFunction
from .modules import DCNv4, FlashDeformAttn