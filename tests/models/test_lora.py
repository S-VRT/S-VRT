import math
import copy
import pytest
import torch
import torch.nn as nn

from models.lora import LoRALinear, inject_lora, merge_lora


def test_lora_initial_output_equals_base():
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    lora = LoRALinear(copy.deepcopy(base), rank=4, alpha=8)
    x = torch.randn(2, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)
