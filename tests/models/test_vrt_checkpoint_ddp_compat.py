import torch

from models.architectures.vrt.stages import TMSA


def test_tmsa_checkpoint_calls_use_non_reentrant(monkeypatch):
    calls = []

    def fake_checkpoint(function, *args, **kwargs):
        calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", fake_checkpoint)
    block = TMSA(
        dim=8,
        input_resolution=(2, 4, 4),
        num_heads=2,
        window_size=(2, 4, 4),
        shift_size=(0, 0, 0),
        use_checkpoint_attn=True,
        use_checkpoint_ffn=True,
        use_flash_attn=False,
    )
    x = torch.randn(1, 2, 4, 4, 8)

    block(x, None)

    assert calls
    assert all(call.get("use_reentrant") is False for call in calls)
