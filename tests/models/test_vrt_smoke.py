import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.smoke
@pytest.mark.integration
def test_vrt_smoke_forward_and_one_train_iter():
    """Smoke contract: VRT supports one forward + one optimizer step."""
    from models.architectures.vrt.vrt import VRT

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VRT(
        upscale=1,
        in_chans=3,
        out_chans=3,
        img_size=[6, 64, 64],
        window_size=[6, 8, 8],
        depths=[1] * 8,
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
    ).to(device)
    model.train()

    x = torch.randn(1, 6, 3, 64, 64, device=device)
    out = model(x)
    assert out.shape == (1, 6, 3, 64, 64)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    target = torch.randn_like(out)
    loss = F.mse_loss(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert bool(torch.isfinite(loss).item())
