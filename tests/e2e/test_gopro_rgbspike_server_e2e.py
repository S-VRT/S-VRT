import copy

import pytest
import torch


@pytest.mark.e2e
@pytest.mark.slow
def test_server_option_e2e_minimal_forward(server_option, require_paths_or_skip_fn):
    from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike
    from models.architectures.vrt.vrt import VRT

    opt = copy.deepcopy(server_option)
    train_opt = opt.get("datasets", {}).get("train", {})

    require_paths_or_skip_fn(
        [
            train_opt.get("dataroot_gt"),
            train_opt.get("dataroot_lq"),
            train_opt.get("dataroot_spike"),
        ],
        reason_prefix="server dataset not ready",
    )

    dataset = TrainDatasetRGBSpike(train_opt)
    sample = dataset[0]

    if "L" in sample:
        x = sample["L"].unsqueeze(0)
    else:
        x_rgb = sample["L_rgb"]
        x_spike = sample["L_spike"]
        x = torch.cat([x_rgb, x_spike], dim=1).unsqueeze(0)

    net_cfg = opt.get("netG", {})
    depths = net_cfg.get("depths") or [1] * 8
    embed_dims = net_cfg.get("embed_dims") or [16] * len(depths)
    num_heads = net_cfg.get("num_heads") or [1] * len(depths)

    model = VRT(
        upscale=net_cfg.get("upscale", 1),
        in_chans=net_cfg.get("in_chans", x.size(2)),
        out_chans=net_cfg.get("out_chans", 3),
        img_size=[x.size(1), x.size(3), x.size(4)],
        window_size=net_cfg.get("window_size", [2, 8, 8]),
        depths=depths,
        indep_reconsts=net_cfg.get("indep_reconsts", []),
        embed_dims=embed_dims,
        num_heads=num_heads,
        pa_frames=net_cfg.get("pa_frames", 2),
        use_flash_attn=False,
        optical_flow=net_cfg.get(
            "optical_flow",
            {"module": "spynet", "checkpoint": None, "params": {}},
        ),
        opt=opt,
    )

    model.eval()
    with torch.no_grad():
        y = model(x)

    assert y.ndim == 5
    assert y.size(0) == 1
    assert y.size(1) == x.size(1)
    assert y.size(2) == 3
