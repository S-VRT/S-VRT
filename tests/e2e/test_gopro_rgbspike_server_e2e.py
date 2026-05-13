import copy

import pytest
import torch

from utils import utils_option as option


def _run_server_option_minimal_forward(
    server_option,
    require_paths_or_skip_fn,
    *,
    force_dual_pack_mode=False,
):
    from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike
    from models.architectures.vrt.vrt import VRT

    opt = copy.deepcopy(server_option)
    train_opt = opt.get("datasets", {}).get("train", {})

    if force_dual_pack_mode:
        train_opt["input_pack_mode"] = "dual"
        train_opt["keep_legacy_l"] = False
        compat_opt = train_opt.get("compat")
        if not isinstance(compat_opt, dict):
            compat_opt = {}
            train_opt["compat"] = compat_opt
        compat_opt["keep_legacy_L"] = False

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if force_dual_pack_mode:
        assert "L" not in sample
        assert "L_rgb" in sample and "L_spike" in sample
        x_rgb = sample["L_rgb"]
        x_spike = sample["L_spike"]
        x = torch.cat([x_rgb, x_spike], dim=1).unsqueeze(0)
    else:
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
    model = model.to(device)
    x = x.to(device)

    model.eval()
    flow_spike = sample.get("L_flow_spike", None)
    if flow_spike is not None:
        flow_spike = flow_spike.unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x, flow_spike=flow_spike)

    assert y.ndim == 5
    assert y.size(0) == 1
    assert y.size(1) == x.size(1)
    assert y.size(2) == 3


@pytest.mark.e2e
@pytest.mark.slow
def test_server_option_e2e_minimal_forward(server_option, require_paths_or_skip_fn):
    _run_server_option_minimal_forward(
        server_option,
        require_paths_or_skip_fn,
        force_dual_pack_mode=False,
    )


@pytest.mark.e2e
@pytest.mark.slow
def test_server_option_e2e_minimal_forward_forced_dual_path(
    server_option,
    require_paths_or_skip_fn,
):
    _run_server_option_minimal_forward(
        server_option,
        require_paths_or_skip_fn,
        force_dual_pack_mode=True,
    )


def test_dual_scale_temporal_mamba_raw_window_config_parses():
    opt = option.parse("options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json", is_train=True)

    assert opt["netG"]["fusion"]["operator"] == "dual_scale_temporal_mamba"
    assert opt["datasets"]["train"]["spike"]["representation"] == "raw_window"
    assert opt["datasets"]["test"]["spike"]["representation"] == "raw_window"
    assert opt["datasets"]["train"]["spike_channels"] == 21
    assert opt["datasets"]["train"]["spike_flow"]["subframes"] == opt["datasets"]["train"]["spike_channels"]
    assert opt["datasets"]["test"]["spike_flow"]["subframes"] == opt["datasets"]["test"]["spike_channels"]
    assert opt["netG"]["input"]["raw_ingress_chans"] == 24
