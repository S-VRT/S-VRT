import torch

from models.architectures.vrt.vrt import VRT


def test_vrt_builds_with_fusion_config():
    opt = {
        "netG": {
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "concat",
                "out_chans": 3,
                "operator_params": {},
            }
        }
    }

    model = VRT(
        upscale=1,
        in_chans=3,
        out_chans=3,
        img_size=[2, 8, 8],
        window_size=[2, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=0,
        use_flash_attn=False,
        opt=opt,
    )

    assert model is not None
    assert hasattr(model, "fusion_enabled")
    assert model.fusion_enabled is True
    assert hasattr(model, "fusion_operator")
    assert model.fusion_operator is not None
    assert hasattr(model, "fusion_adapter")
    assert model.fusion_adapter is not None
