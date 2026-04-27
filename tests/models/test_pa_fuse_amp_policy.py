import pytest
import torch


class _IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x


class _PaFuseProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dtype = None
        self.autocast_enabled = None

    def forward(self, x):
        self.input_dtype = x.dtype
        self.autocast_enabled = torch.is_autocast_enabled(x.device.type)
        return x[..., :1]


def _make_stage(policy="fp32"):
    from models.architectures.vrt.stages import Stage

    stage = Stage(
        in_dim=1,
        dim=1,
        input_resolution=(2, 4, 4),
        depth=0,
        num_heads=1,
        window_size=(2, 4, 4),
        pa_frames=2,
        reshape="none",
        pa_fuse_amp_policy=policy,
    )
    stage.reshape = _IdentityModule()
    stage.residual_group1 = _IdentityModule()
    stage.residual_group2 = _IdentityModule()
    stage.linear1 = _IdentityModule()
    stage.linear2 = _IdentityModule()
    stage.get_aligned_feature_2frames = lambda x, *_args: (torch.zeros_like(x), torch.zeros_like(x))
    stage.pa_fuse = _PaFuseProbe()
    return stage


def test_pa_fuse_defaults_to_fp32_policy():
    stage = _make_stage()

    assert stage.pa_fuse_amp_policy == "fp32"


def test_pa_fuse_autocast_policy_keeps_outer_autocast_enabled_on_cpu():
    stage = _make_stage(policy="autocast")
    x = torch.randn(1, 1, 2, 4, 4)
    flows = [torch.zeros(1, 2, 4, 4)]

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        stage(x, flows, flows)

    assert stage.pa_fuse.autocast_enabled is True


def test_pa_fuse_fp32_policy_disables_outer_autocast_on_cpu():
    stage = _make_stage(policy="fp32")
    x = torch.randn(1, 1, 2, 4, 4)
    flows = [torch.zeros(1, 2, 4, 4)]

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        stage(x, flows, flows)

    assert stage.pa_fuse.autocast_enabled is False
    assert stage.pa_fuse.input_dtype == torch.float32


def test_pa_fuse_invalid_policy_fails_fast():
    with pytest.raises(ValueError, match="pa_fuse_amp_policy"):
        _make_stage(policy="bad")


def test_pa_fuse_profile_range_is_config_gated():
    class _TimerStub:
        def __init__(self, record_ranges=False):
            self.record_ranges = record_ranges
            self.range_names = []

        def profile_range(self, name):
            if self.record_ranges:
                self.range_names.append(name)

            class _Ctx:
                def __enter__(_self):
                    return None

                def __exit__(_self, *_exc):
                    return False

            return _Ctx()

    stage = _make_stage(policy="autocast")
    x = torch.randn(1, 1, 2, 4, 4)
    flows = [torch.zeros(1, 2, 4, 4)]

    timer = _TimerStub(record_ranges=False)
    stage.set_timer(timer)
    stage(x, flows, flows)
    assert timer.range_names == []

    timer = _TimerStub(record_ranges=True)
    stage.set_timer(timer)
    stage(x, flows, flows)
    assert timer.range_names == ["pa_fuse"]


def test_vrt_passes_pa_fuse_amp_policy_to_all_stages():
    from models.architectures.vrt.vrt import VRT

    model = VRT(
        upscale=1,
        in_chans=3,
        out_chans=3,
        img_size=[2, 16, 16],
        window_size=[2, 4, 4],
        depths=[0] * 8,
        embed_dims=[8] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        pa_fuse_amp_policy="autocast",
    )

    for index in range(1, 8):
        assert getattr(model, f"stage{index}").pa_fuse_amp_policy == "autocast"
