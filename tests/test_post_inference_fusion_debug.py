from types import SimpleNamespace

import argparse
import numpy as np
import torch

import main_test_vrt


def _minimal_args(tmp_path):
    ckpt_path = tmp_path / "1_G.pth"
    torch.save({"params": {}}, ckpt_path)
    cfg = {
        "is_train": False,
        "path": {
            "pretrained_netG": str(ckpt_path),
            "pretrained_netE": None,
            "images": str(tmp_path / "images"),
        },
        "netG": {
            "net_type": "vrt",
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 7,
            },
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "mamba",
                "out_chans": 3,
                "operator_params": {},
                "debug": {
                    "enable": True,
                    "save_images": True,
                    "subdir": "fusion_post_infer",
                    "source_view": "main",
                },
            },
            "upscale": 1,
            "in_chans": 7,
            "out_chans": 3,
            "img_size": [2, 8, 8],
            "window_size": [2, 4, 4],
            "depths": [1] * 8,
            "indep_reconsts": [],
            "embed_dims": [16] * 8,
            "num_heads": [1] * 8,
            "optical_flow": {"module": "spynet", "checkpoint": None, "params": {}},
            "pa_frames": 2,
            "deformable_groups": 16,
            "nonblind_denoising": False,
            "use_checkpoint_attn": False,
            "use_checkpoint_ffn": False,
            "no_checkpoint_attn_blocks": [],
            "no_checkpoint_ffn_blocks": [],
            "dcn_type": "DCNv2",
            "dcn_apply_softmax": False,
        },
    }
    return SimpleNamespace(
        cfg=cfg,
        path_cfg=cfg["path"],
        netG_cfg=cfg["netG"],
        task="debug",
        folder_lq=str(tmp_path),
        folder_gt=str(tmp_path),
        rank=0,
        fusion_debug=True,
        fusion_debug_dir=None,
        fusion_debug_subdir="fusion_post_infer",
        fusion_debug_source_view=None,
        fusion_debug_max_batches=1,
    )


def test_prepare_model_dataset_passes_full_opt_to_vrt(monkeypatch, tmp_path):
    captured = {}

    class _FakeVRT(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

        def load_state_dict(self, state_dict, strict=False):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    monkeypatch.setattr(main_test_vrt, "net", _FakeVRT)

    args = _minimal_args(tmp_path)
    args.netG_cfg["in_chans"] = 3
    args.netG_cfg["input"]["raw_ingress_chans"] = 7
    main_test_vrt.prepare_model_dataset(args)

    assert captured["opt"]["netG"]["fusion"]["enable"] is True
    assert captured["in_chans"] == 7


def test_assert_lq_channels_uses_raw_ingress_chans_for_fusion_input():
    main_test_vrt._assert_lq_channels(
        torch.rand(1, 2, 12, 4, 4),
        "Test Video Input",
        {
            "in_chans": 7,
            "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 12},
        },
    )


def test_merge_config_enables_post_inference_fusion_debug_from_json(tmp_path):
    cfg = _minimal_args(tmp_path).cfg
    cfg["netG"]["fusion"]["debug"].update(
        {
            "enable": True,
            "save_images": True,
            "subdir": "fusion_phase1_last_train",
            "source_view": "exec",
            "max_batches": 3,
        }
    )
    args = argparse.Namespace(
        task=None,
        folder_lq=None,
        folder_gt=None,
        sigma=None,
        num_workers=None,
        tile=None,
        tile_overlap=None,
        save_result=False,
        fusion_debug=False,
        fusion_debug_dir=None,
        fusion_debug_subdir=None,
        fusion_debug_source_view=None,
        fusion_debug_max_batches=None,
    )

    main_test_vrt._merge_config_into_args(args, cfg)

    assert args.fusion_debug is True
    assert args.fusion_debug_subdir == "fusion_phase1_last_train"
    assert args.fusion_debug_source_view == "exec"
    assert args.fusion_debug_max_batches == 3


def test_merge_config_keeps_explicit_cli_fusion_debug_overrides(tmp_path):
    cfg = _minimal_args(tmp_path).cfg
    cfg["netG"]["fusion"]["debug"].update(
        {
            "enable": True,
            "save_images": True,
            "subdir": "from_config",
            "source_view": "main",
            "max_batches": 3,
        }
    )
    args = argparse.Namespace(
        task=None,
        folder_lq=None,
        folder_gt=None,
        sigma=None,
        num_workers=None,
        tile=None,
        tile_overlap=None,
        save_result=False,
        fusion_debug=False,
        fusion_debug_dir=None,
        fusion_debug_subdir="from_cli",
        fusion_debug_source_view="exec",
        fusion_debug_max_batches=5,
    )

    main_test_vrt._merge_config_into_args(args, cfg)

    assert args.fusion_debug is True
    assert args.fusion_debug_subdir == "from_cli"
    assert args.fusion_debug_source_view == "exec"
    assert args.fusion_debug_max_batches == 5


def test_dump_post_inference_fusion_debug_uses_cached_fusion_outputs(tmp_path):
    calls = []

    class _FakeDumper:
        def __init__(self, opt):
            self.opt = opt
            self.enabled = True
            self.save_images = True

        def dump_tensor(self, **kwargs):
            calls.append(kwargs)
            return True

    class _FakeModel:
        _last_fusion_main = torch.rand(1, 2, 3, 8, 8)
        _last_fusion_exec = torch.rand(1, 8, 3, 8, 8)
        _last_fusion_meta = {"frame_contract": "expanded", "spike_bins": 4}

    args = _minimal_args(tmp_path)
    batch = {
        "folder": ["GOPR0001"],
        "lq_path": [["/data/GOPR0001/000001.png", "/data/GOPR0001/000002.png"]],
        "H": torch.rand(1, 2, 3, 8, 8),
    }

    dumped = main_test_vrt.dump_post_inference_fusion_debug(
        model=_FakeModel(),
        args=args,
        batch=batch,
        batch_idx=0,
        dumper_cls=_FakeDumper,
    )

    assert dumped is True
    assert calls[0]["fusion_main"].shape == (1, 2, 3, 8, 8)
    assert calls[0]["fusion_meta"]["spike_bins"] == 4
    assert calls[0]["folder"] == ["GOPR0001"]
    assert calls[0]["lq_paths"] == batch["lq_path"]


def test_dump_post_inference_fusion_debug_skips_non_first_batch(tmp_path):
    class _FailingDumper:
        def __init__(self, opt):
            raise AssertionError("dumper should not be created for skipped batches")

    args = _minimal_args(tmp_path)
    args.fusion_debug_max_batches = 1

    assert main_test_vrt.dump_post_inference_fusion_debug(
        model=SimpleNamespace(),
        args=args,
        batch={"folder": ["GOPR0002"]},
        batch_idx=1,
        dumper_cls=_FailingDumper,
    ) is False


def test_load_lazy_flow_patch_uses_temporal_offset_and_subframes(monkeypatch, tmp_path):
    loaded = []

    def _fake_load(base_path, artifact_format, num_subframes, spike_h, spike_w):
        loaded.append(str(base_path))
        value = len(loaded)
        return np.full((num_subframes, 25, spike_h, spike_w), value, dtype=np.float32)

    monkeypatch.setattr(main_test_vrt, "load_encoding25_artifact_with_shape", _fake_load)
    meta = {
        "flow_clip_dir": str(tmp_path),
        "frame_names": ["000001", "000002", "000003"],
        "format": "npy",
        "subframes": 2,
        "source_h": 4,
        "source_w": 4,
    }

    patch = main_test_vrt._load_lazy_flow_patch(
        meta=meta,
        temporal_offset=1,
        clip_len=2,
        h_idx=0,
        w_idx=0,
        patch_h=4,
        patch_w=4,
        full_h=4,
        full_w=4,
        device=torch.device("cpu"),
    )

    assert loaded == [str(tmp_path / "000002"), str(tmp_path / "000003")]
    assert tuple(patch.shape) == (1, 4, 25, 4, 4)


def test_test_clip_passes_lazy_flow_patch_to_model(monkeypatch):
    captured = {}

    def _fake_lazy_flow_patch(**kwargs):
        return torch.ones(1, 2, 25, 4, 4)

    class _FakeModel(torch.nn.Module):
        def forward(self, x, flow_spike=None):
            captured["x_shape"] = tuple(x.shape)
            captured["flow_shape"] = tuple(flow_spike.shape)
            return torch.zeros(x.size(0), x.size(1), 3, x.size(3), x.size(4), device=x.device)

    monkeypatch.setattr(main_test_vrt, "_load_lazy_flow_patch", _fake_lazy_flow_patch)
    args = SimpleNamespace(
        scale=1,
        window_size=[2, 4, 4],
        tile=[2, 4, 4],
        tile_overlap=[0, 0, 0],
        nonblind_denoising=False,
        netG_cfg={"in_chans": 12, "out_chans": 3},
    )

    output = main_test_vrt.test_clip(
        torch.rand(1, 2, 12, 4, 4),
        _FakeModel(),
        args,
        flow_spike_meta={"frame_names": ["000001", "000002"], "subframes": 1},
        temporal_offset=0,
        full_h=4,
        full_w=4,
    )

    assert tuple(output.shape) == (1, 2, 3, 4, 4)
    assert captured["x_shape"] == (1, 2, 12, 4, 4)
    assert captured["flow_shape"] == (1, 2, 25, 4, 4)
