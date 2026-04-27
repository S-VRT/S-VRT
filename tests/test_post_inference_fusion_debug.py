from types import SimpleNamespace

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
