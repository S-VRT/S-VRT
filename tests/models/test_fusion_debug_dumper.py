from pathlib import Path

import torch

from models.fusion.debug import FusionDebugDumper


def _make_opt(image_root):
    return {
        "path": {"images": str(image_root)},
        "netG": {
            "fusion": {
                "placement": "early",
                "debug": {
                    "enable": True,
                    "save_images": True,
                    "subdir": "fusion_debug",
                    "max_frames": 2,
                },
            }
        },
    }


def test_fusion_debug_dumper_writes_center_frame_metrics(tmp_path):
    dumper = FusionDebugDumper(_make_opt(tmp_path))
    fusion = torch.zeros(1, 8, 3, 16, 16)
    gt = torch.ones(1, 2, 3, 16, 16) * 0.25
    fusion[:, 2::4, :, :, :] = gt
    dumper.capture_tensor(fusion)

    dumped = dumper.dump_last(
        current_step=9,
        folder="GOPR0001",
        lq_paths=[["/data/GOPR0001/000001.png", "/data/GOPR0001/000002.png"]],
        gt=gt,
        rank=0,
    )

    assert dumped is True
    metrics_path = tmp_path / "GOPR0001" / "fusion_debug" / "fusion_metrics_9.csv"
    text = metrics_path.read_text()
    assert "clip,frame,subframe,t,psnr,ssim" in text
    assert "000001,0,2,2,inf,1.000000" in text
    assert "000001,1,2,6,inf,1.000000" in text


def test_fusion_debug_hook_captures_fused_main_from_structured_output(tmp_path):
    dumper = FusionDebugDumper(_make_opt(tmp_path))
    dumper.enabled = True
    dumper.save_images = True
    dumper.arm()

    structured = {
        "fused_main": torch.rand(1, 2, 3, 8, 8),
        "backbone_view": torch.rand(1, 8, 3, 8, 8),
        "aux_view": torch.rand(1, 8, 3, 8, 8),
        "meta": {"frame_contract": "expanded", "spike_bins": 4},
    }

    dumper._capture_hook(module=None, inputs=(), output=structured)

    assert dumper._last_output is not None
    assert tuple(dumper._last_output.shape) == (1, 2, 3, 8, 8)


def test_fusion_debug_dumper_defaults_to_main_view(tmp_path):
    dumper = FusionDebugDumper(_make_opt(tmp_path))
    fusion_main = torch.rand(1, 2, 3, 8, 8)
    dumped = dumper.dump_tensor(
        fusion_main=fusion_main,
        fusion_exec=torch.rand(1, 8, 3, 8, 8),
        fusion_meta={"frame_contract": "expanded", "spike_bins": 4},
        current_step=3,
        folder="GOPR0001",
        gt=torch.rand(1, 2, 3, 8, 8),
        rank=0,
    )
    assert dumped is True


def test_fusion_debug_dumper_can_dump_expanded_execution_view_when_requested(tmp_path):
    dumper = FusionDebugDumper(_make_opt(tmp_path))
    fusion_exec = torch.rand(1, 8, 3, 8, 8)
    dumped = dumper.dump_tensor(
        fusion_main=torch.rand(1, 2, 3, 8, 8),
        fusion_exec=fusion_exec,
        fusion_meta={"frame_contract": "expanded", "spike_bins": 4},
        current_step=4,
        folder="GOPR0002",
        gt=torch.rand(1, 2, 3, 8, 8),
        rank=0,
        source_view="exec",
    )
    assert dumped is True
