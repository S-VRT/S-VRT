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
