from types import SimpleNamespace

import torch

from scripts.analysis import fusion_attribution
from scripts.analysis.fusion_attr.io import AnalysisSample


def test_sample_matches_rgbspike_key_batch():
    batch = {"key": ["GOPR0396_11_00/000050"]}

    assert fusion_attribution._sample_matches(batch, "GOPR0396_11_00", "000050")


def test_sample_does_not_match_wrong_rgbspike_key_frame():
    batch = {"key": ["GOPR0396_11_00/000001"]}

    assert not fusion_attribution._sample_matches(batch, "GOPR0396_11_00", "000050")


def test_shard_samples_for_rank_keeps_every_fourth_sample():
    samples = [SimpleNamespace(clip=f"clip_{idx}") for idx in range(7)]

    shard = fusion_attribution._shard_samples_for_rank(samples, rank=2, world_size=4)

    assert [sample.clip for sample in shard] == ["clip_2", "clip_6"]


def test_resolve_device_uses_local_rank_for_torchrun_cuda(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert fusion_attribution._resolve_device("cuda:0") == "cuda:3"


def test_prepare_analysis_batch_crops_temporal_and_spatial_patch():
    batch = {
        "L": torch.zeros(1, 100, 7, 720, 1280),
        "H": torch.zeros(1, 100, 3, 720, 1280),
        "lq_path": [[f"/clip/{idx:06d}.png"] for idx in range(1, 101)],
    }
    sample = AnalysisSample(
        clip="GOPR0396_11_00",
        frame="000050",
        frame_index=50,
        mask_type="box",
        xyxy=(240, 120, 400, 240),
        mask_label="center",
        reason="test",
    )

    cropped_batch, cropped_sample = fusion_attribution._prepare_analysis_batch(
        batch,
        sample,
        num_frames=12,
        crop_size=256,
    )

    assert cropped_batch["L"].shape == (1, 12, 7, 256, 256)
    assert cropped_batch["H"].shape == (1, 12, 3, 256, 256)
    assert cropped_sample.xyxy == (48, 68, 208, 188)
    assert cropped_batch["_analysis_temporal_start"] == 43
    assert cropped_batch["_analysis_crop_left"] == 192
    assert cropped_batch["_analysis_crop_top"] == 52


def test_inject_lora_for_checkpoint_if_needed(monkeypatch):
    calls = []
    opt = {"train": {"use_lora": True}, "path": {}}
    model = SimpleNamespace(netG=torch.nn.Linear(2, 2))
    model.get_bare_model = lambda net: net
    model._inject_lora_adapters = lambda train_opt, bare: calls.append((train_opt, bare)) or True
    monkeypatch.setattr(fusion_attribution.ModelPlain, "_checkpoint_contains_lora", lambda path: True)

    fusion_attribution._inject_lora_for_checkpoint_if_needed(model, opt, "checkpoint.pth")

    assert calls == [(opt["train"], model.netG)]


def test_build_folder_index_maps_dataset_folders():
    dataset = SimpleNamespace(folders=["clip_a", "clip_b"])

    assert fusion_attribution._build_folder_index(dataset) == {"clip_a": 0, "clip_b": 1}


def test_spatial_tiles_cover_full_frame_with_tail_tiles():
    tiles = fusion_attribution._spatial_tiles(height=10, width=12, tile_size=6, stride=5)

    assert tiles == [(0, 0, 6, 6), (0, 5, 6, 11), (0, 6, 6, 12), (4, 0, 10, 6), (4, 5, 10, 11), (4, 6, 10, 12)]


def test_accumulate_tile_averages_overlap():
    accum = torch.zeros(1, 1, 4, 4)
    weight = torch.zeros_like(accum)
    fusion_attribution._accumulate_tile(accum, weight, torch.ones(1, 1, 3, 3), top=0, left=0)
    fusion_attribution._accumulate_tile(accum, weight, torch.ones(1, 1, 3, 3) * 3, top=1, left=1)

    stitched = fusion_attribution._normalize_accumulated(accum, weight)

    assert stitched[0, 0, 1, 1].item() == 2.0
    assert stitched[0, 0, 0, 0].item() == 1.0
    assert stitched[0, 0, 3, 3].item() == 3.0
