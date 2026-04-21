import pytest

from main_train_vrt import resolve_phase_value, build_phase_train_dataset_opt


def test_resolve_phase_value_scalar_kept_for_both_phases():
    assert resolve_phase_value(4, True, "dataloader_batch_size") == 4
    assert resolve_phase_value(4, False, "dataloader_batch_size") == 4


def test_resolve_phase_value_array_phase1_phase2():
    assert resolve_phase_value([8, 4], True, "dataloader_batch_size") == 8
    assert resolve_phase_value([8, 4], False, "dataloader_batch_size") == 4


def test_resolve_phase_value_rejects_bad_array_length():
    with pytest.raises(ValueError, match="must be an int or a length-2 list/tuple"):
        resolve_phase_value([8, 4, 2], True, "dataloader_batch_size")


def test_resolve_phase_value_rejects_non_positive():
    with pytest.raises(ValueError, match="must be > 0"):
        resolve_phase_value([0, 4], True, "dataloader_batch_size")


def test_build_phase_train_dataset_opt_overrides_only_phase_keys():
    base = {
        "dataset_type": "TrainDatasetRGBSpike",
        "gt_size": [128, 96],
        "dataloader_batch_size": [8, 4],
        "dataloader_num_workers": 12,
        "dataloader_shuffle": True,
    }
    phase1 = build_phase_train_dataset_opt(base, is_phase1=True)
    phase2 = build_phase_train_dataset_opt(base, is_phase1=False)

    assert phase1["gt_size"] == 128
    assert phase1["dataloader_batch_size"] == 8
    assert phase2["gt_size"] == 96
    assert phase2["dataloader_batch_size"] == 4

    assert phase1["dataloader_num_workers"] == 12
    assert phase2["dataloader_shuffle"] is True
