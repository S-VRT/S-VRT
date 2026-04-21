from types import SimpleNamespace

import pytest

from utils import utils_option

from main_train_vrt import (
    build_phase_train_dataset_opt,
    build_train_loader_bundle,
    compute_is_phase1,
    record_data_wait,
    resolve_phase_value,
)


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


def test_build_phase_train_dataset_opt_disables_scflow_payload_for_phase1_only():
    base = {
        "dataset_type": "TrainDatasetRGBSpike",
        "gt_size": [128, 96],
        "dataloader_batch_size": [8, 4],
        "spike_flow": {
            "representation": "encoding25",
            "dt": 10,
            "root": "auto",
            "subframes": 4,
        },
    }

    phase1 = build_phase_train_dataset_opt(base, is_phase1=True)
    phase2 = build_phase_train_dataset_opt(base, is_phase1=False)

    assert phase1["spike_flow"]["representation"] == ""
    assert phase1["spike_flow"]["phase1_disabled"] is True
    assert phase2["spike_flow"]["representation"] == "encoding25"
    assert "phase1_disabled" not in phase2["spike_flow"]
    assert base["spike_flow"]["representation"] == "encoding25"


def test_resolve_phase_value_non_int_rejected():
    with pytest.raises(ValueError, match="resolved value must be int"):
        resolve_phase_value([8, 4.5], False, "dataloader_batch_size")


def test_build_train_loader_bundle_resolves_phase_values_for_single_process(monkeypatch):
    captured = {}
    dataset = SimpleNamespace(items=list(range(10)))

    def fake_define_dataset(dataset_opt):
        captured["dataset_opt"] = dataset_opt
        return dataset

    class FakeDataLoader:
        def __init__(self, train_set, **kwargs):
            captured["train_set"] = train_set
            captured["loader_kwargs"] = kwargs

    monkeypatch.setattr("main_train_vrt.define_Dataset", fake_define_dataset)
    monkeypatch.setattr("main_train_vrt.DataLoader", FakeDataLoader)

    opt = {"dist": False}
    train_dataset_opt = {
        "dataset_type": "TrainDatasetRGBSpike",
        "gt_size": [128, 96],
        "dataloader_batch_size": [8, 4],
        "dataloader_num_workers": 0,
        "dataloader_shuffle": True,
    }

    bundle = build_train_loader_bundle(opt, train_dataset_opt, is_phase1=False, seed=123, logger=None)

    assert bundle["dataset_opt"]["gt_size"] == 96
    assert bundle["dataset_opt"]["dataloader_batch_size"] == 4
    assert bundle["train_set"] is dataset
    assert bundle["train_sampler"] is None
    assert captured["dataset_opt"]["gt_size"] == 96
    assert captured["loader_kwargs"] == {
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 0,
        "drop_last": True,
        "pin_memory": True,
    }


def test_build_train_loader_bundle_installs_worker_init_when_workers_enabled(monkeypatch):
    captured = {}
    dataset = SimpleNamespace(items=list(range(10)))

    def fake_define_dataset(dataset_opt):
        captured["dataset_opt"] = dataset_opt
        return dataset

    class FakeDataLoader:
        def __init__(self, train_set, **kwargs):
            captured["loader_kwargs"] = kwargs

    monkeypatch.setattr("main_train_vrt.define_Dataset", fake_define_dataset)
    monkeypatch.setattr("main_train_vrt.DataLoader", FakeDataLoader)

    opt = {"dist": False}
    train_dataset_opt = {
        "dataset_type": "TrainDatasetRGBSpike",
        "gt_size": 128,
        "dataloader_batch_size": 8,
        "dataloader_num_workers": 2,
        "dataloader_shuffle": True,
    }

    build_train_loader_bundle(opt, train_dataset_opt, is_phase1=True, seed=123, logger=None)

    assert callable(captured["loader_kwargs"]["worker_init_fn"])


def test_record_data_wait_adds_wait_time_to_model_timer():
    timer = SimpleNamespace(current_timings={})
    model = SimpleNamespace(timer=timer)

    record_data_wait(model, 1.25)

    assert timer.current_timings["data_wait"] == 1.25


def test_compute_is_phase1_boundary():
    assert compute_is_phase1(0, 10) is True
    assert compute_is_phase1(9, 10) is True
    assert compute_is_phase1(10, 10) is False
    assert compute_is_phase1(11, 10) is False


def test_debug_config_has_phase1_fusion_loss_for_fast_path():
    opt = utils_option.parse("options/gopro_rgbspike_server_debug.json", is_train=True)
    train = opt["train"]

    assert train["fix_iter"] > 0
    assert (
        train.get("phase1_fusion_aux_loss_weight", 0.0)
        + train.get("fusion_passthrough_loss_weight", 0.0)
    ) > 0.0
