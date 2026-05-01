from types import SimpleNamespace

import pytest

from utils import utils_option

from main_train_vrt import (
    build_phase_train_dataset_opt,
    build_train_sampler,
    build_two_run_phase_model_train_opt,
    build_train_loader_bundle,
    compute_is_phase1,
    resolve_phase_value,
)


class SizedDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


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
    dataset = SizedDataset(10)

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


def test_build_train_loader_bundle_passes_repeat_factor_to_distributed_sampler(monkeypatch):
    captured = {}
    dataset = SizedDataset(10)

    def fake_define_dataset(dataset_opt):
        return dataset

    class FakeDataLoader:
        def __init__(self, train_set, **kwargs):
            captured["loader_kwargs"] = kwargs

    monkeypatch.setattr("main_train_vrt.define_Dataset", fake_define_dataset)
    monkeypatch.setattr("main_train_vrt.DataLoader", FakeDataLoader)

    opt = {"dist": True, "num_gpu": 4, "rank": 0}
    train_dataset_opt = {
        "dataset_type": "TrainDatasetRGBSpike",
        "gt_size": 128,
        "dataloader_batch_size": 16,
        "dataloader_num_workers": 0,
        "dataloader_shuffle": True,
        "dataloader_epoch_repeat": 8,
    }

    bundle = build_train_loader_bundle(opt, train_dataset_opt, is_phase1=True, seed=123, logger=None)

    assert bundle["train_sampler"].dataset is dataset
    assert bundle["train_sampler"].epoch_repeat == 8
    assert len(bundle["train_sampler"]) == 16
    assert captured["loader_kwargs"]["sampler"] is bundle["train_sampler"]
    assert captured["loader_kwargs"]["shuffle"] is False


def test_build_train_loader_bundle_installs_worker_init_when_workers_enabled(monkeypatch):
    captured = {}
    dataset = SizedDataset(10)

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


def test_repeat_distributed_sampler_repeats_indices_and_extends_epoch():
    dataset = SizedDataset(5)
    sampler = build_train_sampler(
        dataset,
        shuffle=False,
        seed=123,
        epoch_repeat=3,
        num_replicas=1,
        rank=0,
    )

    assert len(sampler) == 15
    assert list(iter(sampler)) == list(range(5)) * 3


def test_compute_is_phase1_boundary():
    assert compute_is_phase1(0, 10) is True
    assert compute_is_phase1(9, 10) is True
    assert compute_is_phase1(10, 10) is False
    assert compute_is_phase1(11, 10) is False


def test_two_run_phase1_model_fix_iter_covers_final_phase_step():
    phase_train_opt = {
        "total_iter": 4000,
        "fix_iter": 4000,
        "G_optimizer_lr": 2e-4,
    }

    model_train_opt = build_two_run_phase_model_train_opt(phase_train_opt, phase_name="phase1")

    assert model_train_opt["fix_iter"] == 4001
    assert compute_is_phase1(4000, model_train_opt["fix_iter"]) is True
    assert phase_train_opt["fix_iter"] == 4000


def test_two_run_phase2_model_fix_iter_is_not_shifted():
    phase_train_opt = {
        "total_iter": 6000,
        "fix_iter": 1,
        "G_optimizer_lr": 2e-4,
    }

    model_train_opt = build_two_run_phase_model_train_opt(phase_train_opt, phase_name="phase2")

    assert model_train_opt["fix_iter"] == 1


def test_debug_config_has_phase1_fusion_loss_for_fast_path():
    opt = utils_option.parse("options/gopro_rgbspike_server_debug.json", is_train=True)
    train = opt["train"]

    assert train["fix_iter"] > 0
    assert (
        train.get("phase1_fusion_aux_loss_weight", 0.0)
        + train.get("fusion_passthrough_loss_weight", 0.0)
    ) > 0.0


def test_snapshot_config_parses_two_run_block():
    opt = utils_option.parse("options/gopro_rgbspike_server_pase_residual_snapshot.json", is_train=True)
    two_run = opt["train"]["two_run"]

    assert two_run["enable"] is True
    assert two_run["phase1"]["total_iter"] == 4000
    assert two_run["phase2"]["total_iter"] == 6000
