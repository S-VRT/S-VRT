from pathlib import Path

import pytest

from utils.utils_two_run import (
    deep_merge_dict,
    dump_resolved_two_run_opts,
    resolve_two_run_phase_opts,
    validate_two_run_config,
)


def _base_opt(tmp_path: Path):
    return {
        "opt_path": "options/example.json",
        "path": {
            "task": str(tmp_path / "exp"),
            "options": str(tmp_path / "exp" / "options"),
            "pretrained_netG": "weights/base.pth",
            "pretrained_netE": None,
            "pretrained_optimizerG": None,
        },
        "datasets": {
            "train": {
                "gt_size": 128,
                "dataloader_batch_size": 8,
            }
        },
        "train": {
            "freeze_backbone": True,
            "G_optimizer_reuse": True,
            "G_optimizer_lr": 2e-4,
            "G_scheduler_type": "CosineAnnealingWarmRestarts",
            "G_scheduler_periods": 10000,
            "two_run": {
                "enable": True,
                "phase1": {
                    "total_iter": 4000,
                    "G_optimizer_lr": 4e-4,
                    "checkpoint_test": [4000],
                },
                "phase2": {
                    "total_iter": 6000,
                    "G_optimizer_lr": 2e-4,
                    "G_optimizer_reuse": False,
                    "checkpoint_test": [2000, 4000, 6000],
                },
            },
        },
    }


def test_deep_merge_dict_replaces_lists_and_merges_nested_dicts():
    merged = deep_merge_dict(
        {"train": {"a": 1, "lst": [1, 2], "nested": {"x": 1, "y": 2}}},
        {"train": {"lst": [3], "nested": {"y": 9}}},
    )
    assert merged["train"]["a"] == 1
    assert merged["train"]["lst"] == [3]
    assert merged["train"]["nested"] == {"x": 1, "y": 9}


def test_resolve_two_run_phase_opts_applies_phase_overrides(tmp_path):
    phase1_opt, phase2_opt = resolve_two_run_phase_opts(_base_opt(tmp_path))
    assert phase1_opt["train"]["total_iter"] == 4000
    assert phase1_opt["train"]["G_optimizer_lr"] == 4e-4
    assert phase2_opt["train"]["total_iter"] == 6000
    assert phase2_opt["train"]["checkpoint_test"] == [2000, 4000, 6000]


def test_validate_two_run_config_requires_phase_total_iter(tmp_path):
    opt = _base_opt(tmp_path)
    del opt["train"]["two_run"]["phase2"]["total_iter"]
    with pytest.raises(ValueError, match="phase2.total_iter"):
        validate_two_run_config(opt)


def test_validate_two_run_config_rejects_phase2_optimizer_reuse(tmp_path):
    opt = _base_opt(tmp_path)
    opt["train"]["two_run"]["phase2"]["G_optimizer_reuse"] = True
    with pytest.raises(ValueError, match="phase2.*G_optimizer_reuse"):
        validate_two_run_config(opt)


def test_dump_resolved_two_run_opts_writes_three_json_files(tmp_path):
    base = _base_opt(tmp_path)
    phase1_opt, phase2_opt = resolve_two_run_phase_opts(base)
    written = dump_resolved_two_run_opts(base, phase1_opt, phase2_opt)
    assert len(written) == 3
    assert written["base"].exists()
    assert written["phase1"].exists()
    assert written["phase2"].exists()
