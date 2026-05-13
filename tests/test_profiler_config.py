from pathlib import Path


def test_profiler_config_disabled_by_default():
    from utils.utils_profiler import TrainProfilerConfig

    cfg = TrainProfilerConfig.from_opt({}, experiment_dir=Path("experiments/task"), rank=0)

    assert cfg.enable is False
    assert cfg.should_profile_rank is False


def test_profiler_config_builds_trace_dir_for_rank_zero(tmp_path):
    from utils.utils_profiler import TrainProfilerConfig

    cfg = TrainProfilerConfig.from_opt(
        {
            "profiler": {
                "enable": True,
                "start_iter": 100,
                "wait": 1,
                "warmup": 1,
                "active": 2,
                "repeat": 1,
                "ranks": [0],
                "record_shapes": True,
                "with_stack": False,
                "profile_memory": True,
            }
        },
        experiment_dir=tmp_path,
        rank=0,
    )

    assert cfg.enable is True
    assert cfg.should_profile_rank is True
    assert cfg.start_iter == 100
    assert cfg.trace_dir == tmp_path / "profiles" / "rank0"


def test_profiler_config_skips_unselected_rank(tmp_path):
    from utils.utils_profiler import TrainProfilerConfig

    cfg = TrainProfilerConfig.from_opt(
        {"profiler": {"enable": True, "ranks": [0]}},
        experiment_dir=tmp_path,
        rank=1,
    )

    assert cfg.enable is True
    assert cfg.should_profile_rank is False
