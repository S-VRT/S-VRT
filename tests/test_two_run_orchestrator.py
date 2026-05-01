from types import SimpleNamespace

import pytest

from main_train_vrt import (
    compute_global_step,
    should_finish_phase,
)


def test_compute_global_step_adds_phase_step_to_offset():
    assert compute_global_step(global_step_offset=0, phase_step=1) == 1
    assert compute_global_step(global_step_offset=4000, phase_step=1) == 4001
    assert compute_global_step(global_step_offset=4000, phase_step=6000) == 10000


def test_should_finish_phase_uses_greater_equal_boundary():
    assert should_finish_phase(current_phase_step=3, total_iter=4) is False
    assert should_finish_phase(current_phase_step=4, total_iter=4) is True
    assert should_finish_phase(current_phase_step=5, total_iter=4) is True


def test_run_phase_keeps_tracking_logger_open_until_explicit_close(monkeypatch):
    from main_train_vrt import run_phase

    calls = {"close": 0, "log_scalars": []}

    class _Logger:
        def log_scalars(self, step, scalar_dict, tag_prefix=""):
            calls["log_scalars"].append((step, scalar_dict, tag_prefix))

        def close(self):
            calls["close"] += 1

    class _Model:
        def __init__(self):
            self.timer = SimpleNamespace(current_timings={}, clear=lambda: None)

    monkeypatch.setattr("main_train_vrt.build_phase_runtime", lambda *args, **kwargs: {
        "model": _Model(),
        "train_loader": iter([{"L": 1}, {"L": 2}]),
        "train_sampler": None,
        "train_set": [1, 2],
        "active_train_dataset_opt": {"dataloader_batch_size": 1, "gt_size": 64},
    })

    monkeypatch.setattr("main_train_vrt.execute_training_iteration", lambda **kwargs: {"G_loss": 0.1})
    monkeypatch.setattr("main_train_vrt.finalize_phase", lambda **kwargs: {"final_checkpoint": "models/2_G.pth"})

    result = run_phase(
        phase_opt={"train": {"total_iter": 2, "checkpoint_print": 1}, "rank": 0},
        shared_runtime={"tb_logger": _Logger(), "logger": None, "seed": 123},
        phase_name="phase1",
        global_step_offset=0,
        resume_state={"phase_step": 0},
    )

    assert result["last_phase_step"] == 2
    assert result["last_global_step"] == 2
    assert calls["close"] == 0
    assert calls["log_scalars"][0][0] == 1
    assert calls["log_scalars"][1][0] == 2


def test_run_phase_logs_training_status_at_checkpoint_print(monkeypatch):
    from main_train_vrt import run_phase

    messages = []

    class _Logger:
        def info(self, message, *args):
            if args:
                message = message % args
            messages.append(message)

    class _Timer:
        def __init__(self):
            self.current_timings = {}

        def timer(self, name):
            timer = self

            class _Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    timer.current_timings[name] = 1.25
                    return False

            return _Context()

        def get_current_timings(self):
            return self.current_timings.copy()

    class _Model:
        def __init__(self):
            self.timer = _Timer()

        def current_learning_rate(self):
            return 2e-4

    monkeypatch.setattr("main_train_vrt.build_phase_runtime", lambda *args, **kwargs: {
        "model": _Model(),
        "train_loader": iter([{"L": 1}, {"L": 2}]),
        "train_sampler": None,
        "train_set": [1, 2],
        "active_train_dataset_opt": {"dataloader_batch_size": 1, "gt_size": 64},
    })
    monkeypatch.setattr(
        "main_train_vrt.execute_training_iteration",
        lambda **kwargs: {
            "G_loss": 0.1,
            **{
                f"time_{key}": value
                for key, value in kwargs["model"].timer.get_current_timings().items()
            },
            "time_forward": 0.5,
            "phase_step": float(kwargs["phase_step"]),
            "global_step": float(kwargs["global_step"]),
        },
    )
    monkeypatch.setattr("main_train_vrt.finalize_phase", lambda **kwargs: {})

    run_phase(
        phase_opt={"train": {"total_iter": 2, "checkpoint_print": 2}, "rank": 0},
        shared_runtime={"tb_logger": None, "logger": _Logger(), "seed": 123},
        phase_name="phase1",
        global_step_offset=10,
        resume_state={"phase_step": 0},
    )

    assert any(
        "<phase:phase1, epoch:" in message
        and "phase_iter:       2" in message
        and "global_iter:      12" in message
        and "batch_wait: 1.2500s" in message
        and "forward: 0.5000s" in message
        for message in messages
    )


def test_run_phase_logs_outer_iteration_and_state_persist_timing(monkeypatch):
    from main_train_vrt import run_phase

    messages = []

    class _Logger:
        def info(self, message, *args):
            if args:
                message = message % args
            messages.append(message)

    class _Model:
        def __init__(self):
            self.timer = SimpleNamespace(current_timings={}, clear=lambda: None)

        def current_learning_rate(self):
            return 2e-4

    times = iter([
        0.0, 0.1, 0.2, 0.3,
        1.0, 1.1, 1.2, 1.3,
    ])

    monkeypatch.setattr("main_train_vrt.time.perf_counter", lambda: next(times))
    monkeypatch.setattr("main_train_vrt.build_phase_runtime", lambda *args, **kwargs: {
        "model": _Model(),
        "train_loader": iter([{"L": 1}, {"L": 2}]),
        "train_sampler": None,
        "train_set": [1, 2],
        "active_train_dataset_opt": {"dataloader_batch_size": 1, "gt_size": 64},
    })
    monkeypatch.setattr(
        "main_train_vrt.execute_training_iteration",
        lambda **kwargs: {
            "G_loss": 0.1,
            "phase_step": float(kwargs["phase_step"]),
            "global_step": float(kwargs["global_step"]),
        },
    )
    monkeypatch.setattr("main_train_vrt.save_two_run_state", lambda _path, _state: None)
    monkeypatch.setattr("main_train_vrt.finalize_phase", lambda **kwargs: {})

    run_phase(
        phase_opt={
            "train": {
                "total_iter": 2,
                "checkpoint_print": 2,
                "timing": {"enable": True},
            },
            "rank": 0,
        },
        shared_runtime={
            "tb_logger": None,
            "logger": _Logger(),
            "seed": 123,
            "two_run_state": {},
            "two_run_state_path": "state.json",
        },
        phase_name="phase1",
        global_step_offset=10,
        resume_state={"phase_step": 0},
    )

    assert any(
        "iter_total: 0.3000s" in message
        and "state_persist: 0.1000s" in message
        for message in messages
    )


def test_run_phase_suppresses_outer_timing_when_train_timing_disabled(monkeypatch):
    from main_train_vrt import run_phase

    messages = []

    class _Logger:
        def info(self, message, *args):
            if args:
                message = message % args
            messages.append(message)

    class _Model:
        def __init__(self):
            self.timer = SimpleNamespace(current_timings={}, clear=lambda: None)

        def current_learning_rate(self):
            return 2e-4

    times = iter([
        0.0, 0.1, 0.2, 0.3,
        1.0, 1.1, 1.2, 1.3,
    ])

    monkeypatch.setattr("main_train_vrt.time.perf_counter", lambda: next(times))
    monkeypatch.setattr("main_train_vrt.build_phase_runtime", lambda *args, **kwargs: {
        "model": _Model(),
        "train_loader": iter([{"L": 1}, {"L": 2}]),
        "train_sampler": None,
        "train_set": [1, 2],
        "active_train_dataset_opt": {"dataloader_batch_size": 1, "gt_size": 64},
    })
    monkeypatch.setattr(
        "main_train_vrt.execute_training_iteration",
        lambda **kwargs: {
            "G_loss": 0.1,
            "phase_step": float(kwargs["phase_step"]),
            "global_step": float(kwargs["global_step"]),
        },
    )
    monkeypatch.setattr("main_train_vrt.save_two_run_state", lambda _path, _state: None)
    monkeypatch.setattr("main_train_vrt.finalize_phase", lambda **kwargs: {})

    run_phase(
        phase_opt={
            "train": {
                "total_iter": 2,
                "checkpoint_print": 2,
                "timing": {"enable": False},
            },
            "rank": 0,
        },
        shared_runtime={
            "tb_logger": None,
            "logger": _Logger(),
            "seed": 123,
            "two_run_state": {},
            "two_run_state_path": "state.json",
        },
        phase_name="phase1",
        global_step_offset=10,
        resume_state={"phase_step": 0},
    )

    assert messages
    assert all("iter_total:" not in message for message in messages)
    assert all("state_persist:" not in message for message in messages)
    assert all("window_iter_total_" not in message for message in messages)
    assert all("window_state_persist_" not in message for message in messages)


def test_run_phase_logs_checkpoint_window_timing_stats(monkeypatch):
    from main_train_vrt import run_phase

    messages = []

    class _Logger:
        def info(self, message, *args):
            if args:
                message = message % args
            messages.append(message)

    class _Model:
        def __init__(self):
            self.timer = SimpleNamespace(current_timings={}, clear=lambda: None)

        def current_learning_rate(self):
            return 2e-4

    times = iter([
        0.0, 0.1, 0.2, 0.3,
        1.0, 2.0, 2.2, 2.3,
    ])

    monkeypatch.setattr("main_train_vrt.time.perf_counter", lambda: next(times))
    monkeypatch.setattr("main_train_vrt.build_phase_runtime", lambda *args, **kwargs: {
        "model": _Model(),
        "train_loader": iter([{"L": 1}, {"L": 2}]),
        "train_sampler": None,
        "train_set": [1, 2],
        "active_train_dataset_opt": {"dataloader_batch_size": 1, "gt_size": 64},
    })
    monkeypatch.setattr(
        "main_train_vrt.execute_training_iteration",
        lambda **kwargs: {
            "G_loss": 0.1,
            "phase_step": float(kwargs["phase_step"]),
            "global_step": float(kwargs["global_step"]),
        },
    )
    monkeypatch.setattr("main_train_vrt.save_two_run_state", lambda _path, _state: None)
    monkeypatch.setattr("main_train_vrt.finalize_phase", lambda **kwargs: {})

    run_phase(
        phase_opt={
            "train": {
                "total_iter": 2,
                "checkpoint_print": 2,
                "timing": {"enable": True},
            },
            "rank": 0,
        },
        shared_runtime={
            "tb_logger": None,
            "logger": _Logger(),
            "seed": 123,
            "two_run_state": {},
            "two_run_state_path": "state.json",
        },
        phase_name="phase1",
        global_step_offset=10,
        resume_state={"phase_step": 0},
    )

    assert any(
        "window_steps: 2.000e+00" in message
        and "window_iter_total_mean: 0.8000s" in message
        and "window_iter_total_max: 1.3000s" in message
        and "window_state_persist_mean: 0.1500s" in message
        and "window_state_persist_max: 0.2000s" in message
        for message in messages
    )


def test_run_phase_rolls_over_epochs_when_total_iter_exceeds_loader_len(monkeypatch):
    from main_train_vrt import run_phase

    calls = {"steps": []}

    class _Loader:
        def __iter__(self):
            return iter([{"L": "a"}, {"L": "b"}])

    class _Model:
        def __init__(self):
            self.timer = SimpleNamespace(current_timings={}, clear=lambda: None)

    monkeypatch.setattr("main_train_vrt.build_phase_runtime", lambda *args, **kwargs: {
        "model": _Model(),
        "train_loader": _Loader(),
        "train_sampler": None,
        "train_set": [1, 2],
        "active_train_dataset_opt": {"dataloader_batch_size": 1, "gt_size": 64},
    })

    def _execute_training_iteration(**kwargs):
        calls["steps"].append((kwargs["phase_step"], kwargs["global_step"], kwargs["train_data"]["L"]))
        return {"G_loss": 0.1}

    monkeypatch.setattr("main_train_vrt.execute_training_iteration", _execute_training_iteration)
    monkeypatch.setattr("main_train_vrt.finalize_phase", lambda **kwargs: {"final_checkpoint": "models/5_G.pth"})

    result = run_phase(
        phase_opt={"train": {"total_iter": 5, "checkpoint_print": 1}, "rank": 0},
        shared_runtime={"tb_logger": None, "logger": None, "seed": 123},
        phase_name="phase1",
        global_step_offset=10,
        resume_state={"phase_step": 0},
    )

    assert result["last_phase_step"] == 5
    assert result["last_global_step"] == 15
    assert calls["steps"] == [
        (1, 11, "a"),
        (2, 12, "b"),
        (3, 13, "a"),
        (4, 14, "b"),
        (5, 15, "a"),
    ]


def test_run_phase_advances_sampler_epoch_on_epoch_boundaries(monkeypatch):
    from main_train_vrt import run_phase

    class _Sampler:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    class _Loader:
        def __iter__(self):
            return iter([{"L": 1}, {"L": 2}])

    class _Model:
        def __init__(self):
            self.timer = SimpleNamespace(current_timings={}, clear=lambda: None)

    sampler = _Sampler()

    monkeypatch.setattr("main_train_vrt.build_phase_runtime", lambda *args, **kwargs: {
        "model": _Model(),
        "train_loader": _Loader(),
        "train_sampler": sampler,
        "train_set": [1, 2],
        "active_train_dataset_opt": {"dataloader_batch_size": 1, "gt_size": 64},
    })

    monkeypatch.setattr("main_train_vrt.execute_training_iteration", lambda **kwargs: {"G_loss": 0.1})
    monkeypatch.setattr("main_train_vrt.finalize_phase", lambda **kwargs: {})

    result = run_phase(
        phase_opt={"train": {"total_iter": 5, "checkpoint_print": 1}, "rank": 0},
        shared_runtime={"tb_logger": None, "logger": None, "seed": 123},
        phase_name="phase1",
        global_step_offset=0,
        resume_state={"phase_step": 0},
    )

    assert result["last_phase_step"] == 5
    assert sampler.epochs == [0, 1, 2]


def test_run_phase_runs_validation_at_phase_checkpoint_steps(monkeypatch):
    from main_train_vrt import run_phase

    calls = {"validation": []}

    class _Model:
        def __init__(self):
            self.timer = SimpleNamespace(current_timings={}, clear=lambda: None)

    monkeypatch.setattr("main_train_vrt.build_phase_runtime", lambda *args, **kwargs: {
        "model": _Model(),
        "train_loader": iter([{"L": 1}, {"L": 2}, {"L": 3}]),
        "train_sampler": None,
        "test_loader": ["val_batch"],
        "train_set": [1, 2, 3],
        "active_train_dataset_opt": {"dataloader_batch_size": 1, "gt_size": 64},
    })
    monkeypatch.setattr("main_train_vrt.execute_training_iteration", lambda **kwargs: {"G_loss": 0.1})
    monkeypatch.setattr("main_train_vrt.finalize_phase", lambda **kwargs: {})

    def _run_validation_checkpoint(**kwargs):
        calls["validation"].append((kwargs["phase_step"], kwargs["global_step"]))

    monkeypatch.setattr("main_train_vrt.run_validation_checkpoint", _run_validation_checkpoint)

    run_phase(
        phase_opt={
            "train": {"total_iter": 3, "checkpoint_print": 1, "checkpoint_test": [2]},
            "rank": 0,
        },
        shared_runtime={"tb_logger": None, "logger": None, "seed": 123},
        phase_name="phase2",
        global_step_offset=4,
        resume_state={"phase_step": 0},
    )

    assert calls["validation"] == [(2, 6)]


def test_run_experiment_executes_phase1_then_phase2_with_continuous_offset(monkeypatch, tmp_path):
    from main_train_vrt import run_experiment

    phase_calls = []
    saved_states = []

    base_opt = {
        "path": {"task": str(tmp_path / "exp"), "options": str(tmp_path / "exp" / "options")},
        "train": {"two_run": {"enable": True}},
        "rank": 0,
        "datasets": {"train": {}, "test": {}},
    }
    phase1_opt = {"train": {"total_iter": 4}, "datasets": {"train": {}}, "rank": 0}
    phase2_opt = {"train": {"total_iter": 6}, "datasets": {"train": {}}, "rank": 0}

    monkeypatch.setattr("main_train_vrt.resolve_two_run_phase_opts", lambda _opt: (phase1_opt, phase2_opt))
    monkeypatch.setattr("main_train_vrt.dump_resolved_two_run_opts", lambda *args, **kwargs: {})
    monkeypatch.setattr("main_train_vrt.load_two_run_state", lambda _path: None)
    monkeypatch.setattr("main_train_vrt.two_run_state_path", lambda _opt: tmp_path / "two_run_state.json")
    monkeypatch.setattr("main_train_vrt.build_initial_two_run_state", lambda **kwargs: {
        "phase1_total_iter": 4,
        "phase2_total_iter": 6,
        "phase1_completed": False,
        "phase2_started": False,
        "global_step_offset": 0,
        "last_successful_phase_step": 0,
    })
    monkeypatch.setattr("main_train_vrt.resolve_resume_phase", lambda _state: "phase1_fresh")
    monkeypatch.setattr("main_train_vrt.save_two_run_state", lambda _path, state: saved_states.append(dict(state)))

    def fake_run_phase(phase_opt, shared_runtime, phase_name, global_step_offset, resume_state):
        phase_calls.append((phase_name, global_step_offset, resume_state["phase_step"]))
        if phase_name == "phase1":
            return {
                "last_phase_step": 4,
                "last_global_step": 4,
                "final_checkpoint_G": "models/4000_G.pth",
                "final_checkpoint_E": None,
                "runtime": {"model": object()},
            }
        return {
            "last_phase_step": 6,
            "last_global_step": 10,
            "final_checkpoint_G": "models/10000_G.pth",
            "final_checkpoint_E": None,
            "runtime": {"model": object()},
        }

    monkeypatch.setattr("main_train_vrt.run_phase", fake_run_phase)
    monkeypatch.setattr("main_train_vrt.build_shared_runtime", lambda _opt, _logger, _tb_logger, seed: {
        "logger": _logger,
        "tb_logger": _tb_logger,
        "seed": seed,
    })
    monkeypatch.setattr("main_train_vrt.mark_phase1_completed", lambda state, **kwargs: state.update({
        "phase1_completed": True,
        "phase1_final_G": kwargs["phase1_final_g"],
        "phase1_final_E": kwargs["phase1_final_e"],
        "global_step_offset": state["phase1_total_iter"],
    }))
    monkeypatch.setattr("main_train_vrt.mark_phase2_started", lambda state: state.update({"phase2_started": True}))
    monkeypatch.setattr("main_train_vrt.close_shared_runtime", lambda _runtime: None)

    result = run_experiment(base_opt, logger=None, tb_logger=None, seed=123)

    assert phase_calls == [("phase1", 0, 0), ("phase2", 4, 0)]
    assert result["last_global_step"] == 10
    assert saved_states[-1]["phase2_started"] is True


def test_prepare_phase2_opt_forces_boundary_checkpoint_and_disables_optimizer_reuse():
    from main_train_vrt import prepare_phase2_opt

    phase2_opt = {
        "path": {
            "pretrained_netG": "weights/old.pth",
            "pretrained_netE": None,
            "pretrained_optimizerG": "models/old_optimizer.pth",
        },
        "train": {"G_optimizer_reuse": True},
    }

    updated = prepare_phase2_opt(
        phase2_opt,
        phase1_final_g="models/4000_G.pth",
        phase1_final_e="models/4000_E.pth",
    )

    assert updated["path"]["pretrained_netG"] == "models/4000_G.pth"
    assert updated["path"]["pretrained_netE"] == "models/4000_E.pth"
    assert updated["path"]["pretrained_optimizerG"] is None
    assert updated["train"]["G_optimizer_reuse"] is False


def test_prepare_phase2_opt_rejects_positional_phase1_args():
    from main_train_vrt import prepare_phase2_opt

    phase2_opt = {
        "path": {"pretrained_netG": None, "pretrained_netE": None, "pretrained_optimizerG": None},
        "train": {"G_optimizer_reuse": False},
    }
    with pytest.raises(TypeError):
        prepare_phase2_opt(phase2_opt, "models/4000_G.pth", "models/4000_E.pth")


def test_finalize_phase_returns_checkpoint_paths_without_e_when_no_e_decay(tmp_path, monkeypatch):
    from main_train_vrt import finalize_phase

    saved = []

    class _Model:
        def save(self, step):
            saved.append(("save", step))

    phase_opt = {
        "rank": 0,
        "path": {"models": str(tmp_path / "models")},
        "train": {"E_decay": 0, "use_lora": False},
    }

    result = finalize_phase(
        model=_Model(),
        phase_opt=phase_opt,
        phase_name="phase1",
        last_phase_step=4,
        last_global_step=4,
        shared_runtime={},
    )

    assert saved == [("save", 4)]
    assert result["final_checkpoint_G"].endswith("4_G.pth")
    assert result["final_checkpoint_E"] is None


def test_finalize_phase_returns_e_checkpoint_when_e_decay_set(tmp_path):
    from main_train_vrt import finalize_phase

    class _Model:
        def save(self, step):
            pass

    phase_opt = {
        "rank": 0,
        "path": {"models": str(tmp_path / "models")},
        "train": {"E_decay": 0.999, "use_lora": False},
    }

    result = finalize_phase(
        model=_Model(),
        phase_opt=phase_opt,
        phase_name="phase1",
        last_phase_step=4,
        last_global_step=4,
        shared_runtime={},
    )

    assert result["final_checkpoint_G"].endswith("4_G.pth")
    assert result["final_checkpoint_E"].endswith("4_E.pth")


def test_finalize_phase_calls_save_merged_when_use_lora(tmp_path):
    from main_train_vrt import finalize_phase

    calls = []

    class _Model:
        def save(self, step):
            calls.append(("save", step))

        def save_merged(self, step):
            calls.append(("save_merged", step))

    phase_opt = {
        "rank": 0,
        "path": {"models": str(tmp_path / "models")},
        "train": {"E_decay": 0, "use_lora": True},
    }

    finalize_phase(
        model=_Model(),
        phase_opt=phase_opt,
        phase_name="phase2",
        last_phase_step=10,
        last_global_step=10,
        shared_runtime={},
    )

    assert ("save", 10) in calls
    assert ("save_merged", 10) in calls


def test_finalize_phase_returns_global_step_checkpoint_labels(tmp_path):
    from main_train_vrt import finalize_phase

    calls = []

    class _Model:
        def save(self, step):
            calls.append(("save", step))

        def save_merged(self, step):
            calls.append(("save_merged", step))

    phase_opt = {
        "rank": 0,
        "path": {"models": str(tmp_path / "models")},
        "train": {"E_decay": 0, "use_lora": True},
    }

    result = finalize_phase(
        model=_Model(),
        phase_opt=phase_opt,
        phase_name="phase2",
        last_phase_step=6,
        last_global_step=10,
        shared_runtime={},
    )

    assert ("save", 10) in calls
    assert ("save_merged", 10) in calls
    assert result["final_checkpoint_G"].endswith("10_G.pth")


def test_run_phase_persists_state_after_each_iteration(monkeypatch, tmp_path):
    from main_train_vrt import run_phase

    state = {
        "last_successful_phase_step": 0,
        "last_successful_global_step": 0,
    }
    state_path = tmp_path / "two_run_state.json"
    saved_snapshots = []

    class _Model:
        def __init__(self):
            self.timer = SimpleNamespace(current_timings={}, clear=lambda: None)

    monkeypatch.setattr("main_train_vrt.build_phase_runtime", lambda *args, **kwargs: {
        "model": _Model(),
        "train_loader": iter([{"L": 1}, {"L": 2}]),
        "train_sampler": None,
        "train_set": [1, 2],
        "active_train_dataset_opt": {"dataloader_batch_size": 1, "gt_size": 64},
    })
    monkeypatch.setattr("main_train_vrt.execute_training_iteration", lambda **kwargs: {"G_loss": 0.1})
    monkeypatch.setattr("main_train_vrt.finalize_phase", lambda **kwargs: {
        "final_checkpoint_G": None, "final_checkpoint_E": None,
    })

    original_save = __import__("utils.utils_two_run", fromlist=["save_two_run_state"]).save_two_run_state

    def capturing_save(path, s):
        saved_snapshots.append((int(s["last_successful_phase_step"]), int(s["last_successful_global_step"])))
        original_save(path, s)

    monkeypatch.setattr("main_train_vrt.save_two_run_state", capturing_save)

    run_phase(
        phase_opt={"train": {"total_iter": 2, "checkpoint_print": 1}, "rank": 0},
        shared_runtime={
            "tb_logger": None,
            "logger": None,
            "seed": 42,
            "two_run_state": state,
            "two_run_state_path": state_path,
        },
        phase_name="phase1",
        global_step_offset=0,
        resume_state={"phase_step": 0},
    )

    assert saved_snapshots == [(1, 1), (2, 2)]
    assert state["last_successful_phase_step"] == 2
    assert state["last_successful_global_step"] == 2


def test_resume_phase2_uses_zero_phase_step_when_phase2_not_started(monkeypatch, tmp_path):
    from main_train_vrt import run_experiment

    phase_calls = []

    base_opt = {
        "path": {"task": str(tmp_path / "exp"), "options": str(tmp_path / "exp" / "options")},
        "train": {"two_run": {"enable": True}},
        "rank": 0,
        "datasets": {"train": {}, "test": {}},
    }
    phase1_opt = {"train": {"total_iter": 4}, "datasets": {"train": {}}, "rank": 0}
    phase2_opt = {"train": {"total_iter": 6}, "datasets": {"train": {}}, "rank": 0}

    # State: phase1 completed, phase2 not started
    initial_state = {
        "phase1_total_iter": 4,
        "phase2_total_iter": 6,
        "phase1_completed": True,
        "phase1_final_G": "models/4_G.pth",
        "phase1_final_E": None,
        "phase2_started": False,
        "global_step_offset": 4,
        "last_successful_phase_step": 0,
    }

    monkeypatch.setattr("main_train_vrt.resolve_two_run_phase_opts", lambda _opt: (phase1_opt, phase2_opt))
    monkeypatch.setattr("main_train_vrt.dump_resolved_two_run_opts", lambda *args, **kwargs: {})
    monkeypatch.setattr("main_train_vrt.load_two_run_state", lambda _path: dict(initial_state))
    monkeypatch.setattr("main_train_vrt.two_run_state_path", lambda _opt: tmp_path / "two_run_state.json")
    monkeypatch.setattr("main_train_vrt.save_two_run_state", lambda _path, _state: None)
    monkeypatch.setattr("main_train_vrt.mark_phase2_started", lambda state: state.update({"phase2_started": True}))
    monkeypatch.setattr("main_train_vrt.close_shared_runtime", lambda _runtime: None)

    def fake_run_phase(phase_opt, shared_runtime, phase_name, global_step_offset, resume_state):
        phase_calls.append({
            "phase_name": phase_name,
            "global_step_offset": global_step_offset,
            "phase_step": resume_state["phase_step"],
        })
        return {
            "last_phase_step": 6,
            "last_global_step": 10,
            "final_checkpoint_G": "models/10_G.pth",
            "final_checkpoint_E": None,
            "runtime": {"model": object()},
        }

    monkeypatch.setattr("main_train_vrt.run_phase", fake_run_phase)
    monkeypatch.setattr("main_train_vrt.build_shared_runtime", lambda _opt, _logger, _tb_logger, seed: {
        "logger": _logger,
        "tb_logger": _tb_logger,
        "seed": seed,
    })

    run_experiment(base_opt, logger=None, tb_logger=None, seed=123)

    assert len(phase_calls) == 1
    call = phase_calls[0]
    assert call["phase_name"] == "phase2"
    assert call["global_step_offset"] == 4
    assert call["phase_step"] == 0


def test_resume_phase2_uses_phase2_step_not_phase1_boundary(monkeypatch, tmp_path):
    from main_train_vrt import run_experiment

    phase_calls = []
    saved_states = []

    base_opt = {
        "path": {"task": str(tmp_path / "exp"), "options": str(tmp_path / "exp" / "options")},
        "train": {"two_run": {"enable": True}},
        "rank": 0,
        "datasets": {"train": {}, "test": {}},
    }
    phase1_opt = {"train": {"total_iter": 4}, "datasets": {"train": {}}, "rank": 0}
    phase2_opt = {"train": {"total_iter": 6}, "datasets": {"train": {}}, "rank": 0}
    initial_state = {
        "phase1_total_iter": 4,
        "phase2_total_iter": 6,
        "phase1_completed": True,
        "phase1_final_G": "models/4_G.pth",
        "phase1_final_E": None,
        "phase2_started": True,
        "global_step_offset": 4,
        "last_successful_phase_step": 4,
        "last_successful_global_step": 4,
    }

    monkeypatch.setattr("main_train_vrt.resolve_two_run_phase_opts", lambda _opt: (phase1_opt, phase2_opt))
    monkeypatch.setattr("main_train_vrt.dump_resolved_two_run_opts", lambda *args, **kwargs: {})
    monkeypatch.setattr("main_train_vrt.load_two_run_state", lambda _path: dict(initial_state))
    monkeypatch.setattr("main_train_vrt.two_run_state_path", lambda _opt: tmp_path / "two_run_state.json")
    monkeypatch.setattr("main_train_vrt.save_two_run_state", lambda _path, state: saved_states.append(dict(state)))
    monkeypatch.setattr("main_train_vrt.resolve_resume_phase", lambda _state: "phase2_resume")
    monkeypatch.setattr("main_train_vrt.close_shared_runtime", lambda _runtime: None)

    def fake_run_phase(phase_opt, shared_runtime, phase_name, global_step_offset, resume_state):
        phase_calls.append({
            "phase_name": phase_name,
            "global_step_offset": global_step_offset,
            "phase_step": resume_state["phase_step"],
        })
        return {
            "last_phase_step": 6,
            "last_global_step": 10,
            "final_checkpoint_G": "models/10_G.pth",
            "final_checkpoint_E": None,
            "runtime": {"model": object()},
        }

    monkeypatch.setattr("main_train_vrt.run_phase", fake_run_phase)
    monkeypatch.setattr("main_train_vrt.build_shared_runtime", lambda _opt, _logger, _tb_logger, seed: {
        "logger": _logger,
        "tb_logger": _tb_logger,
        "seed": seed,
    })

    run_experiment(base_opt, logger=None, tb_logger=None, seed=123)

    assert phase_calls == [{
        "phase_name": "phase2",
        "global_step_offset": 4,
        "phase_step": 0,
    }]
    assert saved_states[0]["last_successful_phase_step"] == 0
    assert saved_states[0]["last_successful_global_step"] == 4
