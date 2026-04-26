from contextlib import nullcontext

from utils.utils_timer import Timer


class _TimerStub:
    def __init__(self):
        self.current_timings = {}

    def timer(self, name):
        self.current_timings[name] = 1.0
        return nullcontext()


def test_optimize_parameters_preserves_feed_data_timing(monkeypatch):
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.timer = _TimerStub()
    model.timer.current_timings["data_load"] = 0.25

    def stop_after_configure(_current_step):
        raise RuntimeError("stop before optimizer work")

    monkeypatch.setattr(model, "_configure_fusion_warmup_trainability", stop_after_configure)

    try:
        ModelPlain.optimize_parameters(model, current_step=1)
    except RuntimeError as exc:
        assert str(exc) == "stop before optimizer work"

    assert model.timer.current_timings["data_load"] == 0.25


def test_iteration_timer_boundary_clears_then_records_batch_wait():
    timer = Timer(device=None, sync_cuda=False)
    timer.current_timings["forward"] = 9.0

    timer.current_timings.clear()
    with timer.timer("batch_wait"):
        train_data = next(iter([{"L": "batch"}]))

    assert train_data == {"L": "batch"}
    current = timer.get_current_timings()
    assert "forward" not in current
    assert "batch_wait" in current


def test_ddp_timing_summary_adds_max_and_mean_for_time_keys():
    from main_train_vrt import build_timing_summary

    logs = {
        "time_batch_wait": 0.1,
        "time_forward": 0.8,
        "G_loss": 1.0,
    }

    summarized = build_timing_summary(logs, dist_enabled=False, device=None)

    assert summarized["time_batch_wait"] == 0.1
    assert summarized["time_batch_wait_max"] == 0.1
    assert summarized["time_batch_wait_mean"] == 0.1
    assert summarized["time_forward_max"] == 0.8
    assert summarized["time_forward_mean"] == 0.8
    assert summarized["G_loss"] == 1.0


def test_timing_summary_keeps_non_time_keys_unchanged():
    from main_train_vrt import build_timing_summary

    logs = {
        "time_forward": 0.5,
        "fusion_warmup_stage": "writeback_only",
        "G_loss": 0.1,
    }

    summarized = build_timing_summary(logs, dist_enabled=False, device=None)

    assert summarized["fusion_warmup_stage"] == "writeback_only"
    assert summarized["G_loss"] == 0.1
    assert summarized["time_forward"] == 0.5
    assert summarized["time_forward_max"] == 0.5
    assert summarized["time_forward_mean"] == 0.5
    assert "fusion_warmup_stage_max" not in summarized
