from contextlib import nullcontext

import torch

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


def test_model_plain_timer_does_not_sync_cuda_by_default(monkeypatch):
    import models.model_plain as model_plain_module
    from models.model_plain import ModelPlain

    captured = {}

    class _TimerSpy:
        def __init__(self, device=None, sync_cuda=True):
            captured["sync_cuda"] = sync_cuda
            self.current_timings = {}

    monkeypatch.setattr(model_plain_module, "define_G", lambda _opt: object())
    monkeypatch.setattr(model_plain_module.ModelBase, "model_to_device", lambda _self, net: net)
    monkeypatch.setattr(model_plain_module, "Timer", _TimerSpy)

    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"train": {}}
    model.opt_train = {}
    model.netG = type("Net", (), {"train": lambda _self: None})()
    model.device = type("Device", (), {"type": "cpu"})()
    model.load = lambda: None
    model.define_loss = lambda: None
    model.define_optimizer = lambda: None
    model.load_optimizers = lambda: None
    model.define_scheduler = lambda: None
    model.get_bare_model = lambda net: net

    ModelPlain.init_train(model)

    assert captured["sync_cuda"] is False


def test_optimize_parameters_skips_optimizer_when_loss_is_nonfinite():
    from models.model_plain import ModelPlain

    class _Net:
        def __call__(self, _x):
            return torch.tensor(float("nan"), requires_grad=True)

    class _Optimizer:
        def __init__(self):
            self.zero_grad_calls = 0
            self.step_calls = 0

        def zero_grad(self):
            self.zero_grad_calls += 1

        def step(self):
            self.step_calls += 1

    class _Scaler:
        def is_enabled(self):
            return False

    model = ModelPlain.__new__(ModelPlain)
    model._trace_current_step = None
    model.configure_calls = 0
    model.timer = _TimerStub()
    model.opt_train = {
        "G_optimizer_clipgrad": 0,
        "G_regularizer_orthstep": 0,
        "G_regularizer_clipstep": 0,
        "checkpoint_save": 999,
        "E_decay": 0,
    }
    model.opt = {"train": {"checkpoint_save": 999}}
    model.fix_iter = 0
    model.G_optimizer = _Optimizer()
    model.grad_scaler = _Scaler()
    model.amp_train_enabled = False
    model.amp_train_dtype = torch.float16
    model.netG = _Net()
    model.L = torch.tensor([1.0])
    model.L_flow_spike = None
    model.H = torch.tensor([1.0])
    model.G_lossfn_weight = 1.0
    model.G_lossfn = lambda e, _h: e
    model.log_dict = {}
    model.parameters = lambda: []
    model.get_bare_model = lambda net: net
    model._record_fusion_diagnostics_to_log = lambda _meta: None
    model._compute_fusion_aux_loss = lambda **_kwargs: torch.tensor(0.0)
    model._configure_fusion_warmup_trainability = lambda _step: None
    model._autocast_context = lambda **_kwargs: nullcontext()
    model.update_E = lambda _decay: (_ for _ in ()).throw(AssertionError("update_E should be skipped"))

    ModelPlain.optimize_parameters(model, current_step=6001)

    assert model.G_optimizer.zero_grad_calls == 1
    assert model.G_optimizer.step_calls == 0
    assert model.log_dict["G_loss"] != model.log_dict["G_loss"]
    assert model.log_dict["nan_guard_skip"] == 1.0
    assert model.log_dict["nan_guard_step"] == 6001.0
