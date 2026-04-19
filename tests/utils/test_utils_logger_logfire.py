import json
import logging
from pathlib import Path

from utils import utils_logger
from utils import utils_option


def _write_minimal_opt(path_obj: Path):
    """Write a minimal option JSON for utils_option.parse tests."""
    payload = {
        "task": "unit_test_task",
        "scale": 1,
        "n_channels": 3,
        "datasets": {},
        "path": {
            "root": str(path_obj.parent),
        },
        "netG": {},
        "train": {},
    }
    path_obj.write_text(json.dumps(payload), encoding="utf-8")


def test_parse_sets_logfire_logging_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    opt_path = tmp_path / "opt.json"
    _write_minimal_opt(opt_path)

    opt = utils_option.parse(str(opt_path), is_train=True)

    assert opt["logging"]["use_logfire"] is False
    assert opt["logging"]["logfire_token"] is None
    assert opt["logging"]["logfire_project_name"] is None
    assert opt["logging"]["logfire_service_name"] == "s-vrt"
    assert opt["logging"]["logfire_environment"] is None
    assert opt["logging"]["logfire_log_text"] is True
    assert opt["logging"]["logfire_log_metrics"] is True
    assert opt["logging"]["logfire_log_timings"] is True


class _FakeLogfire:
    def __init__(self):
        self.configured = []
        self.events = []

    def configure(self, **kwargs):
        self.configured.append(kwargs)

    def info(self, event, **kwargs):
        self.events.append(("info", event, kwargs))

    def warning(self, event, **kwargs):
        self.events.append(("warning", event, kwargs))

    def error(self, event, **kwargs):
        self.events.append(("error", event, kwargs))


class _FailingInfoLogfire(_FakeLogfire):
    def info(self, event, **kwargs):
        raise RuntimeError("boom")


class _SelectiveFailInfoLogfire(_FakeLogfire):
    def __init__(self, failing_event):
        super().__init__()
        self.failing_event = failing_event

    def info(self, event, **kwargs):
        if event == self.failing_event:
            raise RuntimeError("boom")
        super().info(event, **kwargs)


class _MetricsFailingWarningLogfire(_FakeLogfire):
    def info(self, event, **kwargs):
        if event == "svrt metrics":
            raise RuntimeError("metrics boom")
        super().info(event, **kwargs)

    def warning(self, event, **kwargs):
        raise RuntimeError("warning boom")


def _make_opt(tmp_path, **logging_overrides):
    logging_cfg = {
        "use_tensorboard": False,
        "use_wandb": False,
        "use_swanlab": False,
        "use_logfire": False,
        "logfire_token": None,
        "logfire_project_name": None,
        "logfire_service_name": "s-vrt",
        "logfire_environment": None,
        "logfire_log_text": True,
        "logfire_log_metrics": True,
        "logfire_log_timings": True,
    }
    logging_cfg.update(logging_overrides)
    return {
        "task": "deblur",
        "opt_path": str(tmp_path / "opt.json"),
        "is_train": True,
        "rank": 0,
        "world_size": 1,
        "path": {
            "log": str(tmp_path / "logs"),
            "tensorboard": str(tmp_path / "tb"),
        },
        "logging": logging_cfg,
    }


def test_logger_disables_logfire_when_package_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", False)
    monkeypatch.setattr(utils_logger, "logfire", None)

    logger = utils_logger.Logger(_make_opt(tmp_path, use_logfire=True), logger=None)

    assert logger.logfire_bridge.enabled is False
    assert logger.logfire_bridge.metrics_enabled is False
    assert logger.logfire_bridge.timings_enabled is False


def test_logger_sends_scalars_and_timings_to_logfire(monkeypatch, tmp_path):
    fake = _FakeLogfire()
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    std_logger = logging.getLogger("test_logfire_metrics")
    std_logger.handlers = []
    std_logger.propagate = False

    logger = utils_logger.Logger(
        _make_opt(
            tmp_path,
            use_logfire=True,
            logfire_token="token-123",
            logfire_project_name="Deblur",
            logfire_environment="local",
        ),
        logger=std_logger,
    )

    logger.log_scalars(step=12, scalar_dict={"loss": 0.5}, tag_prefix="train")
    logger.log_timings(step=12, timings_dict={"data": 0.1}, prefix="time")

    assert fake.configured == [
        {
            "token": "token-123",
            "service_name": "s-vrt",
            "environment": "local",
        }
    ]

    metric_events = [event for event in fake.events if event[1] == "svrt metrics"]
    timing_events = [event for event in fake.events if event[1] == "svrt timings"]

    assert metric_events
    assert metric_events[0][2]["metrics"] == {"train/loss": 0.5}
    assert metric_events[0][2]["step"] == 12
    assert metric_events[0][2]["project_name"] == "Deblur"

    assert timing_events
    assert timing_events[0][2]["timings"] == {"time/data": 0.1}
    assert timing_events[0][2]["step"] == 12


def test_logger_info_attaches_single_logfire_text_handler(monkeypatch, tmp_path):
    fake = _FakeLogfire()
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    logger_name = "test_logfire_text"
    py_logger = logging.getLogger(logger_name)
    py_logger.handlers = []
    py_logger.propagate = False

    opt = _make_opt(tmp_path, use_logfire=True, logfire_log_text=True)
    log_path = tmp_path / "train.log"

    utils_logger.logger_info(logger_name, str(log_path), opt=opt)
    utils_logger.logger_info(logger_name, str(log_path), opt=opt)

    py_logger.info("hello logfire")

    text_handlers = [
        handler
        for handler in py_logger.handlers
        if handler.__class__.__name__ == "_LogfireLoggingHandler"
    ]
    text_events = [event for event in fake.events if event[1] == "svrt log record"]

    assert len(text_handlers) == 1
    assert len(text_events) == 1
    assert text_events[0][2]["message"] == "hello logfire"
    assert text_events[0][2]["logger_name"] == logger_name


def test_text_channel_disables_after_first_logfire_emit_error(monkeypatch, tmp_path):
    fake = _FailingInfoLogfire()
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    logger_name = "test_logfire_text_failure_isolation"
    py_logger = logging.getLogger(logger_name)
    py_logger.handlers = []
    py_logger.propagate = False

    opt = _make_opt(tmp_path, use_logfire=True, logfire_log_text=True)
    utils_logger.logger_info(logger_name, str(tmp_path / "train.log"), opt=opt)

    bridge = getattr(py_logger, "_svrt_logfire_bridge", None)
    assert bridge is not None
    assert bridge.text_enabled is True

    py_logger.info("first message should not escape")
    assert bridge.text_enabled is False

    py_logger.info("second message should be ignored")
    assert bridge.text_enabled is False


def test_text_failure_does_not_disable_metrics_channel(monkeypatch, tmp_path):
    fake = _SelectiveFailInfoLogfire(failing_event="svrt log record")
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    logger_name = "test_logfire_text_metrics_isolation"
    py_logger = logging.getLogger(logger_name)
    py_logger.handlers = []
    py_logger.propagate = False

    opt = _make_opt(tmp_path, use_logfire=True, logfire_log_text=True, logfire_log_metrics=True)
    utils_logger.logger_info(logger_name, str(tmp_path / "train.log"), opt=opt)

    bridge = getattr(py_logger, "_svrt_logfire_bridge", None)
    assert bridge is not None
    assert bridge.text_enabled is True
    assert bridge.metrics_enabled is True

    py_logger.info("text event triggers selective failure")

    logger = utils_logger.Logger(opt, logger=py_logger)
    logger.log_scalars(step=3, scalar_dict={"loss": 0.25}, tag_prefix="train")

    metric_events = [event for event in fake.events if event[1] == "svrt metrics"]

    assert bridge.text_enabled is False
    assert bridge.metrics_enabled is True
    assert logger.logfire_bridge.metrics_enabled is True
    assert metric_events
    assert metric_events[0][2]["metrics"] == {"train/loss": 0.25}
    assert metric_events[0][2]["step"] == 3


def test_metrics_failure_disables_only_metrics_not_text(monkeypatch, tmp_path):
    fake = _MetricsFailingWarningLogfire()
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    logger_name = "test_logfire_metrics_text_isolation"
    py_logger = logging.getLogger(logger_name)
    py_logger.handlers = []
    py_logger.propagate = False

    opt = _make_opt(tmp_path, use_logfire=True, logfire_log_text=True, logfire_log_metrics=True)
    utils_logger.logger_info(logger_name, str(tmp_path / "train.log"), opt=opt)

    bridge = getattr(py_logger, "_svrt_logfire_bridge", None)
    assert bridge is not None
    assert bridge.text_enabled is True
    assert bridge.metrics_enabled is True

    logger = utils_logger.Logger(opt, logger=py_logger)
    logger.log_scalars(step=7, scalar_dict={"loss": 0.7}, tag_prefix="train")

    assert bridge.metrics_enabled is False
    assert logger.logfire_bridge.metrics_enabled is False
    assert bridge.text_enabled is True
    assert logger.logfire_bridge.text_enabled is True


def test_timings_failure_disables_only_timings_channel(monkeypatch, tmp_path):
    fake = _SelectiveFailInfoLogfire(failing_event="svrt timings")
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    logger_name = "test_logfire_timings_isolation"
    py_logger = logging.getLogger(logger_name)
    py_logger.handlers = []
    py_logger.propagate = False

    opt = _make_opt(
        tmp_path,
        use_logfire=True,
        logfire_log_text=True,
        logfire_log_metrics=True,
        logfire_log_timings=True,
    )
    utils_logger.logger_info(logger_name, str(tmp_path / "train.log"), opt=opt)

    bridge = getattr(py_logger, "_svrt_logfire_bridge", None)
    assert bridge is not None

    logger = utils_logger.Logger(opt, logger=py_logger)
    logger.log_timings(step=11, timings_dict={"data": 0.11}, prefix="time")

    assert bridge.timings_enabled is False
    assert logger.logfire_bridge.timings_enabled is False
    assert bridge.text_enabled is True
    assert logger.logfire_bridge.text_enabled is True
    assert bridge.metrics_enabled is True
    assert logger.logfire_bridge.metrics_enabled is True

    text_events_before = [event for event in fake.events if event[1] == "svrt log record"]
    metric_events_before = [event for event in fake.events if event[1] == "svrt metrics"]

    py_logger.info("text survives timings failure")
    logger.log_scalars(step=12, scalar_dict={"loss": 0.12}, tag_prefix="train")

    text_events_after = [event for event in fake.events if event[1] == "svrt log record"]
    metric_events_after = [event for event in fake.events if event[1] == "svrt metrics"]

    new_text_events = text_events_after[len(text_events_before):]
    new_metric_events = metric_events_after[len(metric_events_before):]

    assert len(new_text_events) == 1
    assert new_text_events[0][2]["message"] == "text survives timings failure"

    assert len(new_metric_events) == 1
    assert new_metric_events[0][2]["metrics"] == {"train/loss": 0.12}
    assert new_metric_events[0][2]["step"] == 12


def test_metrics_channel_disables_after_first_logfire_error(monkeypatch, tmp_path):
    fake = _FailingInfoLogfire()
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    logger = utils_logger.Logger(_make_opt(tmp_path, use_logfire=True), logger=None)

    logger.log_scalars(step=1, scalar_dict={"loss": 1.0}, tag_prefix="train")
    logger.log_scalars(step=2, scalar_dict={"loss": 0.5}, tag_prefix="train")

    assert logger.logfire_bridge.metrics_enabled is False


def test_timings_channel_disables_after_first_logfire_error(monkeypatch, tmp_path):
    fake = _FailingInfoLogfire()
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    logger = utils_logger.Logger(_make_opt(tmp_path, use_logfire=True), logger=None)

    logger.log_timings(step=1, timings_dict={"data": 0.1}, prefix="time")
    logger.log_timings(step=2, timings_dict={"data": 0.2}, prefix="time")

    assert logger.logfire_bridge.timings_enabled is False
