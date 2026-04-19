# Logfire Logger Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Logfire as an optional parallel logging backend for the unified logger so text logs, scalar metrics, and timings can be sent to Logfire without breaking existing TensorBoard, W&B, or SwanLab behavior.

**Architecture:** Keep the implementation centered in [utils/utils_logger.py](../../../utils/utils_logger.py) by adding a small internal Logfire bridge and a lightweight logging.Handler for text logs. Extend [utils/utils_option.py](../../../utils/utils_option.py) with Logfire config defaults, pass `opt` into `logger_info()` from [main_train_vrt.py](../../../main_train_vrt.py), and cover the behavior with focused pytest tests under `tests/utils/`.

**Tech Stack:** Python stdlib logging, pytest, optional `logfire` dependency.

**Spec:** [docs/superpowers/specs/2026-04-18-logfire-logger-design.md](../specs/2026-04-18-logfire-logger-design.md)

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `utils/utils_option.py` | Modify | Add `opt['logging']` defaults for Logfire |
| `utils/utils_logger.py` | Modify | Add optional Logfire import, `_LogfireBridge`, `_LogfireLoggingHandler`, `logger_info(..., opt=None)`, and Logger integration |
| `main_train_vrt.py` | Modify | Pass `opt` into `logger_info()` so text-log bridge can be configured before training logs begin |
| `tests/utils/test_utils_logger_logfire.py` | Create | Regression tests for defaults, text logs, metrics, timings, missing dependency, and failure isolation |
| `requirement.txt` | Modify | Add `logfire` package to the documented environment |
| `README.md` | Modify | Document Logfire installation/configuration alongside TensorBoard/W&B/SwanLab |

---

## Task 1: Add Logfire config defaults in `utils_option`

**Files:**
- Modify: `utils/utils_option.py`
- Create: `tests/utils/test_utils_logger_logfire.py`

- [ ] **Step 1: Write the failing test for parse defaults**

Create `tests/utils/test_utils_logger_logfire.py` with this initial content:

```python
import json
from pathlib import Path

from utils import utils_option


def _write_minimal_opt(path_obj: Path):
    path_obj.write_text(
        json.dumps(
            {
                "task": "logfire_test",
                "scale": 1,
                "n_channels": 3,
                "datasets": {},
                "path": {"root": str(path_obj.parent / "experiments")},
                "netG": {},
                "train": {},
            }
        ),
        encoding="utf-8",
    )


def test_parse_sets_logfire_logging_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    opt_path = tmp_path / "opt.json"
    _write_minimal_opt(opt_path)

    opt = utils_option.parse(str(opt_path), is_train=True)
    logging_opt = opt["logging"]

    assert logging_opt["use_logfire"] is False
    assert logging_opt["logfire_token"] is None
    assert logging_opt["logfire_project_name"] is None
    assert logging_opt["logfire_service_name"] == "s-vrt"
    assert logging_opt["logfire_environment"] is None
    assert logging_opt["logfire_log_text"] is True
    assert logging_opt["logfire_log_metrics"] is True
    assert logging_opt["logfire_log_timings"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_parse_sets_logfire_logging_defaults -v`
Expected: FAIL with `KeyError: 'use_logfire'`

- [ ] **Step 3: Add the new defaults in `utils_option.parse()`**

Edit [utils/utils_option.py:55-90](../../../utils/utils_option.py#L55-L90). After the existing SwanLab defaults, add:

```python
    if 'use_logfire' not in opt['logging']:
        opt['logging']['use_logfire'] = False
    if 'logfire_token' not in opt['logging']:
        opt['logging']['logfire_token'] = None
    if 'logfire_project_name' not in opt['logging']:
        opt['logging']['logfire_project_name'] = None
    if 'logfire_service_name' not in opt['logging']:
        opt['logging']['logfire_service_name'] = 's-vrt'
    if 'logfire_environment' not in opt['logging']:
        opt['logging']['logfire_environment'] = None
    if 'logfire_log_text' not in opt['logging']:
        opt['logging']['logfire_log_text'] = True
    if 'logfire_log_metrics' not in opt['logging']:
        opt['logging']['logfire_log_metrics'] = True
    if 'logfire_log_timings' not in opt['logging']:
        opt['logging']['logfire_log_timings'] = True
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_parse_sets_logfire_logging_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils/utils_option.py tests/utils/test_utils_logger_logfire.py
git commit -m "feat(logging): add Logfire config defaults"
```

---

## Task 2: Add the Logfire bridge for metrics and timings

**Files:**
- Modify: `utils/utils_logger.py`
- Modify: `tests/utils/test_utils_logger_logfire.py`

- [ ] **Step 1: Write failing tests for missing dependency and structured metric/timing emission**

Append to `tests/utils/test_utils_logger_logfire.py`:

```python
import logging

from utils import utils_logger


class _FakeLogfire:
    def __init__(self):
        self.configured = []
        self.events = []

    def configure(self, **kwargs):
        self.configured.append(kwargs)

    def info(self, message, **kwargs):
        self.events.append(("info", message, kwargs))

    def warning(self, message, **kwargs):
        self.events.append(("warning", message, kwargs))

    def error(self, message, **kwargs):
        self.events.append(("error", message, kwargs))


class _FailingInfoLogfire(_FakeLogfire):
    def info(self, message, **kwargs):
        raise RuntimeError("boom")


def _make_opt(tmp_path, **logging_overrides):
    logging_opt = {
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
    logging_opt.update(logging_overrides)
    return {
        "task": "logfire_test",
        "opt_path": str(tmp_path / "opt.json"),
        "is_train": True,
        "rank": 0,
        "world_size": 1,
        "path": {"log": str(tmp_path / "logs"), "tensorboard": str(tmp_path / "tb")},
        "logging": logging_opt,
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

    py_logger = logging.getLogger("test_logfire_metrics")
    py_logger.handlers = []
    py_logger.propagate = False

    logger = utils_logger.Logger(
        _make_opt(
            tmp_path,
            use_logfire=True,
            logfire_token="token-123",
            logfire_project_name="Deblur",
            logfire_environment="local",
        ),
        logger=py_logger,
    )

    logger.log_scalars(step=12, scalar_dict={"loss": 0.5}, tag_prefix="train")
    logger.log_timings(step=12, timings_dict={"data": 0.1}, prefix="time")

    assert fake.configured == [
        {"token": "token-123", "service_name": "s-vrt", "environment": "local"}
    ]

    metric_events = [event for event in fake.events if event[1] == "svrt metrics"]
    timing_events = [event for event in fake.events if event[1] == "svrt timings"]

    assert metric_events[0][2]["metrics"] == {"train/loss": 0.5}
    assert metric_events[0][2]["step"] == 12
    assert metric_events[0][2]["project_name"] == "Deblur"
    assert timing_events[0][2]["timings"] == {"time/data": 0.1}
    assert timing_events[0][2]["step"] == 12
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_logger_disables_logfire_when_package_missing tests/utils/test_utils_logger_logfire.py::test_logger_sends_scalars_and_timings_to_logfire -v`
Expected: FAIL because `Logger` has no `logfire_bridge` and does not emit Logfire events.

- [ ] **Step 3: Implement `_LogfireBridge` and connect it to `Logger`**

Edit [utils/utils_logger.py](../../../utils/utils_logger.py).

At the top of the file, after the SwanLab import block, add:

```python
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except Exception:
    logfire = None
    LOGFIRE_AVAILABLE = False
```

Then, above `class Logger`, add these helpers:

```python
def _compact_logfire_fields(fields):
    return {k: v for k, v in fields.items() if v is not None}


class _LogfireBridge:
    def __init__(self, opt, logger=None):
        self.logger = logger
        self.logfire = None
        self.enabled = False
        self.text_enabled = False
        self.metrics_enabled = False
        self.timings_enabled = False
        self._disabled_channels = set()

        logging_config = opt.get('logging', {})
        self.context = _compact_logfire_fields({
            'task': opt.get('task'),
            'opt_path': opt.get('opt_path'),
            'rank': opt.get('rank', 0),
            'world_size': opt.get('world_size', 1),
            'is_train': opt.get('is_train'),
            'project_name': logging_config.get('logfire_project_name'),
            'service_name': logging_config.get('logfire_service_name', 's-vrt'),
            'environment': logging_config.get('logfire_environment'),
            'run_name': logging_config.get('wandb_name')
                        or logging_config.get('swanlab_name')
                        or opt.get('task', 'experiment'),
        })

        if opt.get('rank', 0) != 0:
            return
        if not logging_config.get('use_logfire', False):
            return
        if not LOGFIRE_AVAILABLE:
            self._warn_once('Logfire is not available. Please install logfire: pip install logfire')
            return

        configure_kwargs = {}
        token = logging_config.get('logfire_token')
        if token:
            configure_kwargs['token'] = token
        if self.context.get('service_name'):
            configure_kwargs['service_name'] = self.context['service_name']
        if self.context.get('environment'):
            configure_kwargs['environment'] = self.context['environment']

        try:
            logfire.configure(**configure_kwargs)
        except Exception as e:
            self._warn_once(f'Failed to initialize Logfire: {e}')
            return

        self.logfire = logfire
        self.enabled = True
        self.text_enabled = logging_config.get('logfire_log_text', True)
        self.metrics_enabled = logging_config.get('logfire_log_metrics', True)
        self.timings_enabled = logging_config.get('logfire_log_timings', True)

    def _warn_once(self, message):
        if self.logger is not None:
            self.logger.warning(message)
        else:
            print(f'Warning: {message}')

    def _disable_channel(self, channel, exc):
        if channel in self._disabled_channels:
            return
        self._disabled_channels.add(channel)
        if channel == 'text':
            self.text_enabled = False
        elif channel == 'metrics':
            self.metrics_enabled = False
        elif channel == 'timings':
            self.timings_enabled = False
        self._warn_once(f'Disabling Logfire {channel} logging after error: {exc}')

    def emit_metrics(self, step, scalar_dict, tag_prefix=''):
        if not self.enabled or not self.metrics_enabled or not scalar_dict:
            return
        prefix = tag_prefix.rstrip('/') + '/' if tag_prefix else ''
        metrics = {
            f'{prefix}{key}': value
            for key, value in scalar_dict.items()
            if value is not None
        }
        if not metrics:
            return
        try:
            self.logfire.info('svrt metrics', step=step, metrics=metrics, **self.context)
        except Exception as e:
            self._disable_channel('metrics', e)

    def emit_timings(self, step, timings_dict, prefix='timings'):
        if not self.enabled or not self.timings_enabled or not timings_dict:
            return
        timings = {
            f'{prefix}/{key}': value
            for key, value in timings_dict.items()
        }
        try:
            self.logfire.info('svrt timings', step=step, timings=timings, **self.context)
        except Exception as e:
            self._disable_channel('timings', e)
```

Then, inside `Logger.__init__()`, after `logging_config = opt.get('logging', {})`, add:

```python
        attached_bridge = getattr(logger, '_svrt_logfire_bridge', None) if logger is not None else None
        self.logfire_bridge = attached_bridge if attached_bridge is not None else _LogfireBridge(opt, logger=logger)
```

And inside `log_scalars()` / `log_timings()`, append:

```python
        self.logfire_bridge.emit_metrics(step, scalar_dict, tag_prefix=tag_prefix)
```

```python
        self.logfire_bridge.emit_timings(step=step, timings_dict=timings_dict, prefix=prefix)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_logger_disables_logfire_when_package_missing tests/utils/test_utils_logger_logfire.py::test_logger_sends_scalars_and_timings_to_logfire -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils/utils_logger.py tests/utils/test_utils_logger_logfire.py
git commit -m "feat(logging): add Logfire bridge for metrics and timings"
```

---

## Task 3: Bridge standard text logs into Logfire

**Files:**
- Modify: `utils/utils_logger.py`
- Modify: `main_train_vrt.py`
- Modify: `tests/utils/test_utils_logger_logfire.py`

- [ ] **Step 1: Write the failing test for `logger_info(..., opt=...)`**

Append to `tests/utils/test_utils_logger_logfire.py`:

```python
def test_logger_info_attaches_single_logfire_text_handler(monkeypatch, tmp_path):
    fake = _FakeLogfire()
    monkeypatch.setattr(utils_logger, 'LOGFIRE_AVAILABLE', True)
    monkeypatch.setattr(utils_logger, 'logfire', fake)

    logger_name = 'test_logfire_text'
    py_logger = logging.getLogger(logger_name)
    py_logger.handlers = []
    py_logger.propagate = False

    opt = _make_opt(tmp_path, use_logfire=True, logfire_log_text=True)
    log_path = tmp_path / 'train.log'

    utils_logger.logger_info(logger_name, str(log_path), opt=opt)
    utils_logger.logger_info(logger_name, str(log_path), opt=opt)

    py_logger.info('hello logfire')

    text_handlers = [
        handler for handler in py_logger.handlers
        if handler.__class__.__name__ == '_LogfireLoggingHandler'
    ]
    text_events = [event for event in fake.events if event[1] == 'svrt log record']

    assert len(text_handlers) == 1
    assert len(text_events) == 1
    assert text_events[0][2]['message'] == 'hello logfire'
    assert text_events[0][2]['logger_name'] == logger_name
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_logger_info_attaches_single_logfire_text_handler -v`
Expected: FAIL because `logger_info()` does not accept `opt` and no Logfire text handler exists.

- [ ] **Step 3: Add `_LogfireLoggingHandler`, extend `logger_info()`, and pass `opt` from `main_train_vrt.py`**

Edit [utils/utils_logger.py](../../../utils/utils_logger.py).

Above `logger_info()`, add:

```python
class _LogfireLoggingHandler(logging.Handler):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record):
        if not self.bridge.enabled or not self.bridge.text_enabled:
            return
        try:
            level_name = record.levelname.lower()
            log_method = getattr(self.bridge.logfire, level_name, self.bridge.logfire.info)
            log_method(
                'svrt log record',
                message=record.getMessage(),
                logger_name=record.name,
                level=record.levelname,
                pathname=record.pathname,
                lineno=record.lineno,
                **self.bridge.context,
            )
        except Exception as e:
            self.bridge._disable_channel('text', e)
```

Then change the `logger_info` signature and body:

```python
def logger_info(logger_name, log_path='default_logger.log', opt=None):
    try:
        rank_env = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    except Exception:
        rank_env = 0
    if rank_env != 0:
        return

    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        timestamp = datetime.datetime.now().strftime('_%y%m%d_%H%M%S')
        if os.path.isdir(log_path):
            log_file = os.path.join(log_path, logger_name + timestamp + '.log')
        else:
            dir_name = os.path.dirname(log_path)
            file_name = os.path.basename(log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            name, ext = os.path.splitext(file_name)
            log_file = os.path.join(dir_name, name + timestamp + ext) if dir_name else (name + timestamp + ext)
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        print(f'Log file created: {log_file}')
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)

    if opt is not None and not hasattr(log, '_svrt_logfire_bridge'):
        bridge = _LogfireBridge(opt, logger=log)
        log._svrt_logfire_bridge = bridge
        if bridge.enabled and bridge.text_enabled:
            if not any(isinstance(handler, _LogfireLoggingHandler) for handler in log.handlers):
                log.addHandler(_LogfireLoggingHandler(bridge))
```

Then update [main_train_vrt.py:173-183](../../../main_train_vrt.py#L173-L183):

Replace:

```python
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
```

with:

```python
        utils_logger.logger_info(
            logger_name,
            os.path.join(opt['path']['log'], logger_name + '.log'),
            opt=opt,
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_logger_info_attaches_single_logfire_text_handler -v`
Expected: PASS

- [ ] **Step 5: Parse-check the training entrypoint**

Run: `python -m py_compile main_train_vrt.py`
Expected: No output

- [ ] **Step 6: Commit**

```bash
git add utils/utils_logger.py main_train_vrt.py tests/utils/test_utils_logger_logfire.py
git commit -m "feat(logging): bridge text logs into Logfire"
```

---

## Task 4: Add failure isolation so Logfire problems never break training

**Files:**
- Modify: `utils/utils_logger.py`
- Modify: `tests/utils/test_utils_logger_logfire.py`

- [ ] **Step 1: Write failing tests for channel shutdown after errors**

Append to `tests/utils/test_utils_logger_logfire.py`:

```python
def test_metrics_channel_disables_after_first_logfire_error(monkeypatch, tmp_path):
    fake = _FailingInfoLogfire()
    monkeypatch.setattr(utils_logger, 'LOGFIRE_AVAILABLE', True)
    monkeypatch.setattr(utils_logger, 'logfire', fake)

    logger = utils_logger.Logger(_make_opt(tmp_path, use_logfire=True), logger=None)

    logger.log_scalars(step=1, scalar_dict={'loss': 1.0}, tag_prefix='train')
    logger.log_scalars(step=2, scalar_dict={'loss': 0.5}, tag_prefix='train')

    assert logger.logfire_bridge.metrics_enabled is False


def test_timings_channel_disables_after_first_logfire_error(monkeypatch, tmp_path):
    fake = _FailingInfoLogfire()
    monkeypatch.setattr(utils_logger, 'LOGFIRE_AVAILABLE', True)
    monkeypatch.setattr(utils_logger, 'logfire', fake)

    logger = utils_logger.Logger(_make_opt(tmp_path, use_logfire=True), logger=None)

    logger.log_timings(step=1, timings_dict={'data': 0.1}, prefix='time')
    logger.log_timings(step=2, timings_dict={'data': 0.2}, prefix='time')

    assert logger.logfire_bridge.timings_enabled is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_metrics_channel_disables_after_first_logfire_error tests/utils/test_utils_logger_logfire.py::test_timings_channel_disables_after_first_logfire_error -v`
Expected: FAIL before `_disable_channel()` is wired into all Logfire emit paths.

- [ ] **Step 3: Make sure every Logfire emit path isolates its own failures**

Edit [utils/utils_logger.py](../../../utils/utils_logger.py):

- Keep the `emit_metrics()` / `emit_timings()` `try/except` blocks from Task 2.
- In `_LogfireLoggingHandler.emit()`, keep its own `try/except` and call `self.bridge._disable_channel('text', e)`.
- Do **not** re-raise exceptions from any of the three channels.

The final `emit()` body should remain:

```python
    def emit(self, record):
        if not self.bridge.enabled or not self.bridge.text_enabled:
            return
        try:
            level_name = record.levelname.lower()
            log_method = getattr(self.bridge.logfire, level_name, self.bridge.logfire.info)
            log_method(
                'svrt log record',
                message=record.getMessage(),
                logger_name=record.name,
                level=record.levelname,
                pathname=record.pathname,
                lineno=record.lineno,
                **self.bridge.context,
            )
        except Exception as e:
            self.bridge._disable_channel('text', e)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_metrics_channel_disables_after_first_logfire_error tests/utils/test_utils_logger_logfire.py::test_timings_channel_disables_after_first_logfire_error -v`
Expected: PASS

- [ ] **Step 5: Run the full feature test file**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add utils/utils_logger.py tests/utils/test_utils_logger_logfire.py
git commit -m "fix(logging): isolate Logfire failures per channel"
```

---

## Task 5: Update dependency and user-facing docs

**Files:**
- Modify: `requirement.txt`
- Modify: `README.md`

- [ ] **Step 1: Add the optional dependency to the environment file**

Edit [requirement.txt](../../../requirement.txt). After `swanlab`, add:

```text
logfire
```

So the tail of the file becomes:

```text
tensorboard
wandb
swanlab
logfire
matplotlib
snntorch
```

- [ ] **Step 2: Update the dependency list in the README**

Edit [README.md:26-37](../../../README.md#L26-L37). Add `logfire` to the package list:

```markdown
主要依赖包括：
- PyTorch
- opencv-python
- scikit-image
- pillow
- torchvision
- timm
- einops
- tensorboard
- wandb
- swanlab
- logfire
```

- [ ] **Step 3: Update the logging section in the README**

Edit [README.md:310-316](../../../README.md#L310-L316). Replace the three-line list with:

```markdown
训练过程支持 TensorBoard、WANDB、SwanLab 与 Logfire 日志记录：

- **TensorBoard**：`tensorboard --logdir experiments/[experiment_name]/tb_logger`
- **WANDB**：在配置文件中设置 `wandb_api_key` 与 `wandb_project`（或使用 `WANDB_API_KEY` 环境变量）
- **SwanLab**：在配置文件中设置 `swanlab_project`/`swanlab_api_key` 或通过 `SWANLAB_API_KEY` + `swanlab login` 完成授权，使用 `swanlab_mode: "offline"` 可在无网络环境记录。默认会在 `experiments/<task_name>/swanlab_run.id` 中缓存云端 run id，便于在恢复训练（例如从 10k 步 checkpoint 继续）时自动续写同一次 run；如需开启新 run，可删除该文件或将 `swanlab_auto_resume` 置为 `false`。
- **Logfire**：在配置文件的 `logging` 段设置 `use_logfire`、`logfire_token`、`logfire_project_name`、`logfire_service_name`、`logfire_environment`。启用后，训练文本日志以及 `train/*`、`test/*`、`time/*` 指标会并行发送到 Logfire；Logfire 异常不会中断训练。
```

- [ ] **Step 4: Sanity-check the docs-only changes**

Run: `python -m py_compile utils/utils_option.py utils/utils_logger.py main_train_vrt.py`
Expected: No output

- [ ] **Step 5: Commit**

```bash
git add requirement.txt README.md
git commit -m "docs(logging): document Logfire setup and usage"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run the focused Logfire test suite**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py -v`
Expected: All tests pass

- [ ] **Step 2: Run the existing smoke test that exercises `Logger.log_timings()`**

Run: `python tests/smoke/run_vrt_timer_test.py`
Expected: Script exits successfully and prints `output shape:` plus one `timings:` line

- [ ] **Step 3: Verify the main training entrypoint still parses**

Run: `python -m py_compile main_train_vrt.py`
Expected: No output

- [ ] **Step 4: Verify the dependency and docs surface**

Run: `python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text(encoding='utf-8')
reqs = Path('requirement.txt').read_text(encoding='utf-8')
assert 'Logfire' in text
assert 'logfire' in reqs
print('OK')
PY`
Expected: `OK`

---

## Notes for the engineer

- Keep the Logfire implementation inside `utils/utils_logger.py`; do not create a separate integration layer unless the spec changes.
- Preserve backward compatibility for `logger_info()` call sites outside the main training entrypoint by keeping the new `opt` parameter optional.
- Do not send the whole `opt` dict to Logfire. Only send the curated context fields from the spec.
- Keep Logfire rank-gated to rank 0 so DDP does not duplicate text logs or metrics.
- Do not replace TensorBoard, W&B, or SwanLab behavior in this change; Logfire is additive.
