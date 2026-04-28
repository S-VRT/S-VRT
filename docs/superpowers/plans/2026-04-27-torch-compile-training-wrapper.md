# Torch Compile Training Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional wrapper-level `torch.compile` support to the S-VRT training path while preserving KAIR-comparable model semantics.

**Architecture:** Extend `ModelBase` with a small compile helper called from `model_to_device()` after device transfer and before DDP/DataParallel wrapping. Keep compile disabled by default, make failure behavior configurable, and unwrap compiled `_orig_mod` modules for checkpoint-compatible save/load.

**Tech Stack:** PyTorch `torch.compile`, existing `ModelBase`, existing DDP/DataParallel wrappers, pytest.

---

## File Structure

- Modify `models/model_base.py`: add compile config resolution, compile helper, compiled wrapper unwrapping, and call compile inside `model_to_device()`.
- Modify `options/gopro_rgbspike_server.json`: add documented `train.compile` config with `enable=false`.
- Modify `options/gopro_rgbspike_server_debug.json`: add documented `train.compile` config with `enable=false`.
- Create `tests/models/test_torch_compile_wrapper.py`: unit tests for compile ordering, fallback behavior, disabled default, and `_orig_mod` unwrapping.

## Task 1: Compile Helper Tests

**Files:**
- Create: `tests/models/test_torch_compile_wrapper.py`
- Modify: `models/model_base.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_torch_compile_wrapper.py` with:

```python
import pytest
import torch
from torch import nn

from models.model_base import ModelBase


def _base(opt=None):
    base = ModelBase.__new__(ModelBase)
    base.opt = opt or {
        "path": {"models": "/tmp"},
        "is_train": True,
        "dist": False,
        "train": {},
    }
    base.device = torch.device("cpu")
    return base


def test_compile_disabled_by_default(monkeypatch):
    called = []

    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: called.append((model, kwargs)) or model)

    base = _base()
    model = nn.Linear(2, 2)

    assert base.compile_model_if_enabled(model) is model
    assert called == []


def test_compile_uses_train_compile_options(monkeypatch):
    calls = []

    def fake_compile(model, **kwargs):
        calls.append((model, kwargs))
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)

    base = _base(
        {
            "path": {"models": "/tmp"},
            "is_train": True,
            "dist": False,
            "train": {
                "compile": {
                    "enable": True,
                    "mode": "reduce-overhead",
                    "fullgraph": False,
                    "dynamic": True,
                    "backend": "inductor",
                    "fallback_on_error": True,
                }
            },
        }
    )
    model = nn.Linear(2, 2)

    assert base.compile_model_if_enabled(model) is model
    assert calls == [
        (
            model,
            {
                "mode": "reduce-overhead",
                "fullgraph": False,
                "dynamic": True,
                "backend": "inductor",
            },
        )
    ]


def test_compile_falls_back_when_enabled(monkeypatch):
    def failing_compile(model, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", failing_compile)

    base = _base(
        {
            "path": {"models": "/tmp"},
            "is_train": True,
            "dist": False,
            "rank": 0,
            "train": {"compile": {"enable": True, "fallback_on_error": True}},
        }
    )
    model = nn.Linear(2, 2)

    assert base.compile_model_if_enabled(model) is model


def test_compile_raises_when_fallback_disabled(monkeypatch):
    def failing_compile(model, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", failing_compile)

    base = _base(
        {
            "path": {"models": "/tmp"},
            "is_train": True,
            "dist": False,
            "train": {"compile": {"enable": True, "fallback_on_error": False}},
        }
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        base.compile_model_if_enabled(nn.Linear(2, 2))


def test_get_bare_model_unwraps_compiled_orig_mod():
    original = nn.Linear(2, 2)

    class CompiledWrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

    base = _base()

    assert base.get_bare_model(CompiledWrapper(original)) is original
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/models/test_torch_compile_wrapper.py
```

Expected: failures because `compile_model_if_enabled()` does not exist and `get_bare_model()` does not unwrap `_orig_mod`.

- [ ] **Step 3: Implement compile helper and unwrapping**

In `models/model_base.py`, add `import logging` near imports, then add these methods to `ModelBase`:

```python
    def _compile_options(self):
        train_opt = self.opt.get('train', {}) or {}
        compile_opt = train_opt.get('compile', {}) or {}
        if not isinstance(compile_opt, dict):
            raise ValueError("train.compile must be a dict when provided.")
        return compile_opt

    def compile_model_if_enabled(self, network):
        compile_opt = self._compile_options()
        if not bool(compile_opt.get('enable', False)):
            return network
        if not hasattr(torch, 'compile'):
            if bool(compile_opt.get('fallback_on_error', True)):
                if self.opt.get('rank', 0) == 0:
                    logging.getLogger('train').warning('[COMPILE] torch.compile unavailable; using eager model.')
                return network
            raise RuntimeError('torch.compile is unavailable in this PyTorch build.')

        kwargs = {
            'mode': compile_opt.get('mode', 'default'),
            'fullgraph': bool(compile_opt.get('fullgraph', False)),
            'dynamic': bool(compile_opt.get('dynamic', True)),
            'backend': compile_opt.get('backend', 'inductor'),
        }
        try:
            compiled = torch.compile(network, **kwargs)
            if self.opt.get('rank', 0) == 0:
                logging.getLogger('train').info('[COMPILE] Enabled torch.compile with %s', kwargs)
            return compiled
        except Exception as exc:
            if bool(compile_opt.get('fallback_on_error', True)):
                if self.opt.get('rank', 0) == 0:
                    logging.getLogger('train').warning(
                        '[COMPILE] torch.compile failed (%s); using eager model.', exc
                    )
                return network
            raise
```

Update `get_bare_model()`:

```python
    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel, DataParallel, or torch.compile.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        if hasattr(network, '_orig_mod'):
            network = network._orig_mod
        return network
```

Update `model_to_device()` immediately after `network = network.to(self.device)`:

```python
        network = self.compile_model_if_enabled(network)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
uv run pytest tests/models/test_torch_compile_wrapper.py
```

Expected: all tests pass.

## Task 2: Config Defaults

**Files:**
- Modify: `options/gopro_rgbspike_server.json`
- Modify: `options/gopro_rgbspike_server_debug.json`

- [ ] **Step 1: Add compile config**

In each file's `"train"` object, after `"amp"` if present or after `"profiler"`/checkpoint settings if there is no `"amp"`, add:

```json
    ,
    "compile": {
      "enable": false,
      "mode": "reduce-overhead",
      "fullgraph": false,
      "dynamic": true,
      "backend": "inductor",
      "fallback_on_error": true
    }
```

- [ ] **Step 2: Validate config parse**

Run:

```bash
uv run python - <<'PY'
from utils import utils_option as option
for path in ["options/gopro_rgbspike_server.json", "options/gopro_rgbspike_server_debug.json"]:
    opt = option.parse(path, is_train=True)
    cfg = opt["train"]["compile"]
    assert cfg["enable"] is False
    assert cfg["backend"] == "inductor"
    print(path, cfg)
PY
```

Expected: both configs parse and print compile dictionaries.

## Task 3: Regression Suite

**Files:**
- Test only.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/models/test_torch_compile_wrapper.py tests/models/test_lora.py::test_modelplain_injects_lora_during_init_train tests/models/test_lora.py::test_lora_checkpoint_resume_strict_load
```

Expected: pass. This checks compile wrapper behavior and that save/load/LoRA init ordering remains intact.

- [ ] **Step 2: Inspect diff**

Run:

```bash
git diff -- models/model_base.py options/gopro_rgbspike_server.json options/gopro_rgbspike_server_debug.json tests/models/test_torch_compile_wrapper.py
```

Expected: changes are limited to wrapper-level compile support, config knobs, and tests. No KAIR files and no VRT architecture files changed.
