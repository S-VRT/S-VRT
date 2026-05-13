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


def test_fusion_only_scope_skips_full_model_compile(monkeypatch):
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
            "train": {"compile": {"enable": True, "scope": "fusion_only"}},
        }
    )
    model = nn.Linear(2, 2)

    assert base.compile_model_if_enabled(model) is model
    assert calls == []


def test_compile_fusion_modules_compiles_adapter_operator_only(monkeypatch):
    calls = []

    class _Adapter(nn.Module):
        def __init__(self, operator):
            super().__init__()
            self.operator = operator

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.fusion_operator = nn.Linear(2, 2)
            self.fusion_adapter = _Adapter(self.fusion_operator)

    def fake_compile(model, **kwargs):
        def compiled(*args, **kw):
            return model(*args, **kw)

        compiled._orig_mod = model
        calls.append((model, kwargs, compiled))
        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    base = _base(
        {
            "path": {"models": "/tmp"},
            "is_train": True,
            "dist": False,
            "train": {
                "compile": {
                    "enable": True,
                    "scope": "fusion_only",
                    "mode": "reduce-overhead",
                    "fullgraph": False,
                    "dynamic": True,
                    "backend": "inductor",
                }
            },
        }
    )
    net = _Net()
    original_operator = net.fusion_operator
    original_forward = original_operator.forward

    summary = base.compile_fusion_modules_if_enabled(net)

    assert len(calls) == 1
    assert calls[0][0].__self__ is original_operator
    assert calls[0][0].__func__ is original_forward.__func__
    assert calls[0][1] == {
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": True,
        "backend": "inductor",
    }
    assert net.fusion_operator is original_operator
    assert net.fusion_adapter.operator is original_operator
    assert net.fusion_operator.forward is calls[0][2]
    assert list(net.state_dict().keys()) == [
        "backbone.weight",
        "backbone.bias",
        "fusion_operator.weight",
        "fusion_operator.bias",
        "fusion_adapter.operator.weight",
        "fusion_adapter.operator.bias",
    ]
    assert summary == {"scope": "fusion_only", "compiled": ["fusion_operator"]}


def test_model_to_device_compiles_before_data_parallel(monkeypatch):
    import models.model_base as model_base_module

    events = []

    def fake_compile(model, **kwargs):
        events.append("compile")
        return model

    class _FakeDataParallel:
        def __init__(self, module):
            events.append("data_parallel")
            self.module = module

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(model_base_module, "DataParallel", _FakeDataParallel)

    base = _base(
        {
            "path": {"models": "/tmp"},
            "is_train": True,
            "dist": False,
            "train": {"compile": {"enable": True, "scope": "full_model"}},
        }
    )

    wrapped = base.model_to_device(nn.Linear(2, 2))

    assert isinstance(wrapped, _FakeDataParallel)
    assert events == ["compile", "data_parallel"]


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
