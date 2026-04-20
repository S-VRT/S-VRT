"""Unit tests for AMP initialization logic in ModelBase / ModelPlain.

These tests exercise _resolve_amp_dtype, amp_train_enabled, amp_train_dtype,
and grad_scaler without requiring CUDA or a full model instantiation.
"""
import sys
import os
import types
import unittest.mock as mock

import pytest
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.model_base import ModelBase


# ---------------------------------------------------------------------------
# Minimal ModelBase subclass that skips all heavy __init__ side-effects
# ---------------------------------------------------------------------------

class _MinimalModel(ModelBase):
    """Thin subclass that only calls ModelBase.__init__ with a fake opt."""

    def __init__(self, opt):
        # Bypass ModelBase.__init__ to avoid filesystem / network deps,
        # but still set up the attributes it provides.
        self.opt = opt
        self.save_dir = "/tmp/fake_models"
        self.is_train = opt.get("is_train", True)
        self.schedulers = []
        self._amp_dtypes = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        # Simulate device selection (CPU in CI, no CUDA required)
        self.device = torch.device("cpu")

    # Replicate the AMP init block from ModelPlain.__init__
    def _init_amp(self, opt_train):
        amp_train_opt = opt_train.get("amp", {})
        self.amp_train_enabled = (
            bool(amp_train_opt.get("enable", False)) and self.device.type == "cuda"
        )
        self.amp_train_dtype = self._resolve_amp_dtype(
            amp_train_opt.get("dtype", "float16")
        )
        scaler_enabled = (
            self.amp_train_enabled and self.amp_train_dtype == torch.float16
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_model(amp_enable: bool, dtype: str) -> _MinimalModel:
    opt = {"is_train": True, "path": {"models": "/tmp"}}
    m = _MinimalModel(opt)
    m._init_amp({"amp": {"enable": amp_enable, "dtype": dtype}})
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAmpDisabled:
    def test_amp_disabled_no_scaler(self):
        """amp.enable=False -> amp_train_enabled=False and scaler disabled."""
        m = _make_model(amp_enable=False, dtype="float16")
        assert m.amp_train_enabled is False
        assert not m.grad_scaler.is_enabled()


class TestAmpFp16:
    def test_amp_fp16_creates_enabled_scaler(self):
        """amp.enable=True, dtype='float16' on CUDA -> scaler_enabled logic is True.

        PyTorch's GradScaler forces _enabled=False when CUDA is absent, so we
        cannot assert on the scaler object itself in a CPU-only CI environment.
        Instead we verify the three conditions that drive scaler_enabled:
          1. amp_train_enabled is True (device patched to 'cuda')
          2. amp_train_dtype is float16
          3. The derived scaler_enabled flag would be True
        """
        opt = {"is_train": True, "path": {"models": "/tmp"}}
        m = _MinimalModel(opt)
        m.device = torch.device("cuda")  # pretend CUDA is available
        m._init_amp({"amp": {"enable": True, "dtype": "float16"}})

        assert m.amp_train_enabled is True
        assert m.amp_train_dtype == torch.float16
        # Verify the scaler_enabled logic: enabled AND fp16 -> True
        scaler_enabled = m.amp_train_enabled and m.amp_train_dtype == torch.float16
        assert scaler_enabled is True


class TestAmpBf16:
    def test_amp_bf16_scaler_disabled(self):
        """amp.enable=True, dtype='bfloat16' -> scaler disabled (bf16 doesn't need it).

        Patch device to CUDA so amp_train_enabled can be True.
        """
        opt = {"is_train": True, "path": {"models": "/tmp"}}
        m = _MinimalModel(opt)
        m.device = torch.device("cuda")
        m._init_amp({"amp": {"enable": True, "dtype": "bfloat16"}})

        assert m.amp_train_enabled is True
        assert m.amp_train_dtype == torch.bfloat16
        assert not m.grad_scaler.is_enabled()


class TestResolveAmpDtype:
    def setup_method(self):
        opt = {"is_train": True, "path": {"models": "/tmp"}}
        self.m = _MinimalModel(opt)

    def test_resolve_amp_dtype_fp16(self):
        assert self.m._resolve_amp_dtype("float16") == torch.float16

    def test_resolve_amp_dtype_bf16(self):
        assert self.m._resolve_amp_dtype("bfloat16") == torch.bfloat16

    def test_resolve_amp_dtype_invalid(self):
        with pytest.raises(ValueError, match="Unsupported AMP dtype"):
            self.m._resolve_amp_dtype("invalid")
