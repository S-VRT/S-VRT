# Stage C LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage C LoRA fine-tuning: freeze VRT backbone, inject trainable LoRA adapters into attention `qkv_*`/`proj` layers, and auto-export a merged (key-for-key identical to original VRT) checkpoint at end of training.

**Architecture:** Add a self-contained `models/lora.py` module with `LoRALinear` wrapper, `inject_lora()`, and `merge_lora()`. Wire into `ModelPlain.init_train()` (inject before freeze), extend `freeze_backbone()` to keep lora_A/B trainable, add `save_merged()` method, and trigger it at training end in `main_train_vrt.py`.

**Tech Stack:** PyTorch, pytest. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-04-17-lora-stage-c-design.md](../specs/2026-04-17-lora-stage-c-design.md)

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `models/lora.py` | Create | `LoRALinear`, `inject_lora`, `merge_lora` |
| `models/model_plain.py` | Modify | Extend `freeze_backbone` whitelist; call `inject_lora` in `init_train`; add `save_merged` method |
| `main_train_vrt.py` | Modify | Call `model.save_merged(current_step)` at training-end block |
| `tests/models/test_lora.py` | Create | Unit tests for `LoRALinear`, injection, merge |

---

## Task 1: `LoRALinear` wrapper

**Files:**
- Create: `models/lora.py`
- Create: `tests/models/test_lora.py`

- [ ] **Step 1: Write failing test for initial-output equivalence**

Create `tests/models/test_lora.py`:

```python
import math
import copy
import pytest
import torch
import torch.nn as nn

from models.lora import LoRALinear, inject_lora, merge_lora


def test_lora_initial_output_equals_base():
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    lora = LoRALinear(copy.deepcopy(base), rank=4, alpha=8)
    x = torch.randn(2, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `uv run pytest tests/models/test_lora.py::test_lora_initial_output_equals_base -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.lora'`

- [ ] **Step 3: Implement `LoRALinear`**

Create `models/lora.py`:

```python
"""LoRA low-rank adapters for nn.Linear layers.

Usage:
    inject_lora(model, target_substrings=["qkv", "proj"], rank=8, alpha=16)
    # train with base weights frozen
    merge_lora(model)  # in-place fold adapters back into nn.Linear
"""
import math
import copy
from typing import Iterable, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap nn.Linear: y = base(x) + (alpha/rank) * B(A(x)).

    Initialization: A Kaiming-uniform, B zeros — initial forward == base(x).
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling

    def merged_linear(self) -> nn.Linear:
        """Return a standalone nn.Linear with LoRA folded into the weight."""
        fused = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
        )
        delta = self.lora_B.weight @ self.lora_A.weight * self.scaling
        fused.weight.data = self.base.weight.data.detach().clone() + delta.detach()
        if self.base.bias is not None:
            fused.bias.data = self.base.bias.data.detach().clone()
        return fused


def inject_lora(
    model: nn.Module,
    target_substrings: Iterable[str],
    rank: int,
    alpha: float,
) -> List[str]:
    """Replace every `nn.Linear` whose *leaf* module name contains any of
    target_substrings with a `LoRALinear(m, rank, alpha)`.

    Returns dotted paths of replaced modules for logging.
    """
    targets = tuple(target_substrings)
    replaced: List[str] = []

    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if isinstance(child, LoRALinear):
                continue
            if not any(s in child_name for s in targets):
                continue
            wrapper = LoRALinear(child, rank=rank, alpha=alpha)
            wrapper = wrapper.to(child.weight.device, dtype=child.weight.dtype)
            setattr(parent, child_name, wrapper)
            dotted = f"{parent_name}.{child_name}" if parent_name else child_name
            replaced.append(dotted)

    return replaced


def merge_lora(model: nn.Module) -> nn.Module:
    """In-place: replace every LoRALinear in `model` with its merged nn.Linear."""
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                fused = child.merged_linear().to(
                    child.base.weight.device, dtype=child.base.weight.dtype
                )
                setattr(parent, child_name, fused)
    return model
```

- [ ] **Step 4: Run test — expect PASS**

Run: `uv run pytest tests/models/test_lora.py::test_lora_initial_output_equals_base -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/lora.py tests/models/test_lora.py
git commit -m "feat(lora): add LoRALinear wrapper with zero-init B"
```

---

## Task 2: `inject_lora` coverage

**Files:**
- Modify: `tests/models/test_lora.py`

- [ ] **Step 1: Write failing test — injection targets only matching Linears**

Append to `tests/models/test_lora.py`:

```python
class _MiniAttention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.qkv_self = nn.Linear(dim, dim * 3)
        self.qkv_mut = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.other_linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)


def test_inject_lora_hits_only_targets():
    m = _MiniAttention()
    replaced = inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    assert sorted(replaced) == ["proj", "qkv_mut", "qkv_self"]
    assert isinstance(m.qkv_self, LoRALinear)
    assert isinstance(m.qkv_mut, LoRALinear)
    assert isinstance(m.proj, LoRALinear)
    assert isinstance(m.other_linear, nn.Linear)
    assert not isinstance(m.other_linear, LoRALinear)
```

- [ ] **Step 2: Run test — expect PASS**

Run: `uv run pytest tests/models/test_lora.py::test_inject_lora_hits_only_targets -v`
Expected: PASS (`inject_lora` already implemented in Task 1)

- [ ] **Step 3: Write failing test — idempotent injection (double inject does not nest)**

Append:

```python
def test_inject_lora_is_idempotent():
    m = _MiniAttention()
    inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    replaced_second = inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    assert replaced_second == []
    assert isinstance(m.qkv_self, LoRALinear)
    assert isinstance(m.qkv_self.base, nn.Linear)
    assert not isinstance(m.qkv_self.base, LoRALinear)
```

- [ ] **Step 4: Run test — expect PASS**

Run: `uv run pytest tests/models/test_lora.py::test_inject_lora_is_idempotent -v`
Expected: PASS (the `isinstance(child, LoRALinear): continue` guard in `inject_lora` ensures this)

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_lora.py
git commit -m "test(lora): cover injection targeting and idempotence"
```

---

## Task 3: `merge_lora` preserves structure and forward

**Files:**
- Modify: `tests/models/test_lora.py`

- [ ] **Step 1: Write failing test — merge preserves state_dict shape**

Append to `tests/models/test_lora.py`:

```python
def test_merge_lora_state_dict_matches_original():
    torch.manual_seed(0)
    original = _MiniAttention()
    original_keys = set(original.state_dict().keys())
    original_shapes = {k: v.shape for k, v in original.state_dict().items()}

    wrapped = copy.deepcopy(original)
    inject_lora(wrapped, ["qkv", "proj"], rank=4, alpha=8)
    # Perturb LoRA weights so merge is non-trivial
    for mod in wrapped.modules():
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_A.weight, std=0.01)
            nn.init.normal_(mod.lora_B.weight, std=0.01)

    merge_lora(wrapped)
    merged_sd = wrapped.state_dict()
    assert set(merged_sd.keys()) == original_keys
    for k in original_keys:
        assert merged_sd[k].shape == original_shapes[k], k
```

- [ ] **Step 2: Run test — expect PASS**

Run: `uv run pytest tests/models/test_lora.py::test_merge_lora_state_dict_matches_original -v`
Expected: PASS

- [ ] **Step 3: Write failing test — merge forward equivalence**

Append:

```python
def test_merge_lora_forward_equivalence():
    torch.manual_seed(1)
    m = _MiniAttention()
    inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_A.weight, std=0.02)
            nn.init.normal_(mod.lora_B.weight, std=0.02)

    x = torch.randn(2, 8)
    with torch.no_grad():
        y_before = m.qkv_self(x)
    merged = copy.deepcopy(m)
    merge_lora(merged)
    with torch.no_grad():
        y_after = merged.qkv_self(x)
    assert torch.allclose(y_before, y_after, atol=1e-5), (y_before - y_after).abs().max()
```

- [ ] **Step 4: Run test — expect PASS**

Run: `uv run pytest tests/models/test_lora.py::test_merge_lora_forward_equivalence -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_lora.py
git commit -m "test(lora): cover merge structure and forward equivalence"
```

---

## Task 4: Extend `freeze_backbone` for LoRA

**Files:**
- Modify: `models/model_plain.py` (lines 18-24)
- Modify: `tests/models/test_lora.py`

- [ ] **Step 1: Write failing test — freeze leaves lora_A/B trainable**

Append to `tests/models/test_lora.py`:

```python
def test_freeze_backbone_keeps_lora_trainable():
    from models.model_plain import freeze_backbone

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _MiniAttention()
            # simulate fusion adapter on model root
            self.fusion_adapter = nn.Linear(8, 3)

    m = _Wrapper()
    inject_lora(m, ["qkv", "proj"], rank=4, alpha=8)
    freeze_backbone(m)

    # Base weights of LoRA-wrapped layers frozen
    assert m.attn.qkv_self.base.weight.requires_grad is False
    # LoRA adapters trainable
    assert m.attn.qkv_self.lora_A.weight.requires_grad is True
    assert m.attn.qkv_self.lora_B.weight.requires_grad is True
    # Fusion adapter stays trainable
    assert m.fusion_adapter.weight.requires_grad is True
    # Non-target Linear is frozen (backbone)
    assert m.attn.other_linear.weight.requires_grad is False
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `uv run pytest tests/models/test_lora.py::test_freeze_backbone_keeps_lora_trainable -v`
Expected: FAIL — current `freeze_backbone` does not whitelist lora names, so `lora_A.weight.requires_grad` is False.

- [ ] **Step 3: Extend `freeze_backbone` whitelist**

Edit [models/model_plain.py:18-24](../../../models/model_plain.py#L18-L24):

Replace:

```python
def freeze_backbone(model):
    """Freeze all backbone params and keep fusion-specific params trainable."""
    for name, param in model.named_parameters():
        if "fusion_adapter" in name or "fusion_operator" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
```

with:

```python
_TRAINABLE_NAME_MARKERS = ("fusion_adapter", "fusion_operator", "lora_A", "lora_B")


def freeze_backbone(model):
    """Freeze backbone params; keep fusion adapters and LoRA adapters trainable."""
    for name, param in model.named_parameters():
        if any(marker in name for marker in _TRAINABLE_NAME_MARKERS):
            param.requires_grad = True
        else:
            param.requires_grad = False
```

- [ ] **Step 4: Run test — expect PASS**

Run: `uv run pytest tests/models/test_lora.py::test_freeze_backbone_keeps_lora_trainable -v`
Expected: PASS

- [ ] **Step 5: Re-run existing freeze test to ensure no regression**

Run: `uv run pytest tests/models/test_partial_loading.py -v`
Expected: PASS (all existing tests)

- [ ] **Step 6: Commit**

```bash
git add models/model_plain.py tests/models/test_lora.py
git commit -m "feat(training): extend freeze_backbone whitelist for LoRA adapters"
```

---

## Task 5: Wire `inject_lora` into `ModelPlain.init_train`

**Files:**
- Modify: `models/model_plain.py` (imports near top; `init_train` around lines 169-187)
- Modify: `tests/models/test_lora.py`

- [ ] **Step 1: Write failing test — init_train injects LoRA when use_lora=true**

Append to `tests/models/test_lora.py`:

```python
def test_init_train_injects_lora(monkeypatch, tmp_path):
    """When train.use_lora=True, init_train should inject LoRA adapters into netG."""
    from models.model_plain import ModelPlain
    from models.architectures.vrt.attention import WindowAttention

    # Build a minimal ModelPlain stub that skips heavy init
    opt = {
        "train": {
            "use_lora": True,
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_target_modules": ["qkv", "proj"],
            "freeze_backbone": True,
            "E_decay": 0,
            "G_lossfn_type": "l1",
            "G_lossfn_weight": 1.0,
            "G_optimizer_type": "adam",
            "G_optimizer_lr": 1e-4,
            "G_optimizer_betas": [0.9, 0.99],
            "G_optimizer_wd": 0,
            "G_optimizer_reuse": False,
            "G_optimizer_clipgrad": None,
            "G_scheduler_type": "CosineAnnealingWarmRestarts",
            "G_scheduler_periods": 100,
            "G_scheduler_restart_weights": 1,
            "G_scheduler_eta_min": 1e-7,
            "G_regularizer_orthstep": None,
            "G_regularizer_clipstep": None,
            "G_param_strict": False,
        },
        "path": {"pretrained_netG": None, "pretrained_netE": None,
                 "pretrained_optimizerG": None},
        "rank": 0,
        "dist": False,
    }

    model = ModelPlain.__new__(ModelPlain)
    model.opt = opt
    model.opt_train = opt["train"]
    model.device = torch.device("cpu")
    model.schedulers = []
    model.netG = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)

    class _DummyTimer:
        def timer(self, *a, **k):
            from contextlib import nullcontext
            return nullcontext()
    model.timer = _DummyTimer()

    # Stub out calls init_train makes that we don't want to exercise here
    monkeypatch.setattr(model, "load", lambda: None)
    monkeypatch.setattr(model, "load_optimizers", lambda: None)
    monkeypatch.setattr(model, "define_loss", lambda: None)
    monkeypatch.setattr(model, "define_scheduler", lambda: None)
    monkeypatch.setattr(model, "get_bare_model", lambda net: net)

    model.init_train()

    assert isinstance(model.netG.qkv_self, LoRALinear)
    assert isinstance(model.netG.proj, LoRALinear)
    # Base weights frozen, LoRA trainable
    assert model.netG.qkv_self.base.weight.requires_grad is False
    assert model.netG.qkv_self.lora_A.weight.requires_grad is True
```

- [ ] **Step 2: Run test — expect FAIL (no injection in init_train yet)**

Run: `uv run pytest tests/models/test_lora.py::test_init_train_injects_lora -v`
Expected: FAIL — `model.netG.qkv_self` is still a plain `nn.Linear`.

- [ ] **Step 3: Add injection to `init_train`**

Edit [models/model_plain.py](../../../models/model_plain.py) `init_train` method (around line 169):

Replace:

```python
    def init_train(self):
        self.load()                           # load model
        if self.opt.get('train', {}).get('freeze_backbone', False):
            bare_model = self.get_bare_model(self.netG)
            freeze_backbone(bare_model)
            frozen_count = sum(1 for p in bare_model.parameters() if not p.requires_grad)
            trainable_count = sum(1 for p in bare_model.parameters() if p.requires_grad)
            print(f'[Stage A] Frozen {frozen_count} params, trainable {trainable_count} params')
```

with:

```python
    def init_train(self):
        self.load()                           # load model
        train_opt = self.opt.get('train', {})
        bare_model = self.get_bare_model(self.netG)

        if train_opt.get('use_lora', False):
            from models.lora import inject_lora
            targets = train_opt.get('lora_target_modules', ['qkv', 'proj'])
            rank = int(train_opt.get('lora_rank', 8))
            alpha = float(train_opt.get('lora_alpha', 16))
            replaced = inject_lora(bare_model, targets, rank=rank, alpha=alpha)
            print(f'[Stage C] Injected LoRA(rank={rank}, alpha={alpha}) into '
                  f'{len(replaced)} Linear layers; targets={targets}')
            # Mirror into netE so EMA has matching structure for load/save
            if train_opt.get('E_decay', 0) > 0 and hasattr(self, 'netE'):
                bare_e = self.get_bare_model(self.netE)
                inject_lora(bare_e, targets, rank=rank, alpha=alpha)

        if train_opt.get('freeze_backbone', False):
            freeze_backbone(bare_model)
            frozen_count = sum(1 for p in bare_model.parameters() if not p.requires_grad)
            trainable_count = sum(1 for p in bare_model.parameters() if p.requires_grad)
            print(f'[Stage A/C] Frozen {frozen_count} params, trainable {trainable_count} params')
```

- [ ] **Step 4: Run test — expect PASS**

Run: `uv run pytest tests/models/test_lora.py::test_init_train_injects_lora -v`
Expected: PASS

- [ ] **Step 5: Re-run all lora + existing freeze tests**

Run: `uv run pytest tests/models/test_lora.py tests/models/test_partial_loading.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add models/model_plain.py tests/models/test_lora.py
git commit -m "feat(training): inject LoRA in ModelPlain.init_train on use_lora=True"
```

---

## Task 6: `ModelPlain.save_merged` emits fused ckpt

**Files:**
- Modify: `models/model_plain.py` (add method after `save`)
- Modify: `tests/models/test_lora.py`

- [ ] **Step 1: Write failing test — save_merged writes merged ckpt**

Append to `tests/models/test_lora.py`:

```python
def test_save_merged_writes_fused_ckpt(tmp_path):
    from models.model_plain import ModelPlain
    from models.architectures.vrt.attention import WindowAttention

    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"train": {"use_lora": True, "E_decay": 0}, "rank": 0, "dist": False}
    model.opt_train = model.opt["train"]
    model.save_dir = str(tmp_path)
    net = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)
    inject_lora(net, ["qkv", "proj"], rank=4, alpha=8)
    # Perturb LoRA so merge is non-trivial
    for mod in net.modules():
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_A.weight, std=0.02)
            nn.init.normal_(mod.lora_B.weight, std=0.02)
    model.netG = net

    # Reference structure without LoRA
    ref = WindowAttention(dim=8, window_size=(2, 2, 2), num_heads=2, mut_attn=True)
    ref_keys = set(ref.state_dict().keys())

    model.save_merged(iter_label=12345)

    merged_path = tmp_path / "12345_G_merged.pth"
    assert merged_path.exists()
    sd = torch.load(merged_path, map_location="cpu", weights_only=True)
    assert set(sd.keys()) == ref_keys
    # netG in memory untouched (still has LoRA)
    assert isinstance(model.netG.qkv_self, LoRALinear)


def test_save_merged_noop_without_lora(tmp_path):
    from models.model_plain import ModelPlain

    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"train": {"use_lora": False, "E_decay": 0}, "rank": 0, "dist": False}
    model.opt_train = model.opt["train"]
    model.save_dir = str(tmp_path)
    model.netG = nn.Linear(4, 4)

    model.save_merged(iter_label=1)

    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run test — expect FAIL (`save_merged` undefined)**

Run: `uv run pytest tests/models/test_lora.py::test_save_merged_writes_fused_ckpt tests/models/test_lora.py::test_save_merged_noop_without_lora -v`
Expected: FAIL — `AttributeError: 'ModelPlain' object has no attribute 'save_merged'`

- [ ] **Step 3: Implement `save_merged`**

Edit [models/model_plain.py](../../../models/model_plain.py) — add after the existing `save` method (around line 229):

```python
    def save_merged(self, iter_label):
        """Export LoRA-merged checkpoint (`{iter}_G_merged.pth` / `{iter}_E_merged.pth`).

        Produces state_dicts that are structurally identical to the non-LoRA VRT,
        so non-LoRA code paths can load them directly.
        Rank-0 only under DDP; no-op when use_lora is disabled.
        """
        import os
        import copy as _copy
        if not self.opt.get('train', {}).get('use_lora', False):
            return
        if self.opt.get('rank', 0) != 0:
            return
        from models.lora import merge_lora

        pairs = [(self.netG, 'G')]
        if self.opt_train.get('E_decay', 0) > 0 and hasattr(self, 'netE'):
            pairs.append((self.netE, 'E'))

        for net, tag in pairs:
            bare = self.get_bare_model(net)
            net_copy = _copy.deepcopy(bare)
            merge_lora(net_copy)
            state_dict = {k: v.detach().cpu() for k, v in net_copy.state_dict().items()}
            save_path = os.path.join(self.save_dir, f'{iter_label}_{tag}_merged.pth')
            tmp_path = save_path + '.tmp'
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, save_path)
            print(f'[Stage C] Saved merged ckpt -> {save_path}')
```

- [ ] **Step 4: Run test — expect PASS**

Run: `uv run pytest tests/models/test_lora.py::test_save_merged_writes_fused_ckpt tests/models/test_lora.py::test_save_merged_noop_without_lora -v`
Expected: PASS

- [ ] **Step 5: Full test module run**

Run: `uv run pytest tests/models/test_lora.py -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add models/model_plain.py tests/models/test_lora.py
git commit -m "feat(training): add save_merged to export LoRA-folded checkpoint"
```

---

## Task 7: Trigger `save_merged` at end of training

**Files:**
- Modify: `main_train_vrt.py` (around line 715-721)

- [ ] **Step 1: Read current end-of-training block**

File: [main_train_vrt.py:714-721](../../../main_train_vrt.py#L714-L721) currently reads:

```python
            if current_step > opt['train']['total_iter']:
                if opt['rank'] == 0:
                    logger.info('Finish training.')
                    model.save(current_step)  # 保存最终模型
                    if tb_logger is not None:
                        tb_logger.close()  # 关闭日志记录器
                sys.exit()  # 退出程序
```

- [ ] **Step 2: Add merged-ckpt export**

Replace that block with:

```python
            if current_step > opt['train']['total_iter']:
                if opt['rank'] == 0:
                    logger.info('Finish training.')
                    model.save(current_step)  # 保存最终模型
                    if hasattr(model, 'save_merged'):
                        model.save_merged(current_step)
                    if tb_logger is not None:
                        tb_logger.close()  # 关闭日志记录器
                sys.exit()  # 退出程序
```

- [ ] **Step 3: Smoke-verify the file parses**

Run: `uv run python -c "import ast; ast.parse(open('main_train_vrt.py').read())"`
Expected: No output (no syntax errors)

- [ ] **Step 4: Commit**

```bash
git add main_train_vrt.py
git commit -m "feat(training): export merged LoRA ckpt at end of training"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run the full new test module**

Run: `uv run pytest tests/models/test_lora.py -v`
Expected: All 8 tests pass (initial equivalence, injection targeting, idempotence, state_dict preservation, forward equivalence, freeze interaction, init_train injection, save_merged write, save_merged no-op).

- [ ] **Step 2: Run adjacent existing tests for regressions**

Run: `uv run pytest tests/models/test_partial_loading.py tests/models/test_fusion_early_adapter.py -v`
Expected: All pass (freeze_backbone whitelist change did not regress Stage A).

- [ ] **Step 3: Config sanity check**

Run: `uv run python -c "import json, re; [json.loads(re.sub(r'//[^\n]*','',open(p).read())) for p in ['options/gopro_rgbspike_local.json','options/gopro_rgbspike_server.json']]; print('OK')"`
Expected: `OK`

- [ ] **Step 4: No-op behavior when use_lora=false**

Confirm that with `use_lora: false` (current default in both configs) the new code path is inert: inject not called, freeze_backbone whitelist's new markers never match any param name, save_merged returns early. This is already covered by `test_save_merged_noop_without_lora` and the conditional in `init_train`.

---

## Notes for the engineer

- **Stage C run recipe:** set `train.use_lora: true` in the config, set `path.pretrained_netG` to a Stage A checkpoint, set `train.partial_load: true` (already in config). No code changes needed per run — everything is config-driven.
- **LoRA init determinism:** `lora_B` is zero-initialized, so initial forward ≡ base forward. This means Stage C effectively starts from Stage A performance (fusion adapter is inherited via partial_load, base weights are the Stage A-trained VRT).
- **Merged ckpt consumption:** the merged file has the same keys as a stock VRT state_dict; pass it to `pretrained_netG` with `train.use_lora: false` and `train.partial_load: false` / `G_param_strict: true` to load strictly.
- **Do NOT add MLP/DCN/flow to lora targets** unless the spec is updated — increasing target scope changes the parameter budget story and requires re-validation.
