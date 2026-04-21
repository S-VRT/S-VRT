# Fusion Loss 重设计与 GatedFusionOperator 修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 GatedFusionOperator 的初始化破坏问题，并实现 Phase 感知的双 loss 策略，使 Phase 1 直接监督 fusion 输出对 GT，Phase 2 恢复 VRT 重建 loss。

**Architecture:** 将 GatedFusionOperator 改为加法残差结构（`out = rgb + gate * correction`），在 VRT forward 中 hook fusion 输出，在 ModelPlain.optimize_parameters 中按 `current_step < fix_iter` 切换 loss 权重。

**Tech Stack:** PyTorch, existing S-VRT model infrastructure (ModelPlain, ModelVRT, VRT)

---

### Task 1: 修改 GatedFusionOperator 为加法残差结构

**Files:**
- Modify: `models/fusion/operators/gated.py`
- Test: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: 在 test_fusion_early_adapter.py 末尾添加新测试**

```python
def test_gated_operator_passthrough_at_init():
    """At init, gate≈0 and correction≈0, so output should be close to rgb input."""
    op = create_fusion_operator("gated", 3, 1, 3, {})
    rgb = torch.ones(1, 1, 3, 8, 8) * 0.5
    spike = torch.zeros(1, 1, 1, 8, 8)
    with torch.no_grad():
        out = op(rgb, spike)
    # output should be very close to rgb (within 0.05 absolute)
    assert torch.allclose(out, rgb, atol=0.05), \
        f"Expected near-passthrough at init, max diff={( out - rgb).abs().max():.4f}"


def test_gated_operator_no_rgb_proj():
    """GatedFusionOperator must not have an rgb_proj attribute."""
    op = create_fusion_operator("gated", 3, 1, 3, {})
    assert not hasattr(op, 'rgb_proj'), "rgb_proj should be removed from GatedFusionOperator"


def test_gated_operator_has_correction():
    """GatedFusionOperator must have a correction attribute (renamed from fuse)."""
    op = create_fusion_operator("gated", 3, 1, 3, {})
    assert hasattr(op, 'correction'), "GatedFusionOperator must have 'correction' attribute"
```

- [ ] **Step 2: 运行新测试，确认失败**

```bash
cd /home/wuhy/projects/S-VRT
python -m pytest tests/models/test_fusion_early_adapter.py::test_gated_operator_passthrough_at_init tests/models/test_fusion_early_adapter.py::test_gated_operator_no_rgb_proj tests/models/test_fusion_early_adapter.py::test_gated_operator_has_correction -v
```

期望：3 个测试 FAIL（`rgb_proj` 存在，`correction` 不存在，passthrough 不成立）

- [ ] **Step 3: 重写 gated.py**

将 `models/fusion/operators/gated.py` 完整替换为：

```python
from typing import Dict

import torch
from torch import nn


class GatedFusionOperator(nn.Module):
    """Additive gated fusion: out = rgb + gate(concat) * correction(concat).

    Base path is identity (no rgb_proj), so colors are preserved by design.
    Both correction and gate are zero-initialized so the operator starts as
    a pure passthrough and gradually learns to incorporate spike information.
    """

    def __init__(
        self,
        rgb_chans: int,
        spike_chans: int,
        out_chans: int,
        operator_params: Dict,
    ):
        super().__init__()
        self.rgb_chans = rgb_chans
        self.spike_chans = spike_chans
        self.out_chans = out_chans
        self.operator_params = operator_params
        hidden_chans = int(operator_params.get('hidden_chans', 32))
        in_chans = rgb_chans + spike_chans

        self.correction = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_chans, out_chans, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_chans, out_chans, kernel_size=1),
            nn.Sigmoid(),
        )

        # Zero-init correction output → starts at 0
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        # Gate pre-sigmoid bias = -5 → Sigmoid(-5) ≈ 0.007, near-zero gate at init
        nn.init.constant_(self.gate[-2].bias, -5.0)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != spike_feat.dim():
            raise ValueError('rgb and spike must have the same number of dimensions')
        if rgb_feat.dim() == 5:
            bsz, steps, rgb_chans, height, width = rgb_feat.shape
            spike_bsz, spike_steps, spike_chans, spike_height, spike_width = spike_feat.shape
            if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_height, spike_width):
                raise ValueError('rgb and spike must share batch, time, height, and width dimensions')
            if rgb_chans != self.rgb_chans:
                raise ValueError(f'Expected rgb channels={self.rgb_chans}, got {rgb_chans}')
            if spike_chans != self.spike_chans:
                raise ValueError(f'Expected spike channels={self.spike_chans}, got {spike_chans}')
            rgb_flat = rgb_feat.reshape(bsz * steps, rgb_chans, height, width)
            spike_flat = spike_feat.reshape(bsz * steps, spike_chans, height, width)
            concat = torch.cat([rgb_flat, spike_flat], dim=1)
            out = rgb_flat + self.gate(concat) * self.correction(concat)
            return out.reshape(bsz, steps, self.out_chans, height, width)
        if rgb_feat.dim() == 4:
            bsz, rgb_chans, height, width = rgb_feat.shape
            spike_bsz, spike_chans, spike_height, spike_width = spike_feat.shape
            if (bsz, height, width) != (spike_bsz, spike_height, spike_width):
                raise ValueError('rgb and spike must share batch, height, and width dimensions')
            if rgb_chans != self.rgb_chans:
                raise ValueError(f'Expected rgb channels={self.rgb_chans}, got {rgb_chans}')
            if spike_chans != self.spike_chans:
                raise ValueError(f'Expected spike channels={self.spike_chans}, got {spike_chans}')
            concat = torch.cat([rgb_feat, spike_feat], dim=1)
            return rgb_feat + self.gate(concat) * self.correction(concat)
        raise ValueError('Expected rgb and spike features with 4 or 5 dimensions')


__all__ = ['GatedFusionOperator']
```

- [ ] **Step 4: 运行所有 gated 相关测试**

```bash
python -m pytest tests/models/test_fusion_early_adapter.py -v -k "gated"
```

期望：全部 PASS（包括 shape、passthrough、no_rgb_proj、has_correction）

- [ ] **Step 5: 运行完整 fusion 测试套件**

```bash
python -m pytest tests/models/test_fusion_early_adapter.py tests/models/test_fusion_factory.py tests/models/test_fusion_middle_adapter.py -v
```

期望：全部 PASS

- [ ] **Step 6: Commit**

```bash
git add models/fusion/operators/gated.py tests/models/test_fusion_early_adapter.py
git commit -m "feat(fusion): replace rgb_proj with additive correction in GatedFusionOperator

- Remove rgb_proj (kaiming init was destroying color channels)
- Rename fuse -> correction for semantic clarity
- out = rgb + gate * correction (identity base path by design)
- Zero-init correction last layer; gate pre-sigmoid bias = -5 (near-zero at init)"
```

---

### Task 2: VRT forward 中 hook fusion 输出

**Files:**
- Modify: `models/architectures/vrt/vrt.py:522-526`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: 在 test_vrt_fusion_integration.py 末尾添加测试**

先查看该文件末尾：
```bash
tail -20 tests/models/test_vrt_fusion_integration.py
```

然后添加：

```python
def test_vrt_stores_fusion_hook_after_forward():
    """VRT must store _last_fusion_out and _last_spike_bins after forward with early fusion."""
    import torch
    from models.architectures.vrt.vrt import VRT

    model = VRT(
        upscale=1,
        in_chans=7,
        img_size=[6, 16, 16],
        window_size=[6, 8, 8],
        depths=[2, 2],
        indep_reconsts=[1],
        embed_dims=[16, 16],
        num_heads=[2, 2],
        fusion={
            'enable': True,
            'placement': 'early',
            'operator': 'gated',
            'out_chans': 3,
            'operator_params': {},
        },
        input={'strategy': 'fusion', 'mode': 'dual', 'raw_ingress_chans': 7},
        output_mode='restoration',
    ).eval()

    # [B, N, 3+4, H, W] — 3 rgb + 4 spike bins
    x = torch.randn(1, 6, 7, 16, 16)
    with torch.no_grad():
        _ = model(x)

    assert hasattr(model, '_last_fusion_out'), "_last_fusion_out not set after forward"
    assert hasattr(model, '_last_spike_bins'), "_last_spike_bins not set after forward"
    assert model._last_spike_bins == 4
    # fusion output: [B, N*S, 3, H, W] = [1, 24, 3, 16, 16]
    assert model._last_fusion_out.shape == (1, 24, 3, 16, 16)
```

- [ ] **Step 2: 运行新测试，确认失败**

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py::test_vrt_stores_fusion_hook_after_forward -v
```

期望：FAIL（`_last_fusion_out` 不存在）

- [ ] **Step 3: 修改 vrt.py，在 fusion 调用后存储 hook**

在 `models/architectures/vrt/vrt.py` 第 526 行附近，将：

```python
                x = self.fusion_adapter(rgb=rgb, spike=spike)
```

替换为：

```python
                x = self.fusion_adapter(rgb=rgb, spike=spike)
                self._last_fusion_out = x
                self._last_spike_bins = spike_bins
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/models/test_vrt_fusion_integration.py -v
```

期望：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add models/architectures/vrt/vrt.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat(vrt): store _last_fusion_out and _last_spike_bins after early fusion"
```

---

### Task 3: ModelPlain 中添加 _compute_fusion_aux_loss 和 Phase 感知 loss

**Files:**
- Modify: `models/model_plain.py`
- Test: `tests/models/test_model_plain_fusion_aux_loss.py` (新建)

- [ ] **Step 1: 新建测试文件**

创建 `tests/models/test_model_plain_fusion_aux_loss.py`：

```python
"""Tests for ModelPlain._compute_fusion_aux_loss and phase-aware loss weighting."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


def _make_model_with_fusion_out(fusion_out, spike_bins, phase1_aux=1.0, phase2_aux=0.2, passthrough=0.2):
    """Helper: build a minimal ModelPlain-like object with mocked internals."""
    from models.model_plain import ModelPlain

    opt = {
        'scale': 1,
        'n_channels': 3,
        'netG': {
            'net_type': 'vrt',
            'input': {'strategy': 'fusion', 'mode': 'dual', 'raw_ingress_chans': 7},
            'fusion': {'enable': True, 'placement': 'early', 'operator': 'gated', 'out_chans': 3, 'operator_params': {}},
            'in_chans': 7, 'upscale': 1, 'img_size': [6, 8, 8], 'window_size': [6, 8, 8],
            'depths': [2], 'indep_reconsts': [], 'embed_dims': [16], 'num_heads': [2],
            'output_mode': 'restoration',
            'restoration_reducer': {'type': 'index', 'index': 2},
        },
        'train': {
            'G_lossfn_type': 'charbonnier',
            'G_lossfn_weight': 1.0,
            'G_charbonnier_eps': 1e-6,
            'phase1_fusion_aux_loss_weight': phase1_aux,
            'phase2_fusion_aux_loss_weight': phase2_aux,
            'fusion_passthrough_loss_weight': passthrough,
            'G_optimizer_type': 'adam',
            'G_optimizer_lr': 1e-4,
            'G_optimizer_betas': [0.9, 0.99],
            'G_optimizer_wd': 0,
            'G_optimizer_clipgrad': None,
            'G_optimizer_reuse': False,
            'G_scheduler_type': 'MultiStepLR',
            'G_scheduler_milestones': [100],
            'G_scheduler_gamma': 0.5,
            'G_regularizer_orthstep': None,
            'G_regularizer_clipstep': None,
            'G_param_strict': False,
            'E_param_strict': False,
            'E_decay': 0,
            'manual_seed': 0,
            'fix_iter': 10,
            'fix_keys': [],
            'checkpoint_save': 100,
            'checkpoint_test': 100,
            'checkpoint_print': 10,
            'amp': {'enable': False},
        },
        'path': {'root': '/tmp', 'pretrained_netG': None, 'pretrained_netE': None},
        'rank': 0,
        'dist': False,
    }

    model = ModelPlain(opt)
    model.define_loss()

    # Inject mock fusion hook
    bare = model.get_bare_model(model.netG)
    bare._last_fusion_out = fusion_out
    bare._last_spike_bins = spike_bins

    # Set self.H (GT) and self.L (concat input)
    B, N, S = fusion_out.shape[0], fusion_out.shape[1] // spike_bins, spike_bins
    model.H = torch.zeros(B, N, 3, fusion_out.shape[-2], fusion_out.shape[-1])
    # self.L: [B, N, 3+spike_chans, H, W] — first 3 channels are blur_rgb
    model.L = torch.ones(B, N, 7, fusion_out.shape[-2], fusion_out.shape[-1])
    return model


def test_fusion_aux_loss_phase1_returns_nonzero():
    fusion_out = torch.randn(1, 24, 3, 8, 8)  # N=6, S=4
    model = _make_model_with_fusion_out(fusion_out, spike_bins=4)
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    assert loss.item() > 0.0


def test_fusion_aux_loss_phase2_uses_phase2_weight():
    fusion_out = torch.randn(1, 24, 3, 8, 8)
    model = _make_model_with_fusion_out(fusion_out, spike_bins=4, phase1_aux=1.0, phase2_aux=0.2)
    loss_p1 = model._compute_fusion_aux_loss(is_phase1=True)
    loss_p2 = model._compute_fusion_aux_loss(is_phase1=False)
    # Phase 2 weight (0.2) < Phase 1 weight (1.0), so loss_p2 < loss_p1
    assert loss_p2.item() < loss_p1.item()


def test_fusion_aux_loss_zero_when_weights_zero():
    fusion_out = torch.randn(1, 24, 3, 8, 8)
    model = _make_model_with_fusion_out(fusion_out, spike_bins=4, phase1_aux=0.0, phase2_aux=0.0, passthrough=0.0)
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    assert loss.item() == 0.0


def test_fusion_aux_loss_no_hook_returns_zero():
    fusion_out = torch.randn(1, 24, 3, 8, 8)
    model = _make_model_with_fusion_out(fusion_out, spike_bins=4)
    # Remove the hook
    bare = model.get_bare_model(model.netG)
    del bare._last_fusion_out
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    assert loss.item() == 0.0


def test_fusion_aux_loss_center_frame_shape():
    """fusion_center must be [B, N, 3, H, W] — S//2::S indexing."""
    fusion_out = torch.zeros(1, 24, 3, 8, 8)  # N=6, S=4
    # Set center frames (index 2, 6, 10, 14, 18, 22) to 1.0
    fusion_out[:, 2::4, :, :, :] = 1.0
    model = _make_model_with_fusion_out(fusion_out, spike_bins=4)
    model.H = torch.ones(1, 6, 3, 8, 8)  # GT = 1.0
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    # fusion_center = 1.0, GT = 1.0 → loss should be ~0
    assert loss.item() < 1e-4
```

- [ ] **Step 2: 运行新测试，确认失败**

```bash
python -m pytest tests/models/test_model_plain_fusion_aux_loss.py -v
```

期望：FAIL（`_compute_fusion_aux_loss` 方法不存在）

- [ ] **Step 3: 在 model_plain.py 中添加 _compute_fusion_aux_loss 方法**

在 `models/model_plain.py` 的 `optimize_parameters` 方法之前（约第 400 行），插入：

```python
    def _compute_fusion_aux_loss(self, is_phase1: bool) -> torch.Tensor:
        if is_phase1:
            aux_weight = self.opt_train.get('phase1_fusion_aux_loss_weight', 0.0)
            pass_weight = self.opt_train.get('fusion_passthrough_loss_weight', 0.0)
        else:
            aux_weight = self.opt_train.get('phase2_fusion_aux_loss_weight', 0.0)
            pass_weight = 0.0
        if aux_weight == 0.0 and pass_weight == 0.0:
            return torch.tensor(0.0, device=self.device)
        vrt = self.get_bare_model(self.netG)
        if not hasattr(vrt, '_last_fusion_out'):
            return torch.tensor(0.0, device=self.device)
        fusion_out = vrt._last_fusion_out
        S = vrt._last_spike_bins
        fusion_center = fusion_out[:, S // 2 :: S, :, :, :]  # [B, N, 3, H, W]
        loss = torch.tensor(0.0, device=self.device)
        if aux_weight > 0.0:
            loss = loss + aux_weight * self.G_lossfn(fusion_center, self.H)
        if pass_weight > 0.0:
            blur_rgb = self.L[:, :, :3, :, :]
            loss = loss + pass_weight * self.G_lossfn(fusion_center, blur_rgb)
        return loss
```

- [ ] **Step 4: 修改 optimize_parameters 添加 Phase 感知 loss**

在 `models/model_plain.py` 的 `optimize_parameters` 方法中，将：

```python
        with self.timer.timer('loss_compute'):
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
```

替换为：

```python
        with self.timer.timer('loss_compute'):
            is_phase1 = (
                hasattr(self, 'fix_iter') and
                self.fix_iter > 0 and
                current_step < self.fix_iter
            )
            vrt_loss_weight = 0.0 if is_phase1 else self.G_lossfn_weight
            G_loss = vrt_loss_weight * self.G_lossfn(self.E, self.H)
            G_loss = G_loss + self._compute_fusion_aux_loss(is_phase1=is_phase1)
```

注意：`ModelPlain` 本身没有 `fix_iter` 属性，`ModelVRT` 才有。`hasattr` 检查保证在 `ModelPlain` 直接使用时不报错（此时 `is_phase1=False`，行为与原来相同）。

- [ ] **Step 5: 运行测试**

```bash
python -m pytest tests/models/test_model_plain_fusion_aux_loss.py -v
```

期望：全部 PASS

- [ ] **Step 6: 运行回归测试**

```bash
python -m pytest tests/models/ -v --timeout=60
```

期望：全部 PASS（无回归）

- [ ] **Step 7: Commit**

```bash
git add models/model_plain.py tests/models/test_model_plain_fusion_aux_loss.py
git commit -m "feat(model): add phase-aware fusion aux loss to ModelPlain

- _compute_fusion_aux_loss(is_phase1): reads _last_fusion_out hook from VRT
- Phase 1: aux_weight from phase1_fusion_aux_loss_weight, passthrough from fusion_passthrough_loss_weight
- Phase 2: aux_weight from phase2_fusion_aux_loss_weight, no passthrough
- optimize_parameters: vrt_loss_weight=0 in Phase 1, full weight in Phase 2"
```

---

### Task 4: checkpoint_test 列表化支持

**Files:**
- Modify: `main_train_vrt.py:487`
- Test: inline verification

- [ ] **Step 1: 修改 main_train_vrt.py 中的 checkpoint_test 检查**

在 `main_train_vrt.py` 第 487 行，将：

```python
            if current_step % opt['train']['checkpoint_test'] == 0:
```

替换为：

```python
            _ckpt_test = opt['train']['checkpoint_test']
            _ckpt_test_steps = (
                _ckpt_test if isinstance(_ckpt_test, list) else [_ckpt_test]
            )
            if current_step in _ckpt_test_steps or (
                not isinstance(_ckpt_test, list) and current_step % _ckpt_test == 0
            ):
```

- [ ] **Step 2: 验证语法正确**

```bash
python -c "import main_train_vrt; print('OK')"
```

期望：`OK`（无语法错误）

- [ ] **Step 3: Commit**

```bash
git add main_train_vrt.py
git commit -m "feat(train): support checkpoint_test as list of steps (e.g. [6000, 30000])"
```

---

### Task 5: 更新 gopro_rgbspike_server.json

**Files:**
- Modify: `options/gopro_rgbspike_server.json`

- [ ] **Step 1: 修改 fix_iter**

将 `"fix_iter": 20000` 改为 `"fix_iter": 6000`

- [ ] **Step 2: 添加 fusion loss 权重字段**

在 `"G_lossfn_weight": 1.0` 之后添加：

```json
,
"phase1_fusion_aux_loss_weight": 1.0
,
"phase2_fusion_aux_loss_weight": 0.2
,
"fusion_passthrough_loss_weight": 0.2
```

- [ ] **Step 3: 更新 checkpoint_test**

将 `"checkpoint_test": 30000` 改为 `"checkpoint_test": [6000, 30000]`

- [ ] **Step 4: 验证 JSON 合法**

```bash
python -c "import json; json.load(open('options/gopro_rgbspike_server.json', encoding='utf-8')); print('JSON OK')"
```

注意：该 JSON 文件含注释（`//`），需要用支持注释的解析器。如果上述命令报错，改用：

```bash
python -c "
import re, json
raw = open('options/gopro_rgbspike_server.json').read()
stripped = re.sub(r'//[^\n]*', '', raw)
json.loads(stripped)
print('JSON OK')
"
```

期望：`JSON OK`

- [ ] **Step 5: Commit**

```bash
git add options/gopro_rgbspike_server.json
git commit -m "config: update server config for fusion loss redesign

- fix_iter: 20000 -> 6000 (Phase 1 fusion warmup, Phase 2 gets 24k iters)
- add phase1/phase2_fusion_aux_loss_weight and fusion_passthrough_loss_weight
- checkpoint_test: [6000, 30000] to validate fusion quality at Phase 1 end"
```

---

### Task 6: 端到端冒烟测试

**Files:**
- Test: `tests/e2e/` (已有)

- [ ] **Step 1: 运行现有 e2e 测试**

```bash
python -m pytest tests/e2e/ -v --timeout=120 -x
```

期望：全部 PASS（无回归）

- [ ] **Step 2: 快速 forward 冒烟测试（验证 hook + loss 联通）**

```bash
python -c "
import torch
from models.fusion.operators.gated import GatedFusionOperator

# 验证 passthrough 初始化
op = GatedFusionOperator(3, 1, 3, {})
rgb = torch.ones(1, 1, 3, 8, 8) * 0.5
spike = torch.zeros(1, 1, 1, 8, 8)
with torch.no_grad():
    out = op(rgb, spike)
diff = (out - rgb).abs().max().item()
assert diff < 0.05, f'passthrough failed: max_diff={diff}'
print(f'GatedFusionOperator passthrough OK (max_diff={diff:.6f})')
"
```

期望：`GatedFusionOperator passthrough OK (max_diff=0.000000)`（或极小值）

- [ ] **Step 3: 最终 commit（如有未提交内容）**

```bash
git status
# 确认无未提交文件
```

---

## Self-Review

**Spec coverage:**
- ✅ RC1/RC2 修复：Task 1（去 rgb_proj，additive correction，零初始化）
- ✅ RC3 修复：Task 3（Phase 1 关闭 VRT loss，aux loss 直接监督 fusion）
- ✅ Fusion hook：Task 2（`_last_fusion_out`，`_last_spike_bins`）
- ✅ `_compute_fusion_aux_loss`：Task 3
- ✅ Phase 感知 loss 权重：Task 3
- ✅ `checkpoint_test` 列表化：Task 4
- ✅ Config 更新：Task 5（`fix_iter=6000`，三个新 loss 字段，`checkpoint_test=[6000,30000]`）
- ✅ 端到端验证：Task 6

**Placeholder 扫描：** 无 TBD/