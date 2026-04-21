# Fusion Loss 重设计与 GatedFusionOperator 修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 GatedFusionOperator 的初始化破坏问题，并实现 Phase 感知的双 loss 策略，使 Phase 1 直接监督 fusion 输出对 GT，Phase 2 恢复 VRT 重建 loss。

**Architecture:** 将 GatedFusionOperator 改为加法残差结构（`out = rgb + gate * correction`），在 VRT forward 中 hook fusion 输出到 `self._last_fusion_out`，在 `ModelPlain.optimize_parameters` 中按 `current_step < fix_iter` 切换 loss 权重。`fix_iter` 属性在 `ModelVRT.__init__` 中设置，`ModelPlain` 通过 `hasattr` 检查安全访问。

**Tech Stack:** PyTorch, existing S-VRT model infrastructure (ModelPlain, ModelVRT, VRT)

---

### Task 1: 修改 GatedFusionOperator 为加法残差结构

**Files:**
- Modify: `models/fusion/operators/gated.py`
- Modify: `tests/models/test_fusion_early_adapter.py`

- [ ] **Step 1: 在 `tests/models/test_fusion_early_adapter.py` 末尾追加三个新测试**

```python
def test_gated_operator_passthrough_at_init():
    """At init, gate≈0 and correction≈0, so output should equal rgb input."""
    op = create_fusion_operator("gated", 3, 1, 3, {})
    rgb = torch.ones(1, 1, 3, 8, 8) * 0.5
    spike = torch.zeros(1, 1, 1, 8, 8)
    with torch.no_grad():
        out = op(rgb, spike)
    assert torch.allclose(out, rgb, atol=0.05), (
        f"Expected near-passthrough at init, max diff={(out - rgb).abs().max():.4f}"
    )


def test_gated_operator_no_rgb_proj():
    """GatedFusionOperator must not have an rgb_proj attribute after redesign."""
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

期望：3 个测试 FAIL

- [ ] **Step 3: 完整替换 `models/fusion/operators/gated.py`**

```python
from typing import Dict

import torch
from torch import nn


class GatedFusionOperator(nn.Module):
    """Additive gated fusion: out = rgb + gate(concat) * correction(concat).

    Base path is identity (no rgb_proj), so colors are preserved by design.
    correction last layer and gate pre-sigmoid bias are zero-initialized so
    the operator starts as a pure passthrough.
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

        # correction output starts at 0
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        # gate pre-sigmoid bias = -5 → Sigmoid(-5) ≈ 0.007
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

期望：全部 PASS

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
- Modify: `models/architectures/vrt/vrt.py` (line 526)
- Modify: `tests/models/test_vrt_fusion_integration.py`

背景：VRT 的所有配置（fusion、output_mode、restoration_reducer）均通过 `opt` dict 传入，不接受直接 kwargs。现有测试使用 `opt=opt` 模式。

- [ ] **Step 1: 在 `tests/models/test_vrt_fusion_integration.py` 末尾追加测试**

```python
def test_vrt_stores_fusion_hook_after_forward():
    """VRT must store _last_fusion_out and _last_spike_bins after early fusion forward."""
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 7,
            },
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            },
            "output_mode": "restoration",
            "restoration_reducer": {"type": "index", "index": 2},
        }
    }
    model = VRT(
        upscale=1,
        in_chans=7,
        img_size=[6, 16, 16],
        window_size=[6, 8, 8],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=opt,
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

- [ ] **Step 3: 修改 `models/architectures/vrt/vrt.py`**

在第 526 行，将：

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

### Task 3: ModelPlain 中添加 `_compute_fusion_aux_loss` 和 Phase 感知 loss

**Files:**
- Modify: `models/model_plain.py`
- Create: `tests/models/test_model_plain_fusion_aux_loss.py`

背景：
- `ModelPlain.optimize_parameters` 在 line 410 计算 `G_loss`，line 413 开始 `backward` timer block。aux loss 必须在 backward 之前加入 `G_loss`，因此需要在 `loss_compute` timer block 内完成。
- `fix_iter` 属性只在 `ModelVRT.__init__` 中设置（`self.fix_iter = self.opt_train.get('fix_iter', 0)`）。`ModelPlain` 本身没有该属性，需用 `hasattr(self, 'fix_iter')` 安全访问。
- `self.device` 是 `ModelBase.__init__` 中设置的实例属性，`ModelPlain` 继承后可直接使用。

- [ ] **Step 1: 新建 `tests/models/test_model_plain_fusion_aux_loss.py`**

```python
"""Tests for ModelPlain._compute_fusion_aux_loss and phase-aware loss weighting."""
import torch
import pytest


def _make_opt(phase1_aux=1.0, phase2_aux=0.2, passthrough=0.2, fix_iter=10):
    return {
        'scale': 1,
        'n_channels': 3,
        'netG': {
            'net_type': 'vrt',
            'input': {'strategy': 'fusion', 'mode': 'dual', 'raw_ingress_chans': 7},
            'fusion': {
                'enable': True, 'placement': 'early', 'operator': 'gated',
                'out_chans': 3, 'operator_params': {},
            },
            'in_chans': 7, 'upscale': 1,
            'img_size': [6, 8, 8], 'window_size': [6, 8, 8],
            'depths': [2], 'indep_reconsts': [], 'embed_dims': [16], 'num_heads': [2],
            'output_mode': 'restoration',
            'restoration_reducer': {'type': 'index', 'index': 2},
            'pa_frames': 2,
            'use_flash_attn': False,
            'optical_flow': {'module': 'spynet', 'checkpoint': None, 'params': {}},
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
            'fix_iter': fix_iter,
            'fix_keys': [],
            'checkpoint_save': 100,
            'checkpoint_test': 100,
            'checkpoint_print': 10,
            'amp': {'enable': False},
            'freeze_backbone': False,
        },
        'path': {'root': '/tmp', 'pretrained_netG': None, 'pretrained_netE': None},
        'rank': 0,
        'dist': False,
    }


def _inject_fusion_hook(model, fusion_out, spike_bins):
    bare = model.get_bare_model(model.netG)
    bare._last_fusion_out = fusion_out
    bare._last_spike_bins = spike_bins
    B = fusion_out.shape[0]
    N = fusion_out.shape[1] // spike_bins
    H, W = fusion_out.shape[-2], fusion_out.shape[-1]
    model.H = torch.zeros(B, N, 3, H, W)
    # self.L: [B, N, 3+spike_chans, H, W]; first 3 channels = blur_rgb
    model.L = torch.ones(B, N, 7, H, W)


def test_fusion_aux_loss_phase1_returns_nonzero():
    from models.model_plain import ModelPlain
    model = ModelPlain(_make_opt())
    model.define_loss()
    fusion_out = torch.randn(1, 24, 3, 8, 8)  # N=6, S=4
    _inject_fusion_hook(model, fusion_out, spike_bins=4)
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    assert loss.item() > 0.0


def test_fusion_aux_loss_phase2_smaller_than_phase1():
    from models.model_plain import ModelPlain
    model = ModelPlain(_make_opt(phase1_aux=1.0, phase2_aux=0.2))
    model.define_loss()
    fusion_out = torch.randn(1, 24, 3, 8, 8)
    _inject_fusion_hook(model, fusion_out, spike_bins=4)
    loss_p1 = model._compute_fusion_aux_loss(is_phase1=True)
    loss_p2 = model._compute_fusion_aux_loss(is_phase1=False)
    assert loss_p2.item() < loss_p1.item()


def test_fusion_aux_loss_zero_when_all_weights_zero():
    from models.model_plain import ModelPlain
    model = ModelPlain(_make_opt(phase1_aux=0.0, phase2_aux=0.0, passthrough=0.0))
    model.define_loss()
    fusion_out = torch.randn(1, 24, 3, 8, 8)
    _inject_fusion_hook(model, fusion_out, spike_bins=4)
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    assert loss.item() == 0.0


def test_fusion_aux_loss_no_hook_returns_zero():
    from models.model_plain import ModelPlain
    model = ModelPlain(_make_opt())
    model.define_loss()
    fusion_out = torch.randn(1, 24, 3, 8, 8)
    _inject_fusion_hook(model, fusion_out, spike_bins=4)
    del model.get_bare_model(model.netG)._last_fusion_out
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    assert loss.item() == 0.0


def test_fusion_aux_loss_center_frame_indexing():
    """S//2::S indexing must select frames 2,6,10,14,18,22 for S=4."""
    from models.model_plain import ModelPlain
    model = ModelPlain(_make_opt())
    model.define_loss()
    fusion_out = torch.zeros(1, 24, 3, 8, 8)
    fusion_out[:, 2::4, :, :, :] = 1.0   # center frames = 1.0
    _inject_fusion_hook(model, fusion_out, spike_bins=4)
    model.H = torch.ones(1, 6, 3, 8, 8)  # GT = 1.0
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    # fusion_center == GT == 1.0 → Charbonnier loss ≈ 0
    assert loss.item() < 1e-4
```

- [ ] **Step 2: 运行新测试，确认失败**

```bash
python -m pytest tests/models/test_model_plain_fusion_aux_loss.py -v
```

期望：FAIL（`_compute_fusion_aux_loss` 不存在）

- [ ] **Step 3: 在 `models/model_plain.py` 中插入 `_compute_fusion_aux_loss` 方法**

在 `optimize_parameters` 方法定义之前（line 398 的注释行之前），插入：

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

- [ ] **Step 4: 修改 `optimize_parameters` 中的 loss_compute block**

将 `models/model_plain.py` 中（line 410-411）：

```python
        with self.timer.timer('loss_compute'):
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
```

替换为：

```python
        with self.timer.timer('loss_compute'):
            is_phase1 = (
                hasattr(self, 'fix_iter')
                and self.fix_iter > 0
                and current_step < self.fix_iter
            )
            vrt_loss_weight = 0.0 if is_phase1 else self.G_lossfn_weight
            G_loss = vrt_loss_weight * self.G_lossfn(self.E, self.H)
            G_loss = G_loss + self._compute_fusion_aux_loss(is_phase1=is_phase1)
```

- [ ] **Step 5: 运行新测试**

```bash
python -m pytest tests/models/test_model_plain_fusion_aux_loss.py -v
```

期望：全部 PASS

- [ ] **Step 6: 运行 models 回归测试**

```bash
python -m pytest tests/models/ -v --timeout=60
```

期望：全部 PASS

- [ ] **Step 7: Commit**

```bash
git add models/model_plain.py tests/models/test_model_plain_fusion_aux_loss.py
git commit -m "feat(model): add phase-aware fusion aux loss to ModelPlain

- _compute_fusion_aux_loss(is_phase1): reads _last_fusion_out hook from VRT
- Phase 1: vrt_loss_weight=0, aux from phase1_fusion_aux_loss_weight + passthrough
- Phase 2: vrt_loss_weight=G_lossfn_weight, aux from phase2_fusion_aux_loss_weight
- is_phase1 uses hasattr(self, 'fix_iter') for safe ModelPlain/ModelVRT compat"
```

---

### Task 4: `checkpoint_test` 列表化支持

**Files:**
- Modify: `main_train_vrt.py` (line 487, single occurrence)

- [ ] **Step 1: 修改 `main_train_vrt.py` line 487**

将：

```python
            if current_step % opt['train']['checkpoint_test'] == 0:
```

替换为：

```python
            _ckpt_test = opt['train']['checkpoint_test']
            if isinstance(_ckpt_test, list):
                _do_test = current_step in _ckpt_test
            else:
                _do_test = current_step % _ckpt_test == 0
            if _do_test:
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('main_train_vrt.py').read()); print('syntax OK')"
```

期望：`syntax OK`

- [ ] **Step 3: Commit**

```bash
git add main_train_vrt.py
git commit -m "feat(train): support checkpoint_test as list of specific steps"
```

---

### Task 5: 更新 `options/gopro_rgbspike_server.json`

**Files:**
- Modify: `options/gopro_rgbspike_server.json`

- [ ] **Step 1: 将 `"fix_iter": 20000` 改为 `"fix_iter": 6000`**

- [ ] **Step 2: 在 `"G_lossfn_weight": 1.0` 之后添加三个新字段**

```json
    ,
    "phase1_fusion_aux_loss_weight": 1.0
    ,
    "phase2_fusion_aux_loss_weight": 0.2
    ,
    "fusion_passthrough_loss_weight": 0.2
```

- [ ] **Step 3: 将 `"checkpoint_test": 30000` 改为 `"checkpoint_test": [6000, 30000]`**

- [ ] **Step 4: 验证 JSON 可解析（文件含 `//` 注释，用 strip 后解析）**

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

### Task 6: 冒烟验证

- [ ] **Step 1: passthrough 初始化验证**

```bash
python -c "
import torch
from models.fusion.operators.gated import GatedFusionOperator
op = GatedFusionOperator(3, 1, 3, {})
rgb = torch.ones(1, 1, 3, 8, 8) * 0.5
spike = torch.zeros(1, 1, 1, 8, 8)
with torch.no_grad():
    out = op(rgb, spike)
diff = (out - rgb).abs().max().item()
assert diff < 1e-6, f'passthrough failed: max_diff={diff}'
print(f'passthrough OK (max_diff={diff:.2e})')
"
```

期望：`passthrough OK (max_diff=0.00e+00)`

- [ ] **Step 2: 运行 e2e 测试**

```bash
python -m pytest tests/e2e/ -v --timeout=120 -x
```

期望：全部 PASS

- [ ] **Step 3: 确认无未提交文件**

```bash
git status
```

期望：`nothing to commit, working tree clean`
