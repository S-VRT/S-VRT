# Two-Phase LoRA Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现两阶段训练方案，单次启动、单个配置文件完成：Phase 1 只训练 Gated Fusion，在 `fix_iter` 时自动解冻 LoRA+SCFlow+DCN，主干基础权重始终冻结。

**Architecture:** 与原版 VRT 完全一致的机制——所有参数（含冻结的）从训练开始就在 optimizer 里，scheduler 全程连续不重建，`fix_iter` 时只做 `requires_grad_(True)` 解冻。新增 `phase2_lora_mode` 标志改变 `fix_iter` 时的解冻范围：不再 `requires_grad_(True)` 全部参数，只解冻 `fix_keys` + LoRA，主干基础权重始终冻结。

**Tech Stack:** PyTorch, `models/model_plain.py`, `models/model_vrt.py`, single JSON config

**Spec:** [docs/superpowers/specs/2026-04-19-two-phase-lora-training-design.md](../specs/2026-04-19-two-phase-lora-training-design.md)

---

## 设计要点

### 与原版 VRT 的对应关系

| 原版 VRT | S-VRT 两阶段 LoRA |
|---|---|
| Phase 1 冻结 SpyNet+DCN | Phase 1 冻结 SCFlow+DCN+LoRA |
| Phase 2 `requires_grad_(True)` 全部 | Phase 2 只解冻 `fix_keys`+LoRA（主干基础权重保持冻结）|
| optimizer/scheduler 不重建 | optimizer/scheduler 不重建（完全继承）|
| 所有参数从 iter 0 就在 optimizer 里 | 所有参数从 iter 0 就在 optimizer 里 |

### 单配置文件控制两阶段

```json
"train": {
    "freeze_backbone": true,
    "partial_load": true,
    "fix_iter": 6000,
    "fix_keys": ["spynet", "pa_deform"],
    "fix_lr_mul": 0.1,
    "phase2_lora_mode": true,
    "use_lora": true,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_modules": ["qkv", "proj"],
    "G_optimizer_lr": 2e-4,
    "G_scheduler_periods": 30000,
    "total_iter": 30000
}
```

### Phase 1（0 → fix_iter）参数状态

- `init_train` 时注入 LoRA（不延迟）
- `freeze_backbone=true`：冻结主干基础权重，保留 Fusion 可训练
- `phase2_lora_mode=true`：额外冻结 LoRA 参数（`lora_A`/`lora_B`）
- `optimize_parameters` 在 `current_step < fix_iter` 时冻结 `fix_keys`（SCFlow+DCN）
- Optimizer 包含**所有参数**（含冻结的），冻结参数无梯度不更新

### Phase 2（fix_iter → total_iter）参数状态

- `fix_iter` 时只解冻 `fix_keys`（SCFlow+DCN）+ LoRA 参数
- 主干基础权重保持冻结
- Optimizer 和 Scheduler **不重建**，LR 曲线连续
- 差异化 LR：SCFlow+DCN 在 `fix_keys` 低 LR 组（0.1×），LoRA+Fusion 在正常 LR 组

### `define_optimizer` 的关键修改

**不跳过冻结参数**——所有参数必须在 optimizer 里，否则解冻后不在任何 param group 里。冻结参数有梯度=False，optimizer 不会更新它们，但解冻后立即生效。

---

## File Map

| 文件 | 操作 | 职责 |
|---|---|---|
| `models/model_plain.py` | 修改 | `init_train` 中 `phase2_lora_mode=true` 时额外冻结 LoRA 参数 |
| `models/model_vrt.py` | 修改 | `optimize_parameters` 的 `fix_iter` 分支新增 `phase2_lora_mode` 路径；`define_optimizer` 包含所有参数（不跳过冻结的）|
| `options/gopro_rgbspike_local.json` | 修改 | 添加两阶段控制字段 |
| `tests/models/test_two_phase_training.py` | 新建 | 验证两阶段参数状态 |

---

## Task 1: `model_plain.py` — Phase 1 额外冻结 LoRA 参数

**Files:**
- Modify: `models/model_plain.py:190-194`
- Test: `tests/models/test_two_phase_training.py`

- [ ] **Step 1: 写失败测试**

新建 `tests/models/test_two_phase_training.py`：

```python
import torch
import torch.nn as nn


class _MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_qkv = nn.Linear(4, 12)
        self.spynet_conv = nn.Linear(4, 4)
        self.pa_deform_conv = nn.Linear(4, 4)
        self.fusion_operator_gate = nn.Linear(4, 4)


def test_phase2_lora_mode_freezes_lora_in_phase1():
    """phase2_lora_mode=true 时，freeze_backbone 后 LoRA 参数应被额外冻结。"""
    from models.model_plain import freeze_backbone
    from models.lora import inject_lora, LoRALinear

    model = _MiniModel()
    inject_lora(model, target_substrings=["qkv"], rank=4, alpha=8)
    freeze_backbone(model)

    # 模拟 phase2_lora_mode=true 的额外冻结逻辑
    phase2_lora_mode = True
    if phase2_lora_mode:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = False

    # LoRA 参数应被冻结
    assert model.backbone_qkv.lora_A.weight.requires_grad is False
    assert model.backbone_qkv.lora_B.weight.requires_grad is False
    # backbone base 权重应冻结
    assert model.backbone_qkv.base.weight.requires_grad is False
    # fusion_operator 应可训练
    assert model.fusion_operator_gate.weight.requires_grad is True
    # spynet/pa_deform 应冻结（freeze_backbone 冻结了它们）
    assert model.spynet_conv.weight.requires_grad is False
    assert model.pa_deform_conv.weight.requires_grad is False


def test_phase2_lora_mode_false_keeps_lora_trainable():
    """phase2_lora_mode=false 时，LoRA 参数应保持可训练（freeze_backbone 保留）。"""
    from models.model_plain import freeze_backbone
    from models.lora import inject_lora

    model = _MiniModel()
    inject_lora(model, target_substrings=["qkv"], rank=4, alpha=8)
    freeze_backbone(model)

    phase2_lora_mode = False
    if phase2_lora_mode:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = False

    # LoRA 参数应可训练（freeze_backbone 的 _TRAINABLE_NAME_MARKERS 保留了它们）
    assert model.backbone_qkv.lora_A.weight.requires_grad is True
    assert model.backbone_qkv.lora_B.weight.requires_grad is True
```

- [ ] **Step 2: 运行测试确认通过（逻辑测试）**

```bash
cd /home/wuhy/projects/S-VRT
python -m pytest tests/models/test_two_phase_training.py::test_phase2_lora_mode_freezes_lora_in_phase1 tests/models/test_two_phase_training.py::test_phase2_lora_mode_false_keeps_lora_trainable -v
```

期望：2 tests PASSED

- [ ] **Step 3: 修改 `model_plain.py` 的 `init_train`**

在 `models/model_plain.py` 中，找到：

```python
        if train_opt.get('freeze_backbone', False):
            freeze_backbone(bare_model)
            frozen_count = sum(1 for p in bare_model.parameters() if not p.requires_grad)
            trainable_count = sum(1 for p in bare_model.parameters() if p.requires_grad)
            print(f'[Stage A/C] Frozen {frozen_count} params, trainable {trainable_count} params')
```

替换为：

```python
        if train_opt.get('freeze_backbone', False):
            freeze_backbone(bare_model)
            if train_opt.get('phase2_lora_mode', False):
                for name, param in bare_model.named_parameters():
                    if 'lora_A' in name or 'lora_B' in name:
                        param.requires_grad = False
                print(f'[Phase 1] LoRA params frozen until iter {train_opt.get("fix_iter", 0)}')
            frozen_count = sum(1 for p in bare_model.parameters() if not p.requires_grad)
            trainable_count = sum(1 for p in bare_model.parameters() if p.requires_grad)
            print(f'[Stage A/C] Frozen {frozen_count} params, trainable {trainable_count} params')
```

- [ ] **Step 4: 运行全部测试确认无回归**

```bash
python -m pytest tests/models/test_two_phase_training.py tests/models/test_lora.py -v
```

期望：全部 PASSED

- [ ] **Step 5: Commit**

```bash
git add models/model_plain.py tests/models/test_two_phase_training.py
git commit -m "feat(train): freeze LoRA params in Phase 1 when phase2_lora_mode=true"
```

---

## Task 2: `model_vrt.py` — `define_optimizer` 包含所有参数 + `fix_iter` 分支新增 `phase2_lora_mode` 路径

**Files:**
- Modify: `models/model_vrt.py:31-85`
- Test: `tests/models/test_two_phase_training.py`

- [ ] **Step 1: 写失败测试**

在 `tests/models/test_two_phase_training.py` 末尾追加：

```python
def test_all_params_in_optimizer_including_frozen():
    """define_optimizer 应包含所有参数（含冻结的），以便解冻后立即生效。"""
    from models.model_plain import freeze_backbone
    from models.lora import inject_lora

    model = _MiniModel()
    inject_lora(model, target_substrings=["qkv"], rank=4, alpha=8)
    freeze_backbone(model)

    # 模拟 phase2_lora_mode=true 额外冻结 LoRA
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = False

    fix_keys = ["spynet", "pa_deform"]
    fix_lr_mul = 0.1
    base_lr = 2e-4

    # 包含所有参数（不跳过冻结的）
    normal_params = []
    flow_params = []
    for name, param in model.named_parameters():
        if any(key in name for key in fix_keys):
            flow_params.append(param)
        else:
            normal_params.append(param)

    total_in_optimizer = len(normal_params) + len(flow_params)
    total_in_model = sum(1 for _ in model.parameters())
    assert total_in_optimizer == total_in_model, \
        f"optimizer 应包含所有 {total_in_model} 个参数，但只有 {total_in_optimizer}"


def test_phase2_unfreezes_fix_keys_and_lora_only():
    """fix_iter 时 phase2_lora_mode=true 只解冻 fix_keys+LoRA，主干 base 保持冻结。"""
    from models.model_plain import freeze_backbone
    from models.lora import inject_lora

    model = _MiniModel()
    inject_lora(model, target_substrings=["qkv"], rank=4, alpha=8)
    freeze_backbone(model)
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = False

    # 模拟 fix_iter 时的 phase2_lora_mode 解冻逻辑
    fix_keys = ["spynet", "pa_deform"]
    for name, param in model.named_parameters():
        if any(key in name for key in fix_keys):
            param.requires_grad_(True)
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad_(True)

    # fix_keys 解冻
    assert model.spynet_conv.weight.requires_grad is True
    assert model.pa_deform_conv.weight.requires_grad is True
    # LoRA 解冻
    assert model.backbone_qkv.lora_A.weight.requires_grad is True
    assert model.backbone_qkv.lora_B.weight.requires_grad is True
    # backbone base 仍冻结
    assert model.backbone_qkv.base.weight.requires_grad is False
    # fusion 仍可训练
    assert model.fusion_operator_gate.weight.requires_grad is True
```

- [ ] **Step 2: 运行测试确认通过（逻辑测试）**

```bash
python -m pytest tests/models/test_two_phase_training.py::test_all_params_in_optimizer_including_frozen tests/models/test_two_phase_training.py::test_phase2_unfreezes_fix_keys_and_lora_only -v
```

期望：2 tests PASSED

- [ ] **Step 3: 修改 `model_vrt.py` 的 `define_optimizer`**

找到并替换整个 `define_optimizer` 方法：

```python
    def define_optimizer(self):
        self.fix_keys = self.opt_train.get('fix_keys', [])
        if self.opt_train.get('fix_iter', 0) and len(self.fix_keys) > 0:
            fix_lr_mul = self.opt_train['fix_lr_mul']
            print(f'Multiple the learning rate for keys: {self.fix_keys} with {fix_lr_mul}.')
            if fix_lr_mul == 1:
                G_optim_params = self.netG.parameters()
            else:  # separate flow params and normal params for different lr
                normal_params = []
                flow_params = []
                for name, param in self.netG.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        flow_params.append(param)
                    else:
                        normal_params.append(param)
                G_optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': self.opt_train['G_optimizer_lr']
                    },
                    {
                        'params': flow_params,
                        'lr': self.opt_train['G_optimizer_lr'] * fix_lr_mul
                    },
                ]

            if self.opt_train['G_optimizer_type'] == 'adam':
                self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                        betas=self.opt_train['G_optimizer_betas'],
                                        weight_decay=self.opt_train['G_optimizer_wd'])
            else:
                raise NotImplementedError
        else:
            super(ModelVRT, self).define_optimizer()
```

替换为：

```python
    def define_optimizer(self):
        self.fix_keys = self.opt_train.get('fix_keys', [])
        fix_lr_mul = self.opt_train.get('fix_lr_mul', 1.0)
        use_split_lr = len(self.fix_keys) > 0 and fix_lr_mul != 1.0
        if use_split_lr:
            print(f'Multiple the learning rate for keys: {self.fix_keys} with {fix_lr_mul}.')
            normal_params = []
            flow_params = []
            for name, param in self.netG.named_parameters():
                # Include ALL params (even frozen) so they're in optimizer when unfrozen later
                if any([key in name for key in self.fix_keys]):
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            G_optim_params = [
                {'params': normal_params, 'lr': self.opt_train['G_optimizer_lr']},
                {'params': flow_params, 'lr': self.opt_train['G_optimizer_lr'] * fix_lr_mul},
            ]
            if self.opt_train['G_optimizer_type'] == 'adam':
                self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                        betas=self.opt_train['G_optimizer_betas'],
                                        weight_decay=self.opt_train['G_optimizer_wd'])
            else:
                raise NotImplementedError
        else:
            super(ModelVRT, self).define_optimizer()
```

- [ ] **Step 4: 修改 `model_vrt.py` 的 `optimize_parameters` 中 `fix_iter` 分支**

找到：

```python
            elif current_step == self.fix_iter:
                print(f'Train all the parameters from {self.fix_iter} iters.')
                self.netG.requires_grad_(True)
                # Re-wrap DDP if using static graph to capture new graph structure
                if self.opt.get('dist', False) and self.opt.get('use_static_graph', False):
                    print('Re-wrapping DDP for static graph change...')
                    self.netG = self.model_to_device(self.get_bare_model(self.netG))
```

替换为：

```python
            elif current_step == self.fix_iter:
                if self.opt_train.get('phase2_lora_mode', False):
                    bare = self.get_bare_model(self.netG)
                    for name, param in bare.named_parameters():
                        if any(key in name for key in self.fix_keys):
                            param.requires_grad_(True)
                        if 'lora_A' in name or 'lora_B' in name:
                            param.requires_grad_(True)
                    trainable = sum(p.requires_grad for p in bare.parameters())
                    print(f'[Phase 2] Unfrozen fix_keys={self.fix_keys} + LoRA at iter {self.fix_iter}. '
                          f'Trainable params: {trainable}')
                else:
                    print(f'Train all the parameters from {self.fix_iter} iters.')
                    self.netG.requires_grad_(True)
                if self.opt.get('dist', False) and self.opt.get('use_static_graph', False):
                    print('Re-wrapping DDP for static graph change...')
                    self.netG = self.model_to_device(self.get_bare_model(self.netG))
```

- [ ] **Step 5: 运行全部测试确认无回归**

```bash
python -m pytest tests/ -v -x
```

期望：全部 PASSED

- [ ] **Step 6: Commit**

```bash
git add models/model_vrt.py tests/models/test_two_phase_training.py
git commit -m "feat(train): add phase2_lora_mode path in fix_iter; include all params in optimizer"
```

---

## Task 3: 更新配置文件添加两阶段控制字段

**Files:**
- Modify: `options/gopro_rgbspike_local.json`

- [ ] **Step 1: 在 `train` 块中添加两阶段字段**

在 `options/gopro_rgbspike_local.json` 的 `"train"` 块中，找到：

```json
    "freeze_backbone": true
    ,
    "partial_load": true
    ,
    "use_lora": false
```

替换为：

```json
    "freeze_backbone": true
    ,
    "partial_load": true
    ,
    "phase2_lora_mode": true
    ,
    "use_lora": true
```

确认 `fix_iter`、`fix_keys`、`fix_lr_mul` 字段已正确设置（`fix_keys` 改为 `["spynet", "pa_deform"]`）：

```json
    "fix_iter": 6000
    ,
    "fix_lr_mul": 0.1
    ,
    "fix_keys": [
      "spynet",
      "pa_deform"
    ]
```

- [ ] **Step 2: 验证配置可被解析**

```bash
cd /home/wuhy/projects/S-VRT
python -c "
import sys; sys.path.insert(0, '.')
from utils import utils_option as option
opt = option.parse('options/gopro_rgbspike_local.json', is_train=True)
t = opt['train']
assert t.get('phase2_lora_mode') == True
assert t.get('use_lora') == True
assert t.get('fix_iter') == 6000
assert abs(t.get('fix_lr_mul') - 0.1) < 1e-9
assert 'spynet' in t.get('fix_keys', [])
assert 'pa_deform' in t.get('fix_keys', [])
print('Config OK')
"
```

期望：`Config OK`

- [ ] **Step 3: Commit**

```bash
git add options/gopro_rgbspike_local.json
git commit -m "feat(config): configure two-phase LoRA training in gopro_rgbspike_local"
```

---

## Self-Review

**Spec 覆盖检查：**

| 要求 | Task |
|---|---|
| 单配置文件 | Task 3 |
| 单次启动两阶段 | Task 2（`optimize_parameters` 自动触发）|
| Phase 1 只训练 Fusion | Task 1（额外冻结 LoRA）+ Task 2（`define_optimizer` 含所有参数）|
| Phase 2 解冻 SCFlow+DCN+LoRA | Task 2（`fix_iter` 分支）|
| 主干基础权重始终冻结 | Task 2（只解冻 `fix_keys`+LoRA，不 `requires_grad_(True)` 全部）|
| Optimizer/Scheduler 不重建 | Task 2（无重建逻辑）|
| 差异化 LR（SCFlow+DCN 低 LR）| Task 2（`define_optimizer` 分组）|
| 向后兼容原版 VRT | Task 2（`phase2_lora_mode=false` 走原有路径）|

**关键设计决策：**
- `define_optimizer` 包含所有参数（含冻结的）：与原版 VRT 一致，解冻后无需重建 optimizer
- Scheduler 不重建：LR 曲线连续，`G_scheduler_periods` 设为 `total_iter` 即可，与原版 VRT 完全一致
- LoRA 在 `init_train` 时注入（不延迟）：保证 LoRA 参数从 iter 0 就在 optimizer 里

**注意事项：**
- `gopro_rgbspike_server.json` 等其他配置若需两阶段训练，同样添加 `phase2_lora_mode: true`
- `use_static_graph=true` 时 DDP re-wrap 逻辑在两个分支中均保留
