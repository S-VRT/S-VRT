# Phase-wise Batch/GT Size Array Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 支持在 `datasets.train` 中将 `dataloader_batch_size` 和 `gt_size` 配置为 `[phase1, phase2]` 数组，并在 `current_step` 跨过 `fix_iter` 时立即重建 train dataloader 生效。

**Architecture:** 在 `main_train_vrt.py` 中新增 phase-aware 配置解析与 train loader 构建函数，把现有训练集创建逻辑抽到可重入路径。训练循环里检测 `is_phase1` 边界变化，一次性重建 `train_set/train_sampler/train_loader`。数据集侧只消费已解析后的 `gt_size` 数值，不改模型结构，不改 `netG.img_size`。

**Tech Stack:** Python 3.11, PyTorch DataLoader/DistributedSampler, existing S-VRT training entry (`main_train_vrt.py`), pytest

---

### Task 1: 增加 phase-aware 配置解析单元测试（先红）

**Files:**
- Create: `tests/test_phasewise_loader_config.py`
- Modify: `main_train_vrt.py` (导出可测试函数)

- [ ] **Step 1: 新建 `tests/test_phasewise_loader_config.py` 并写失败测试**

```python
import pytest

from main_train_vrt import resolve_phase_value, build_phase_train_dataset_opt


def test_resolve_phase_value_scalar_kept_for_both_phases():
    assert resolve_phase_value(4, True, "dataloader_batch_size") == 4
    assert resolve_phase_value(4, False, "dataloader_batch_size") == 4


def test_resolve_phase_value_array_phase1_phase2():
    assert resolve_phase_value([8, 4], True, "dataloader_batch_size") == 8
    assert resolve_phase_value([8, 4], False, "dataloader_batch_size") == 4


def test_resolve_phase_value_rejects_bad_array_length():
    with pytest.raises(ValueError, match="must be an int or a length-2 list/tuple"):
        resolve_phase_value([8, 4, 2], True, "dataloader_batch_size")


def test_resolve_phase_value_rejects_non_positive():
    with pytest.raises(ValueError, match="must be > 0"):
        resolve_phase_value([0, 4], True, "dataloader_batch_size")


def test_build_phase_train_dataset_opt_overrides_only_phase_keys():
    base = {
        "dataset_type": "TrainDatasetRGBSpike",
        "gt_size": [128, 96],
        "dataloader_batch_size": [8, 4],
        "dataloader_num_workers": 12,
        "dataloader_shuffle": True,
    }
    phase1 = build_phase_train_dataset_opt(base, is_phase1=True)
    phase2 = build_phase_train_dataset_opt(base, is_phase1=False)

    assert phase1["gt_size"] == 128
    assert phase1["dataloader_batch_size"] == 8
    assert phase2["gt_size"] == 96
    assert phase2["dataloader_batch_size"] == 4

    assert phase1["dataloader_num_workers"] == 12
    assert phase2["dataloader_shuffle"] is True
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
uv run pytest -q tests/test_phasewise_loader_config.py
```

Expected:

- FAIL，提示 `resolve_phase_value` / `build_phase_train_dataset_opt` 未定义。

- [ ] **Step 3: Commit（红测基线）**

```bash
git add tests/test_phasewise_loader_config.py
git commit -m "test(train): add failing tests for phase-wise loader config resolver"
```

---

### Task 2: 实现 phase-aware 解析函数并使单测转绿

**Files:**
- Modify: `main_train_vrt.py`
- Test: `tests/test_phasewise_loader_config.py`

- [ ] **Step 1: 在 `main_train_vrt.py` 顶层 helper 区新增函数**

```python
def resolve_phase_value(value, is_phase1, key_name):
    """Resolve scalar or [phase1, phase2] value to active phase value."""
    if isinstance(value, int):
        resolved = value
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        resolved = value[0] if is_phase1 else value[1]
    else:
        raise ValueError(
            f"{key_name} must be an int or a length-2 list/tuple [phase1, phase2], got {value!r}"
        )

    if not isinstance(resolved, int):
        raise ValueError(f"{key_name} resolved value must be int, got {type(resolved).__name__}")
    if resolved <= 0:
        raise ValueError(f"{key_name} must be > 0, got {resolved}")
    return resolved


def build_phase_train_dataset_opt(train_dataset_opt, is_phase1):
    """Build per-phase dataset options with resolved gt_size and dataloader_batch_size."""
    resolved = dict(train_dataset_opt)
    resolved["gt_size"] = resolve_phase_value(train_dataset_opt.get("gt_size", 256), is_phase1, "gt_size")
    resolved["dataloader_batch_size"] = resolve_phase_value(
        train_dataset_opt["dataloader_batch_size"], is_phase1, "dataloader_batch_size"
    )
    return resolved
```

- [ ] **Step 2: 导出函数供测试直接 import**

要求：函数定义在 `main()` 之外（模块顶层），保持 `from main_train_vrt import resolve_phase_value` 可用。

- [ ] **Step 3: 运行单测确认通过**

Run:

```bash
uv run pytest -q tests/test_phasewise_loader_config.py
```

Expected:

- PASS

- [ ] **Step 4: Commit（最小实现）**

```bash
git add main_train_vrt.py tests/test_phasewise_loader_config.py
git commit -m "feat(train): add phase-aware config resolver for batch size and gt size"
```

---

### Task 3: 将 train loader 构建逻辑抽取为可重建函数

**Files:**
- Modify: `main_train_vrt.py`
- Test: `tests/test_phasewise_loader_config.py`

- [ ] **Step 1: 在 `main_train_vrt.py` 新增可重入构建函数**

在 `main()` 外新增：

```python
def build_train_loader_bundle(opt, train_dataset_opt, is_phase1, seed, logger):
    dataset_opt = build_phase_train_dataset_opt(train_dataset_opt, is_phase1)
    train_set = define_Dataset(dataset_opt)

    batch_size = dataset_opt["dataloader_batch_size"]
    if opt["dist"]:
        if batch_size % opt["num_gpu"] != 0:
            raise ValueError(
                f"dataloader_batch_size={batch_size} is not divisible by num_gpu={opt['num_gpu']}"
            )
        per_gpu_batch = batch_size // opt["num_gpu"]
        if per_gpu_batch <= 0:
            raise ValueError(
                f"per-GPU batch size must be > 0, got {per_gpu_batch} "
                f"(global batch={batch_size}, num_gpu={opt['num_gpu']})"
            )

        train_sampler = DistributedSampler(
            train_set,
            shuffle=dataset_opt["dataloader_shuffle"],
            drop_last=True,
            seed=seed,
        )
        per_gpu_workers = dataset_opt["dataloader_num_workers"] // opt["num_gpu"]
        kwargs = dict(
            batch_size=per_gpu_batch,
            shuffle=False,
            num_workers=per_gpu_workers,
            drop_last=True,
            pin_memory=True,
            sampler=train_sampler,
        )
        if per_gpu_workers > 0:
            kwargs["persistent_workers"] = dataset_opt.get("dataloader_persistent_workers", False)
            kwargs["prefetch_factor"] = dataset_opt.get("dataloader_prefetch_factor", 2)
            kwargs["multiprocessing_context"] = "spawn"
        train_loader = DataLoader(train_set, **kwargs)
    else:
        train_sampler = None
        workers = dataset_opt["dataloader_num_workers"]
        kwargs = dict(
            batch_size=batch_size,
            shuffle=dataset_opt["dataloader_shuffle"],
            num_workers=workers,
            drop_last=True,
            pin_memory=True,
        )
        if workers > 0:
            kwargs["persistent_workers"] = dataset_opt.get("dataloader_persistent_workers", False)
            kwargs["prefetch_factor"] = dataset_opt.get("dataloader_prefetch_factor", 2)
            kwargs["multiprocessing_context"] = "spawn"
        train_loader = DataLoader(train_set, **kwargs)

    return {
        "dataset_opt": dataset_opt,
        "train_set": train_set,
        "train_sampler": train_sampler,
        "train_loader": train_loader,
    }
```

- [ ] **Step 2: 用新函数替换 `main()` 里原 train 数据集构建代码块**

替换 `for phase, dataset_opt in opt['datasets'].items(): if phase == 'train': ...` 中从 `define_Dataset` 到 `train_loader` 的重复逻辑，改为调用：

```python
train_dataset_opt_base = opt["datasets"]["train"]
is_phase1 = opt["train"].get("fix_iter", 0) > 0 and current_step < opt["train"].get("fix_iter", 0)
bundle = build_train_loader_bundle(opt, train_dataset_opt_base, is_phase1, seed, logger)
train_set = bundle["train_set"]
train_sampler = bundle["train_sampler"]
train_loader = bundle["train_loader"]
active_train_dataset_opt = bundle["dataset_opt"]
```

- [ ] **Step 3: 添加初始 phase 日志**

在 rank0 输出：

```python
logger.info(
    "[TRAIN_PHASE] phase=%s batch_size=%d gt_size=%d",
    "phase1" if is_phase1 else "phase2",
    active_train_dataset_opt["dataloader_batch_size"],
    active_train_dataset_opt["gt_size"],
)
```

- [ ] **Step 4: 添加一条针对 DDP 非整除的解析测试（先红后绿）**

在 `tests/test_phasewise_loader_config.py` 增加：

```python
def test_resolve_phase_value_non_int_rejected():
    with pytest.raises(ValueError, match="resolved value must be int"):
        resolve_phase_value([8, 4.5], False, "dataloader_batch_size")
```

- [ ] **Step 5: 运行测试**

Run:

```bash
uv run pytest -q tests/test_phasewise_loader_config.py
```

Expected:

- PASS

- [ ] **Step 6: Commit**

```bash
git add main_train_vrt.py tests/test_phasewise_loader_config.py
git commit -m "refactor(train): extract phase-aware train loader builder"
```

---

### Task 4: 训练循环内实现 fix_iter 边界即时重建

**Files:**
- Modify: `main_train_vrt.py`
- Test: `tests/test_phasewise_loader_config.py`

- [ ] **Step 1: 在训练主循环初始化 phase 状态跟踪变量**

在进入 epoch 循环前定义：

```python
fix_iter = opt["train"].get("fix_iter", 0)
last_is_phase1 = is_phase1
```

- [ ] **Step 2: 在 `current_step += 1` 后插入 phase 切换检测与重建逻辑**

```python
is_phase1_now = fix_iter > 0 and current_step < fix_iter
if is_phase1_now != last_is_phase1:
    bundle = build_train_loader_bundle(opt, train_dataset_opt_base, is_phase1_now, seed, logger)
    train_set = bundle["train_set"]
    train_sampler = bundle["train_sampler"]
    train_loader = bundle["train_loader"]
    active_train_dataset_opt = bundle["dataset_opt"]

    if opt["rank"] == 0:
        logger.info(
            "[TRAIN_PHASE] switch=%s batch_size=%d gt_size=%d (rebuild train loader)",
            "phase1" if is_phase1_now else "phase2",
            active_train_dataset_opt["dataloader_batch_size"],
            active_train_dataset_opt["gt_size"],
        )
    last_is_phase1 = is_phase1_now
```

实现要求：
- 切换只会发生一次（phase1->phase2）。
- 下一 step 使用新 loader（不延迟到下一 epoch）。

- [ ] **Step 3: 为切换判定新增纯函数测试**

在 `main_train_vrt.py` 增加：

```python
def compute_is_phase1(current_step, fix_iter):
    return fix_iter > 0 and current_step < fix_iter
```

在 `tests/test_phasewise_loader_config.py` 增加：

```python
from main_train_vrt import compute_is_phase1


def test_compute_is_phase1_boundary():
    assert compute_is_phase1(0, 10) is True
    assert compute_is_phase1(9, 10) is True
    assert compute_is_phase1(10, 10) is False
    assert compute_is_phase1(11, 10) is False
```

- [ ] **Step 4: 运行测试**

Run:

```bash
uv run pytest -q tests/test_phasewise_loader_config.py
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add main_train_vrt.py tests/test_phasewise_loader_config.py
git commit -m "feat(train): rebuild train dataloader immediately at phase boundary"
```

---

### Task 5: 配置示例与回归验证

**Files:**
- Modify: `options/gopro_rgbspike_server.json`
- Modify: `options/gopro_rgbspike_server_debug.json`
- Test: `tests/test_phasewise_loader_config.py`

- [ ] **Step 1: 在两个配置文件原字段位置改为数组并补注释**

示例（放在原键附近，保持可读）：

```jsonc
// 批次大小：支持单值或 [phase1, phase2]
"dataloader_batch_size": [8, 4],

// 真实标签图像尺寸：支持单值或 [phase1, phase2]
"gt_size": [128, 96],
```

注意：仅改训练集 `datasets.train` 的这两个键，不改 `netG.img_size`。

- [ ] **Step 2: 跑单测 + 现有受影响测试**

Run:

```bash
uv run pytest -q tests/test_phasewise_loader_config.py tests/models/test_model_plain_fusion_aux_loss.py tests/models/test_two_phase_training.py
```

Expected:

- PASS

- [ ] **Step 3: 语法检查（配置含注释，走现有解析路径）**

Run:

```bash
uv run python - <<'PY'
from utils import utils_option
opt = utils_option.parse('options/gopro_rgbspike_server_debug.json', is_train=True)
train = opt['datasets']['train']
print('train_batch=', train['dataloader_batch_size'])
print('train_gt_size=', train['gt_size'])
PY
```

Expected:

- 能成功解析，无异常。

- [ ] **Step 4: Commit**

```bash
git add options/gopro_rgbspike_server.json options/gopro_rgbspike_server_debug.json tests/test_phasewise_loader_config.py main_train_vrt.py
git commit -m "feat(config): support [phase1,phase2] batch and gt size for train dataset"
```

---

### Task 6: 端到端最小验证与收尾

**Files:**
- Modify: `docs/superpowers/specs/2026-04-21-phasewise-batch-gt-size-design.md` (如实现偏差需回填)

- [ ] **Step 1: 运行最小训练烟测（短步数）**

Run:

```bash
uv run python main_train_vrt.py --opt options/gopro_rgbspike_server_debug.json
```

Expected:

- 日志中出现初始 phase 行。
- 在跨 `fix_iter` 时出现 switch + rebuild 行。
- 切换前后打印的 `batch_size/gt_size` 与数组索引一致。

- [ ] **Step 2: 验证 spec 对齐（无偏差可不改）**

检查是否满足 spec 的 5 个 Goals；若实现细节调整但目标不变，无需改 spec。

- [ ] **Step 3: 最终回归测试**

Run:

```bash
uv run pytest -q tests/test_phasewise_loader_config.py tests/models/test_model_plain_fusion_aux_loss.py tests/models/test_two_phase_training.py tests/models/test_amp_model.py
```

Expected:

- PASS

- [ ] **Step 4: 最终 Commit（若有变更）**

```bash
git add -A
git commit -m "test(train): verify phase-wise loader switch and compatibility"
```

---

## Self-Review

- Spec coverage: 已覆盖数组 schema、立即切换、可读性、向后兼容、DDP 校验与日志可观测性。
- Placeholder scan: 无 TBD/TODO；每个代码步骤都给出可执行片段与命令。
- Type consistency: `resolve_phase_value`、`build_phase_train_dataset_opt`、`build_train_loader_bundle`、`compute_is_phase1` 在测试与实现命名一致。

---

Plan complete and saved to `docs/superpowers/plans/2026-04-21-phasewise-batch-gt-size-array-implementation.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
