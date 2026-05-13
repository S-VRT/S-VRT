# S-VRT Dual-Input Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 `concat` 与 `dual` 双输入路径的配置化切换，把融合职责收敛到模型侧，并保证旧 `L` 单键训练/测试配置继续可用。

**Architecture:** 数据集层新增 `input_pack_mode` 和 `keep_legacy_l`，输出 `L` 或 `L_rgb + L_spike (+L)`；模型层新增 `input_mode` 与统一输入归一化入口，在 `dual` 模式下优先 `L_rgb/L_spike`，缺失时可回退 `L`。VRT 保持当前 fusion placement（early/middle/hybrid）实现，仅补齐输入契约与生效日志。

**Tech Stack:** Python, PyTorch, existing VRT/fusion modules, pytest.

---

## File Structure Map

### Modify
1. `data/dataset_video_train_rgbspike.py`
2. `models/model_plain.py`
3. `models/model_vrt.py`
4. `models/architectures/vrt/vrt.py`
5. `options/006_train_vrt_videodeblurring_gopro_rgbspike.json`
6. `options/gopro_rgbspike_local_debug.json`
7. `docs/superpowers/specs/2026-04-10-svrt-dual-input-fusion-design.md`（追加实现状态/链接）

### Create
1. `tests/data/test_dataset_rgbspike_pack_modes.py`
2. `tests/models/test_model_plain_dual_input_feed.py`
3. `tests/models/test_vrt_dual_input_priority.py`

## Task 1: 数据集支持 `concat | dual` 打包契约

**Files:**
- Modify: `data/dataset_video_train_rgbspike.py`
- Test: `tests/data/test_dataset_rgbspike_pack_modes.py`

- [ ] **Step 1: 先写失败测试，锁定输出键契约**
Run: `pytest tests/data/test_dataset_rgbspike_pack_modes.py -v`
Expected: FAIL（`input_pack_mode`、`L_rgb/L_spike` 相关断言未满足）

- [ ] **Step 2: 增加配置解析**
在 `TrainDatasetRGBSpike.__init__` 增加：
`input_pack_mode`（默认 `concat`）、`keep_legacy_l`（默认 `true`）。

- [ ] **Step 3: 实现 dual 打包输出**
在 `__getitem__` 中构造：
`L_rgb: [T,3,H,W]`、`L_spike: [T,S,H,W]`，并按模式返回：
`concat -> {'L', 'H', 'key'}`
`dual -> {'L_rgb','L_spike','H','key'}`，若 `keep_legacy_l=true` 额外保留 `L`。

- [ ] **Step 4: 校验通道与形状一致性**
保留现有通道断言；新增 dual 下 rgb/spike 的 shape 检查和错误信息。

- [ ] **Step 5: 跑测试确认通过**
Run: `pytest tests/data/test_dataset_rgbspike_pack_modes.py -v`
Expected: PASS

## Task 2: 训练/验证入口支持 dual 输入

**Files:**
- Modify: `models/model_plain.py`
- Modify: `models/model_vrt.py`
- Test: `tests/models/test_model_plain_dual_input_feed.py`

- [ ] **Step 1: 先写失败测试**
Run: `pytest tests/models/test_model_plain_dual_input_feed.py -v`
Expected: FAIL（`feed_data` 仅支持 `data['L']`）

- [ ] **Step 2: 增加统一输入归一化函数**
在 `ModelPlain` 增加私有方法（例如 `_build_model_input_tensor`）：
根据 `netG.input_mode` 处理 `concat/dual`，统一产出 `self.L`。

- [ ] **Step 3: dual 优先级与回退策略**
当 `input_mode=dual`：
优先 `L_rgb + L_spike` 拼接；
若缺失双键但存在 `L`，打印兼容回退日志并使用 `L`；
若都缺失，抛明确异常。

- [ ] **Step 4: 与现有通道断言对齐**
拼接后继续走 `_assert_lq_channels`，确保 `netG.in_chans` 约束不被绕过。

- [ ] **Step 5: 跑测试确认通过**
Run: `pytest tests/models/test_model_plain_dual_input_feed.py -v`
Expected: PASS

## Task 3: VRT 输入模式与生效路径日志

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Test: `tests/models/test_vrt_dual_input_priority.py`

- [ ] **Step 1: 先写失败测试（路径优先级）**
Run: `pytest tests/models/test_vrt_dual_input_priority.py -v`
Expected: FAIL（缺少 `input_mode` 语义或日志断言）

- [ ] **Step 2: 增加 `input_mode` 配置语义**
在 VRT 初始化阶段读取 `netG.input_mode`（默认 `concat`）；
保留现有 fusion 配置解析，不改 placement/adapter 逻辑。

- [ ] **Step 3: 明确日志输出**
在 forward 起始处打印一次路径标识（可通过 logger）：
`concat_path` / `dual_path` / `dual_fallback_to_concat_path`。

- [ ] **Step 4: dual+fusion 约束检查**
当 `input_mode=dual` 且 `fusion.enable=true`，校验 early/middle/hybrid 的输入维度前置条件，报错信息要含当前 mode 与关键维度。

- [ ] **Step 5: 跑测试确认通过**
Run: `pytest tests/models/test_vrt_dual_input_priority.py tests/models/test_vrt_fusion_integration.py -v`
Expected: PASS

## Task 4: 配置迁移与样例收敛

**Files:**
- Modify: `options/006_train_vrt_videodeblurring_gopro_rgbspike.json`
- Modify: `options/gopro_rgbspike_local_debug.json`

- [ ] **Step 1: 引入分层配置字段**
将数据配置迁移为：
`datasets.*.spike.reconstruction.type/num_bins`
并保留旧字段兼容读取（仅过渡期）。

- [ ] **Step 2: 增加 pack/mode 配置**
数据侧新增 `input_pack_mode`、`keep_legacy_l`；
模型侧新增 `input_mode`。

- [ ] **Step 3: 提供两套可直接运行样例**
`concat baseline` 与 `dual + fusion` 各一套关键片段，避免“写了但没生效”。

- [ ] **Step 4: 配置静态检查**
Run: `python -m pytest tests/models/test_vrt_fusion_integration.py -k "config or full_t" -v`
Expected: PASS

## Task 5: 验收矩阵与回归

**Files:**
- Test: `tests/data/test_dataset_rgbspike_pack_modes.py`
- Test: `tests/models/test_model_plain_dual_input_feed.py`
- Test: `tests/models/test_vrt_dual_input_priority.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: 覆盖最小验收矩阵**
1) `concat + fusion.disable`  
2) `dual + early(preserve_n)`  
3) `dual + middle`  
4) `dual + hybrid`  
5) `dual + early(expand_to_full_t=true)`（实验模式）

- [ ] **Step 2: 执行回归命令**
Run: `pytest tests/data/test_dataset_rgbspike_pack_modes.py tests/models/test_model_plain_dual_input_feed.py tests/models/test_vrt_dual_input_priority.py tests/models/test_vrt_fusion_integration.py -v`
Expected: PASS

- [ ] **Step 3: 手工 smoke**
Run: `python main_train_vrt.py --opt options/gopro_rgbspike_local_debug.json`
Expected: 单步可启动；日志能看到 pack_mode/input_mode/fusion_placement/shape_policy 生效摘要。

## Self-Review

1. **Spec coverage:** 已覆盖数据输出契约、模型输入契约、fusion 放置语义、兼容迁移、验收矩阵。  
2. **Placeholder scan:** 无 `TBD/TODO/implement later`。  
3. **Type consistency:** 输出形状统一按 `[T,C,H,W]`（dataset）和 `[B,T,C,H,W]`（model）约束。

