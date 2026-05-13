# 两阶段 LoRA 训练方案设计

**状态**：已设计，待实施
**前置**：[2026-04-17-lora-stage-c-design.md](2026-04-17-lora-stage-c-design.md)（LoRA 基础设施已实现）

## 1. 背景与目标

VRT 主干在纯 RGB 数据（GoPro）上预训练。现在引入 spike+RGB 双模态输入，新增了 Gated Fusion 算子（early placement）。目标是在保护主干预训练权重的前提下，让模型适应新的多模态输入。

**约束**：
- VRT 主干基础权重**始终冻结**，只通过 LoRA adapter 适配
- SCFlow 有预训练权重（其他 spike 数据集），需要域适应
- DCN 有预训练权重（随主干 RGB 预训练），需要适应 SCFlow 的流统计特性
- Gated Fusion 无预训练权重，从头训练

## 2. 理论依据

### 2.1 原版 VRT 两阶段训练的先例

原版 VRT（`KAIR/options/vrt/006_train_vrt_videodeblurring_gopro.json`）：
```
fix_keys: ["spynet", "deform"]
fix_iter: 20000
fix_lr_mul: 0.125
```
先让主干学会任务，再解冻流估计（SpyNet）和对齐模块（DCN）。我们的场景与之完全同构：Fusion 对应"主干学习新任务"，SCFlow+DCN 对应"流估计+对齐模块"。

### 2.2 DCN 的流残差机制

DCN offset 公式（`models/blocks/dcn.py`）：
```
offset = max_residue_magnitude * tanh(conv_offset(features)) + flow.flip(1)
```
即使 DCN 冻结，SCFlow 的新光流仍通过加法残差直接参与对齐。**冻结 DCN ≠ 切断光流信号**。

### 2.3 Gated Fusion 的收敛特性

Gated Fusion 公式：`out = rgb_proj(rgb) + gate * fused`

主干冻结时，损失信号强烈推动 Fusion 输出接近 RGB（主干只能处理 RGB-like 输入）。~1k iter 后 Fusion 输出趋于合理。Phase 1 设 6k iter 足够让 Fusion 稳定并为 Phase 2 提供有意义的特征。

### 2.4 `fix_iter` 机制的问题

`model_vrt.py:77-79`：
```python
elif current_step == self.fix_iter:
    self.netG.requires_grad_(True)  # 解冻所有参数，包括主干基础权重
```
若用 `fix_iter` 做阶段切换，Phase 2 主干基础权重会被解冻，破坏 LoRA 的语义。因此采用**两次独立训练运行**（two-run）方案。

## 3. 训练方案

### Phase 1：Fusion 预热（Run 1）

**目标**：让 Gated Fusion 学会基本的 spike-RGB 融合，同时让 SCFlow 开始域适应。

| 模块 | 状态 | 理由 |
|---|---|---|
| Gated Fusion | 训练，4e-4 | 唯一全新模块，必须先稳定 |
| SCFlow | 冻结 | 遵循原版 VRT 哲学；流残差机制保证对齐不中断 |
| DCN | 冻结 | 保护预训练权重；Fusion 噪声阶段的梯度会破坏 conv_offset |
| VRT 主干基础权重 | 冻结 | 保护预训练权重 |
| LoRA | 不注入 | Phase 1 不需要 |

**实现**：`freeze_backbone=true` 已冻结主干+SCFlow+DCN，只保留 Fusion 可训练。

**时长**：6000 iter（Fusion 收敛 ~1k，额外 5k 确保稳定）

### Phase 2：LoRA + 对齐适配（Run 2）

**目标**：主干通过 LoRA 适配融合后的特征分布，SCFlow+DCN 完成域适应。

| 模块 | 学习率 | 理由 |
|---|---|---|
| Gated Fusion | 2e-4 | 继续微调，降低 LR |
| LoRA (qkv+proj) | 2e-4（base LR） | 主干适配新特征分布 |
| SCFlow | 2e-5（0.1× base） | 域适应，低 LR 保护预训练权重 |
| DCN | 2e-5（0.1× base） | 适应 SCFlow 流统计，低 LR |
| VRT 主干基础权重 | 冻结（0） | `freeze_backbone=true` 保证 |

**实现**：
- `freeze_backbone=true` + `use_lora=true`：主干基础冻结，Fusion+LoRA 可训练
- 新增 `trainable_extra_keys: ["spynet", "pa_deform"]`：解冻 SCFlow+DCN
- `fix_keys: ["spynet", "pa_deform"]` + `fix_lr_mul: 0.1` + `fix_iter: 0`：差异化学习率，不触发冻结逻辑

**时长**：24000 iter（总计 30000 iter）

## 4. 需要的代码变更

### 4.1 `model_plain.py`：新增 `trainable_extra_keys` 支持

在 `init_train()` 的 `freeze_backbone` 调用之后，新增：

```python
extra_keys = train_opt.get('trainable_extra_keys', [])
if extra_keys:
    for name, param in bare_model.named_parameters():
        if any(key in name for key in extra_keys):
            param.requires_grad = True
```

### 4.2 `model_vrt.py`：解耦差异化学习率与 `fix_iter`

`define_optimizer` 当前只在 `fix_iter > 0` 时创建差异化参数组。需要改为：当 `fix_keys` 非空且 `fix_lr_mul != 1` 时，**无论 `fix_iter` 是否为 0**，都创建差异化参数组。

修改条件：
```python
# 原来
if self.opt_train.get('fix_iter', 0) and len(self.fix_keys) > 0:

# 改为
fix_lr_mul = self.opt_train.get('fix_lr_mul', 1.0)
if len(self.fix_keys) > 0 and fix_lr_mul != 1.0:
```

同时，参数组构建时跳过 `requires_grad=False` 的参数（避免 optimizer 警告）。

## 5. 配置示例

### Phase 1 配置（新文件 `options/gopro_rgbspike_phase1.json`）

关键字段：
```json
"path": {
  "pretrained_netG": "weights/vrt/006_VRT_videodeblurring_GoPro.pth"
},
"train": {
  "freeze_backbone": true,
  "partial_load": true,
  "use_lora": false,
  "fix_iter": 0,
  "fix_keys": [],
  "G_optimizer_lr": 4e-4,
  "total_iter": 6000,
  "checkpoint_save": 1000,
  "checkpoint_test": 6000
}
```

### Phase 2 配置（新文件 `options/gopro_rgbspike_phase2.json`）

关键字段：
```json
"path": {
  "pretrained_netG": "experiments/<phase1_task>/models/6000_G.pth"
},
"train": {
  "freeze_backbone": true,
  "partial_load": true,
  "use_lora": true,
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_target_modules": ["qkv", "proj"],
  "trainable_extra_keys": ["spynet", "pa_deform"],
  "fix_iter": 0,
  "fix_keys": ["spynet", "pa_deform"],
  "fix_lr_mul": 0.1,
  "G_optimizer_lr": 2e-4,
  "total_iter": 24000,
  "checkpoint_save": 2000,
  "checkpoint_test": 24000
}
```

## 6. Checkpoint 策略

| Checkpoint | 内容 | 用途 |
|---|---|---|
| Phase 1 `6000_G.pth` | 主干预训练权重 + Fusion 权重 | Phase 2 起点 |
| Phase 2 `*_G.pth` | 主干预训练权重 + LoRA A/B + Fusion + SCFlow + DCN | 中途恢复 |
| Phase 2 `{total_iter}_G_merged.pth` | 与原 VRT state_dict 同构（LoRA 已合并） | 交付 |

Phase 2 加载 Phase 1 checkpoint 时：
- `partial_load=true` 按 key+shape 匹配加载
- `lora_A/B` 在 Phase 1 ckpt 中不存在 → 保持新初始化（B=0，初始 forward 等于 Phase 1 结果）
- Fusion 权重自然继承

## 7. 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 阶段切换方式 | 两次独立运行 | 避免 `fix_iter` 的 `requires_grad_(True)` 解冻主干基础权重 |
| Phase 1 是否解冻 SCFlow | 否 | 遵循原版 VRT 哲学；流残差机制保证对齐不中断 |
| Phase 1 是否解冻 DCN | 否 | 保护预训练权重；Fusion 噪声阶段梯度有害 |
| Phase 2 SCFlow+DCN LR | 0.1× base | 对应原版 VRT fix_lr_mul=0.125，略低因域差距更大 |
| LoRA 目标层 | qkv+proj | attention 层最需要适配新特征分布；FFN 不需要 |
| LoRA rank | 8 | 参数量约 1-3%，与 Stage C 设计一致 |

## 8. 范围外

- Phase 1 中 SCFlow 的解冻（可作为消融实验）
- DCN 和 SCFlow 使用不同学习率（当前统一 0.1×，可后续细化）
- Phase 3（可选的全量低 LR 收尾）
