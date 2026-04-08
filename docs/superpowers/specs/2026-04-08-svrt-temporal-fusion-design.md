# S-VRT Temporal Fusion 设计文档（Early/Middle 可插拔）

## 1. 背景与目标

当前 S-VRT 基线为 RGB 与 Spike 的早期通道拼接。该方案缺少两类能力：

1. 融合位置不可切换（仅 early）。
2. 融合算子不可插拔（难以在 concat / gated / PASE / Mamba 间切换）。

本设计目标是在尽量低侵入下，实现：

1. 通过配置切换融合位置：`early | middle | hybrid`。
2. 通过配置切换融合算子：`concat | gated | pase | mamba`。
3. middle 融合支持仅在部分 stage 注入，并可配置注入行为：`replace | residual`（默认 `replace`）。
4. early full-T 路径采用“重建后 Spike 时序”而非原始 `.dat` 直接融合。


## 2. 范围与非目标

### 2.1 范围

1. 设计并实现统一 fusion 配置协议与工厂。
2. 在 VRT 主干中以低耦合方式接入 early/middle/hybrid。
3. 保持现有训练/推理入口脚本不改调用方式。

### 2.2 非目标

1. 本阶段不修改数据重建算法本身（TFP/middle_tfp/snn）。
2. 本阶段不保证 `middle_tfp/snn` 支持 full-T early 路径。
3. 本阶段不引入新任务损失，仅改网络前向融合策略。


## 3. 关键约束与结论

1. “每个 RGB 对应 T 个 Spike 帧”中的 `T` 指重建后的固定时序长度（当前由配置控制，典型对应 `spike_channels`），不是原始 `.dat` 的原生比特流全长度。
2. early full-T 模式下输出时序长度为：
   `T_out = num_frame * T`。
3. middle 默认模式为 `replace`，可切换 `residual`。
4. middle 注入采用 stage 列表控制，支持部分注入并可扩展到全注入。


## 4. 总体架构（两层解耦）

将“融合位置”和“融合算子”解耦：

1. `FusionAdapter`（位置层）：
   1. `EarlyFusionAdapter`
   2. `MiddleFusionAdapter`
   3. `HybridFusionAdapter`（组合 early + middle）
2. `FusionOperator`（算法层）：
   1. `ConcatFusionOperator`
   2. `GatedFusionOperator`
   3. `PaseFusionOperator`
   4. `MambaFusionOperator`

位置层只负责任务编排、时序对齐与尺度对齐；算法层只做特征融合计算。这样可复用同一算子于 early 与 middle。


## 5. 配置协议

在 `netG` 下新增统一字段：

```json
{
  "fusion": {
    "enable": true,
    "placement": "early",
    "operator": "gated",
    "mode": "replace",
    "inject_stages": [1, 3, 5],
    "out_chans": 3,
    "flow_input": "rgb",
    "operator_params": {},
    "early": {
      "expand_to_full_t": true
    },
    "middle": {
      "allow_scale_align": true
    }
  }
}
```

字段语义：

1. `placement`: `early | middle | hybrid`。
2. `operator`: `concat | gated | pase | mamba`。
3. `mode`: `replace | residual`，默认 `replace`（对 middle/hybrid-middle 生效）。
4. `inject_stages`: 仅 middle/hybrid 使用，1-based stage 索引列表。
5. `flow_input`: 光流输入源，初版默认 `rgb`，后续可扩展 `fused`。


## 6. 数据流设计

### 6.1 Early（full-T）

输入：

1. RGB: `[B, N, 3, H, W]`
2. Spike: `[B, N, T, H, W]`（重建后固定长度）

流程：

1. 将每个 RGB 帧复制/广播为 `[B, N, T, 3, H, W]`。
2. 与对应 Spike 子帧逐时刻对齐，调用 `FusionOperator`。
3. 重排为 `[B, N*T, C_f, H, W]` 送入后续主干。

说明：此路径使主干时间分辨率与 Spike 对齐。

### 6.2 Middle

输入主序列保持当前主干时序（不强制扩展到 `N*T`），并维护一份 Spike memory（固定 `T`）。

在 stage `i`：

1. 若 `i` 不在 `inject_stages`，直接跳过。
2. 若命中注入，提取当前 stage 特征 `x_i` 与对齐后的 `spike_ctx_i`。
3. 执行 `y_i = FusionOperator(x_i, spike_ctx_i)`。
4. 应用模式：
   1. `replace`: `x_i = y_i`
   2. `residual`: `x_i = x_i + y_i`

### 6.3 Hybrid

先执行 early，再在指定 stage 执行 middle。`inject_stages` 仅控制 middle 部分。


## 7. 模块与文件组织

建议新增：

1. `models/fusion/base.py`
2. `models/fusion/factory.py`
3. `models/fusion/operators/{concat.py,gated.py,pase.py,mamba.py}`
4. `models/fusion/adapters/{early.py,middle.py,hybrid.py}`
5. `models/fusion/__init__.py`

接口建议：

1. `FusionOperator.forward(rgb_feat, spike_feat, context=None) -> fused_feat`
2. `FusionAdapter.forward(inputs, states, stage_idx=None) -> outputs`


## 8. 与现有代码集成策略（低侵入）

1. `VRT.__init__` 读取 `netG.fusion` 并通过 factory 构建 adapter。
2. `VRT.forward` 在输入端调用 early/hybrid 的 early 部分。
3. `Stage.forward` 增加可选 hook，调用 middle/hybrid-middle。
4. 不改变现有训练脚本参数接口；默认 `fusion.enable=false` 时行为与当前一致。


## 9. 兼容性与降级策略

1. 初版 full-T early 仅支持 `spikecv_tfp` 路径。
2. 若配置 `middle_tfp` 或 `snn` 且请求 full-T early，给出明确错误提示并拒绝启动（fail-fast）。
3. 对非法配置（未知 operator / placement / mode / stage 索引越界）统一在构建阶段报错。


## 10. 错误处理与可观测性

关键检查：

1. 形状校验（时序长度、通道数、空间尺寸）。
2. `inject_stages` 范围校验（必须属于现有 stage）。
3. `mode` 与 `placement` 组合合法性校验。

建议日志：

1. 启动时打印 fusion 配置摘要。
2. 首个 batch 打印关键 tensor shape（可由 debug 开关控制）。


## 11. 测试与验收标准

### 11.1 单元测试

1. factory 构建测试：不同 `placement/operator/mode` 组合。
2. operator 形状测试：输入输出 shape 与 dtype。
3. adapter 形状测试：early 的 `N -> N*T`、middle 注入行为。

### 11.2 集成测试

1. `fusion.enable=false` 时与现有 baseline 前向一致（shape/无异常）。
2. `placement=early` + `operator=gated` 能完成一次前向。
3. `placement=middle` + `inject_stages=[1,3]` 能完成一次前向。
4. `placement=hybrid` 能完成一次前向。

### 11.3 验收

满足以下条件即通过：

1. 配置可一键切换 early/middle/hybrid 与 operator。
2. middle 默认 `replace`，可切换 `residual`。
3. 在不改训练入口脚本情况下完成训练前向初始化与 smoke test。


## 12. 迭代顺序建议

1. 先交付 `concat + gated`（early + middle + hybrid 全路径可跑）。
2. 再接入 `pase`（先复用 operator 接口）。
3. 最后接入 `mamba`（放在同接口下，先保证可切换可跑，再优化性能）。

