# S-VRT Dual-Input Fusion Design Spec

## 1. 背景与问题

当前 S-VRT 存在“数据集先拼接 `L=(RGB+Spike)`，模型 early fusion 再拆分 RGB/Spike”的耦合路径。该路径可运行，但在架构语义上不清晰：

- 数据集职责与融合职责交叉
- concat baseline 与 fusion 实验切换成本高
- 配置语义混杂（`spike_channels` 同时承担“重建 bins”和“网络输入通道”语义）

## 2. 目标

实现以下目标，并保持向后兼容：

1. 通过配置文件在 `concat` baseline 与 `fusion`（early/middle/hybrid）之间切换
2. 数据侧仅负责重建与打包，不负责融合
3. 模型侧负责融合策略与融合执行
4. 兼容旧路径（`L` 单键输入）

## 3. 非目标

本次不引入新任务监督，不直接实现“去模糊+插帧联合训练”的训练头与损失重构。
本期默认任务仅为去模糊，不引入插帧 GT、插帧 loss、插帧专用解码头。

## 4. 设计原则

1. 单一职责：数据集做 reconstruction/packing，模型做 fusion
2. 配置驱动：同一代码路径支持不同实验模式
3. 向后兼容：旧配置/旧脚本可继续运行
4. 语义清晰：显式区分“重建 bins”和“模型输入模式”

## 5. 目标架构

### 5.1 数据输出契约

数据集支持两类打包：

- `input_pack_mode = "concat"`：返回 `L`（历史行为）
- `input_pack_mode = "dual"`：返回 `L_rgb` + `L_spike`，可选保留 `L`

约定形状：

- `L_rgb: [T, 3, H, W]`
- `L_spike: [T, S, H, W]`（`S` 为 spike reconstruction bins）
- `L: [T, 3+S, H, W]`（仅 concat 或 dual+compat 时提供）

职责边界约束：

- 数据侧只负责 reconstruction + packing，不执行 fusion
- 数据侧不产出 early/middle/hybrid 的中间融合结果键

### 5.2 模型输入契约

模型支持两种入口：

- `input_mode = "concat"`：按历史输入 `x=L`
- `input_mode = "dual"`：优先读取 `rgb/spike` 双输入；若仅有 `L` 则可兼容回退

输入优先级约束：

- 当 `input_mode="dual"` 且 batch 同时存在 `L_rgb/L_spike` 与 `L` 时，优先使用 `L_rgb/L_spike`
- 实际生效路径必须在日志中显式打印（`concat_path` / `dual_path`）

### 5.3 fusion 放置语义

- `early`：在 backbone 前融合
- `middle`：在 stage 特征注入融合
- `hybrid`：early + middle 结合

默认去模糊任务保持输出：`[B, N, 3, H, W]`

### 5.4 early 执行边界

- early fusion 在模型 `forward` 前段执行
- early 输出不新增 batch 键，不在数据管线中传递
- early 融合结果仅作为模型内部局部变量 `x` 进入主干

## 6. 关键设计决策

### 决策 A：early 默认采用 shape-preserving

- 默认：`[B,N,3] + [B,N,S] -> [B,N,C_out]`
- 可选实验：`expand_to_full_t=true` 允许 full-T 变体（需单独实验标注）

理由：与现有 GT 监督 `[B,N,3,H,W]` 对齐，避免时序语义和损失定义错位。

### 决策 B：配置分层

将语义拆分为三层：

1. `datasets.*.spike.reconstruction.*`：Spike 重建策略与 bins
2. `datasets.*.input_pack_mode`：数据打包方式（concat/dual）
3. `netG.input_mode` + `netG.fusion.*`：模型输入方式与融合策略

### 决策 C：配置字段适用范围必须显式化

- concat-only 字段只在 `input_mode=concat` 生效
- dual+fusion 字段只在 `input_mode=dual` 且 `fusion.enable=true` 生效
- 禁止同一字段同时承载“重建 bins”和“主干输入通道”两种语义

## 7. 配置草案

```json
{
  "datasets": {
    "train": {
      "input_pack_mode": "dual",
      "keep_legacy_l": true,
      "spike": {
        "reconstruction": {
          "type": "spikecv_tfp",
          "num_bins": 8
        }
      }
    }
  },
  "netG": {
    "input_mode": "dual",
    "in_chans": 11,
    "fusion": {
      "enable": true,
      "placement": "early",
      "operator": "gated",
      "mode": "replace",
      "early": {
        "out_chans": 11,
        "shape_policy": "preserve_n",
        "expand_to_full_t": false
      }
    }
  }
}
```

说明：

- concat baseline：`input_pack_mode=concat`, `input_mode=concat`, `fusion.enable=false`
- fusion 实验：推荐 `input_pack_mode=dual`, `input_mode=dual`, `fusion.enable=true`
- `netG.in_chans` 表示进入 VRT 主干的通道数约束；在 early replace 下要求 `early.out_chans == in_chans`
- `expand_to_full_t=true` 仅用于实验分支，不作为默认训练配置

## 8. 兼容性与迁移

1. 若旧配置仅提供 `L`，模型维持旧行为
2. 若新配置提供 `L_rgb/L_spike`，模型优先双输入路径
3. 逐步将历史 `spike_channels` 迁移到 `spike.reconstruction.num_bins`
4. 迁移期保留映射：
   - `datasets.*.spike_channels (旧)` -> `datasets.*.spike.reconstruction.num_bins (新)`
   - `datasets.*.spike_reconstruction (旧)` -> `datasets.*.spike.reconstruction.type (新)`

## 9. 风险与缓解

1. 风险：双路径并存导致分支复杂
- 缓解：统一入口归一化，内部统一成同一执行路径

2. 风险：full-T 误用到去模糊主实验
- 缓解：默认关闭 `expand_to_full_t`，并在配置注释明确其为实验模式

3. 风险：评测脚本依赖 `L` 单键
- 缓解：`keep_legacy_l=true` 作为迁移期默认

4. 风险：配置“写了但未生效”导致实验结论失真
- 缓解：启动时打印关键配置生效摘要（pack_mode/input_mode/fusion_placement/shape_policy）

## 10. 验收标准

1. 不改训练脚本时，旧 concat 配置可继续训练/测试
2. 新 dual+fusion 配置可跑通 train/test forward
3. early/middle/hybrid 三种 placement 在 dual 模式下可切换
4. 输出维度在去模糊模式下保持 `[B,N,3,H,W]`
5. 验收矩阵至少覆盖：
   - `concat + fusion.disable`
   - `dual + early(preserve_n)`
   - `dual + middle`
   - `dual + hybrid`
   - `dual + early(expand_to_full_t=true)`（标记为实验模式，仅验证可运行与日志标识，不纳入默认主结果）
