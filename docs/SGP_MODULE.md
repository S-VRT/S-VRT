# SGP 模块说明（详实版）

## 1. 背景与目的（更详）

SGP（Scalable Granularity Perception）来自 TriDet（arXiv:2303.07347 / CVPR 图示），设计目标是用「瞬时分支（instant-level） + 多尺度窗口卷积分支（window-level）」替代自注意力的全局 token-mixing，以达到两类目的：

- **效率**：避免在大窗口 / 长时间序列上计算 QK^T/Softmax 的 O(N²) 成本；在时间维度用 depthwise Conv1D 将计算复杂度降为 O(D·C·k) 级别。  
- **表征多样性**：通过不同时间粒度的卷积分支与门控（φ / ψ），缓解 Transformer 在视频特征上出现的通道/特征趋同（rank loss / instant discriminability 下降）。

在本仓库，我们将 SGP 作为可选的 self-attention 替代算子（严格只作用于纯 self-attention 子层，即 mut_attn=False 的分支），并确保语义上的时间维聚合（按空间位置逐时间序列地做 1D 聚合），而不是把 window token 任意 flatten 为“时间”轴。

## 2. 代码位置（当前实现 — 精确指引）

下列文件包含实现细节与调用点（用以快速定位与改动）：

- `models/blocks/sgp.py` — 包含 `SGPBlock`（TriDet 复现核心）、`LayerNorm`（TriDet 风格，支持 [B,C,T]）与 `SGPWrapper`（VRT ↔ SGP 的张量适配器）。  
- `models/architectures/vrt/stages.py` — 在 `TMSA`/`TMSAG` 中，通过 `use_sgp` 开关调度 SGP 的替换逻辑；当 `use_sgp=True` 且 `mut_attn=False` 时会使用 `SGPWrapper`。  
- `options/*.json` — 实验/训练配置，关键字段：`use_sgp`, `sgp_w`, `sgp_k`, `sgp_reduction`（见第 5 节）。

注意：文档与实现严格区分两种替换策略（见第 3 节关于 `use_inner` 的说明），确保可做“干净的算子替换”或“整体 block 替换”两种可比消融。
 

## 3. 模块结构（扩展说明）

下面把 SGP 层的实现细节、形状语义、门控机制与残差策略写清楚，便于开发者理解差异并做可复现的消融。

### 3.1 设计目标与输入语义

- SGP 的设计目标是在**时间维度**提供多粒度的局部/全局融合能力（instant-level + window-level），用 depthwise Conv1D 替代自注意力的全局 token-mixing，从而降低计算复杂度并增强通道/时间的判别性。  
- 在 VRT 中我们严格把 SGP 的作用限定为“逐空间位置的时间序列聚合”。因此实现通过 `SGPWrapper` 把 VRT 的 `[B, D, H, W, C]` 重塑为 `[N, C, T]`（N = B * H * W），再调用 `SGPBlock`（channel-first, Conv1D 风格）。

### 3.2 分支与门控（phi / psi）——实现细节

- instant-level（瞬时分支）：
  - 操作流程：沿时间维做全局平均池化 -> 两层 MLP（压缩比 `sgp_reduction`）-> 激活（GELU 或 ReLU）-> 输出门控向量 φ，形状 `[N, C, 1]`。  
  - 与 `FC(x)` 相乘：`FC(x)` 为 pointwise Conv1D（或 Linear），得到 `[N, C, T]`，与 φ 广播相乘后形成 instant 输出。

- window-level（窗口分支）：
  - 包含两个 depthwise Conv1D 路径：`Conv_w(kernel=sgp_w)` 与 `Conv_kw(kernel=sgp_w*sgp_k)`（核大小按实现向上取奇数保证 same padding）。两者输出 `[N, C, T]`。  
  - `psi` 门控由 `psi_conv`（通常也是 depthwise Conv1D）产生，或直接用 conv 的某一路输出作为门控特征，最后与 `(Conv_w + Conv_kw)` 相乘。

- 融合公式（与 Figure 4 一致）：
  - fused = φ * FC(x) + ψ * (Conv_w(x) + Conv_kw(x))
  - 最终输出由 fused 与 identity（shortcut）按 `use_inner` 的策略决定是否合并。

### 3.3 SGPBlock 内部行为与 `use_inner`

- 为了支持两类可比消融，代码实现中提供 `use_inner` 参数（**默认值为 True**，即保留 TriDet 的原始行为）：
  - `use_inner=True`（完整替换模式，默认）：SGPBlock 内部完成 `LN -> SGP -> DropPath -> identity add -> GN -> FFN`（即 SGPBlock 封装了内部的 residual + 内部 FFN），这最接近 TriDet 的“完整图示”实现；在这种模式下，外层不应再对该子层重复做外层残差/FFN（以避免双重残差）。  
  - `use_inner=False`（算子替换模式）：SGPBlock 只返回 fused（或 fused + 内部 drop_path），外层 Transformer block 负责 `x + DropPath(SGP(...))` 与后续 `norm2 + FFN`，这是做“仅替换 self-attn 算子”的首选设置，便于把性能变化归因到 SGP 运算本身。

### 3.4 DropPath / AffineDropPath 与 Norm 行为

- `DropPath` 与 `AffineDropPath` 提供随机深度（stochastic depth）能力：实现中两者都可在 SGPBlock 内部配置（`path_pdrop`）。当 SGPBlock 自带 drop_path 时，TMSA 外层会把外层的 drop_path 设为 Identity，避免重复应用。
- 按 Figure 4 的建议，SGP 后的第二个归一化（原来的 `LayerNorm`）需替换为 `GroupNorm`；仓内实现中，TMSA 在 `use_sgp` 情况下会把 `norm2` 设置为 `GroupNorm` 并在 forward 中对 5D 张量做合适 reshape（`[B,D,H,W,C] -> [B*D*H*W, C]`）来执行 GN。

### 3.5 参数约定与实现细节

- `sgp_w`：window-level 主卷积核（奇数，默认 3）。  
- `sgp_k`：大核倍率，实际大核近似为 `sgp_w * sgp_k`（实现会确保为奇数，默认 3）。  
- `sgp_reduction`：instant 分支 MLP 压缩率（默认 4）。  
- `use_inner`：内部 residual/FFN 开关（**默认 True**，即 SGPBlock 默认包含内部 shortcut + FFN；如需做算子级消融请显式设为 `False`）。

### 3.6 兼容性与边界检查

- `SGPWrapper` 会严格检查输入维度，**仅接受 5D 输入 `[B,D,H,W,C]`**，并在 boundary 上返回与输入完全相同的 shape（shape conservation）。这可防止错误地把 window token 当时间轴处理。  
- 所有 Conv1D 均为 depthwise（groups=C），stride=1，padding 保持长度不变；pointwise projection 用 1x1 Conv 等效实现通道混合。

`DropPath` 与 `AffineDropPath` 在 SGPBlock 中已有实现（可配置 `path_pdrop`），因此 TMSA 在启用 SGP 时会根据 SGPBlock 的行为决定是否再包裹外层 drop path（见第 4 节）。


### 3.3 本项目的替换示意图（TMSA/RTMSA 中的 self-attention → SGP）

当 `use_sgp=true` 且 `mut_attn=false` 时，`TMSA` 的纯自注意力子层被 `SGPBlock` 替换；互注意力子层保持原 `WindowAttention`。以下流程基于 `models/network_vrt.py` 中 `TMSA`/`TMSAG`/`RTMSA` 的实际实现：

```mermaid
flowchart LR
    subgraph Original Self-Attn Block
        X0[x]
        N1[LayerNorm (norm1)]
        SA[WindowAttention\n(mut_attn=False)]
        DP1[DropPath\n(drop_path>0)]
        SUM1[(+)]
        N2[LayerNorm (norm2)]
        FFN[GEGLU-MLP]
        DP2[DropPath\n(drop_path>0)]
        SUM2[(+)]

        X0 --> N1 --> SA --> DP1 --> SUM1
        X0 -. residual .- SUM1
        SUM1 --> N2 --> FFN --> DP2 --> SUM2
        SUM1 -. residual .- SUM2
    end

    subgraph SGP Replacement Block
        X1[x]
        SGPB[SGPBlock\n(LayerNorm + SGP + DropPath 内置)]
        SUM3[(+)]
        N3[LayerNorm (norm2)]
        FFN2[GEGLU-MLP]
        DP3[DropPath\n(drop_path>0)]
        SUM4[(+)]

        X1 --> SGPB --> SUM3
        X1 -. residual .- SUM3
        SUM3 --> N3 --> FFN2 --> DP3 --> SUM4
        SUM3 -. residual .- SUM4
    end

    Original Self-Attn Block ---替换---> SGP Replacement Block
```

要点：
- SGP 只作用于 **纯 self-attention** 层（`mut_attn=False`）；互注意力层仍用 `WindowAttention`。
- `SGPBlock` 与 `SGPWrapper` 的集成细节：
  - 项目中实现了 `SGPWrapper`（`models/blocks/sgp.py`），它把 VRT 的 `[B, D, H, W, C]` 输入按“空间位置为单元”重塑为 `[B*H*W, C, D]` 以调用 `SGPBlock`，然后再把输出还原回 `[B, D, H, W, C]`。关键实现片段如下：

```265:293:models/blocks/sgp.py
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, D, H, W, C] - VRT format
            mask: Optional mask tensor
        Returns:
            y: [B, D, H, W, C] - Same shape as input
        """
        B, D, H, W, C = x.shape

        # Reshape to [B*H*W, D, C] - each spatial position becomes a sequence
        x_seq = x.permute(0, 2, 3, 1, 4).reshape(B*H*W, D, C)  # [B*H*W, D, C]

        # Convert to channel-first for Conv1D: [B*H*W, C, D]
        x_conv = x_seq.permute(0, 2, 1)  # [B*H*W, C, D]
```

  - 由于 wrapper 在模块边界做了形状适配，SGP 的语义上严格限定为“逐空间位置的时间序列聚合”（不会把 window token 混淆为时间轴）。
  - `SGPWrapper` 生成的输出与输入严格同形（shape conservation）：`[B,D,H,W,C] -> [B,D,H,W,C]`。

  - 在 `TMSA` 的实现里，启用 `use_sgp` 时会采取两种可能的调用策略（取决于 SGPBlock 是否启用内部 residual/FFN）：
    * 若 `sgp_block.use_inner == True`（SGPBlock 自带内部 identity/FFN）：TMSA 将视其为完整的替换块，不再为该子层添加外层 `x + ...` 的残差与外层 FFN（以避免双重残差）。  
    * 若 `sgp_block.use_inner == False`：SGP 只作为 self-attn 的 operator，被外层负责做 `x + DropPath(SGP(...))` 然后再进入 FFN（这更接近“只替换 self-attn 算子”的形式）。

  - 因为实现保留了 `use_inner` 可选项，你可以通过两种替换策略做可比消融：
    * 干净替换（`use_inner=False`）：外层残差/FFN 保持不变，方便直接把 self-attn 算子换成 SGP 算子的对比实验。  
    * 完整替换（`use_inner=True`）：将 SGPBlock 当作完整替代块使用（内部包含 identity + FFN）。
- FFN 分支保持不变，仍使用 `norm2 + GEGLU-MLP + DropPath` 的残差结构。

## 4. 在网络中的嵌入方式
### 4. 在网络中的嵌入方式（详细）

下面给出具体集成注意点、调用时的分支行为、以及对工程/实验人员的可执行清单，便于在不同阶段（开发 / 消融 / 训练）使用。

#### 4.1 TMSA 层的两类集成模式

在 `models/architectures/vrt/stages.py` 的 `TMSA`（及其变体）中，我们支持两种集成策略：

- 算子级替换（operator replacement，推荐用于可比消融）
  - 条件：`use_sgp=True` 且 `mut_attn=False`，并且构造 `SGPBlock`/`SGPWrapper` 时设置 `use_inner=False`（或使用 wrapper 返回纯 operator）。  
  - 行为：外层仍然执行 `x = x + DropPath(SGP(LN(x)))`，随后执行 `norm2 -> FFN -> residual`。这样 SGP 仅作为 self-attn 的替换算子，方便将差异归因于 attention->SGP 本身。

- 完整块替换（full-block replacement，接近 TriDet 图示）
  - 条件：`use_sgp=True` 且 `mut_attn=False`，并且 `SGPBlock.use_inner=True`（默认 TriDet 风格）。  
  - 行为：SGPBlock 内部完成 `LN -> SGP -> DropPath -> identity add -> GN -> FFN`，外层 **不再** 对该子层添加外层残差/FFN（TMSA 的 forward 在检测到 SGPBlock 为完整块时，会跳过外层的重复残差与 FFN，避免双重残差）。

实现注意：
 - 在代码中通过 lazy import 引入 `SGPWrapper` / `SGPBlock`，并在构造时根据 `use_sgp` 与 `mut_attn` 决定 attn 对象类型。  
 - 如果 SGPBlock 含内部 drop_path 或内部 FFN，外层应当保持相容（不要再包裹 drop_path 或重复做 FFN）。
 - 注意：当前 `TMSA` 的默认实现直接实例化 `SGPWrapper(...)` 并**未**显式传入 `use_inner`，而 `SGPBlock`/`SGPWrapper` 的 `use_inner` 默认为 `True`，因此默认行为是完整块替换（内部包含 residual + FFN）。如需算子级替换，请在构造 `SGPWrapper` / `SGPBlock` 时显式传 `use_inner=False`，或在 Stage 构造层暴露该选项。

#### 4.2 Stage / RTMSA 的开关语义

- Stage 中通常包含多个 residual groups，其中有一部分为 self-only（mut_attn=False）。配置 `use_sgp` 会将这些 self-only 子层替换为 SGP（按上文的两种策略之一）。  
- RTMSA（例如 Stage 8 的精炼阶段）通常以 mut_attn=False 堆叠，建议在该阶段尝试启用 SGP 来观察对精细纹理恢复的影响。

#### 4.3 集成检查清单（开发/PR 必做）

在把 SGP 合并进某个分支前，务必通过下列检查项以保证实现语义正确且可复现：
 - [ ] 输入/输出 shape 守恒：对随机输入 `x.shape = [B,D,H,W,C]` 做 forward，确认输出同形。  
 - [ ] 无 3D 分支污染：SGPWrapper 只接受 5D 输入，不应有把 window tokens flatten 为时间的分支路径。  
 - [ ] 单一残差路径：若选择 `use_inner=False`，确保 SGPBlock 不含内部 identity；若 `use_inner=True`，确保外层不再添加重复残差。  
 - [ ] 第二个 Norm 已替换为 GroupNorm（仅当层为 SGP 替换时）：确认 `norm2` 为 GN，并且对 5D 输入做了 reshape 处理。  
 - [ ] φ/ψ 门控结构按图实现：instant-level 的 φ 基于池化 + MLP，window-level 的 ψ 基于 conv 输出再做 gating。  
 - [ ] DropPath 行为一致：如果 SGP 内部含 drop_path，外层 drop_path 应为 Identity（或跳过）。  

这些检查已经在 `tests/models/test_sgp_integration.py` 中以单测形式部分覆盖，开发者应在 PR 前确保相关测试通过。

## 5. 配置与参数

| 配置项 | 说明 | 默认 |
| --- | --- | --- |
| `use_sgp` | 是否让纯 self-attention 层改用 SGP | `false` |
| `sgp_w` | 窗口分支主卷积核大小 | `3` |
| `sgp_k` | 大核倍率，实际核大小 `sgp_w * sgp_k` | `3` |
| `sgp_reduction` | 瞬时分支 MLP 的通道压缩比 | `4` |

示例（`options/gopro_rgbspike_local_debug.json`）：

```json
"use_sgp": true,
"sgp_w": 3,
"sgp_k": 3,
"sgp_reduction": 4
```

开启后，所有 Stage 中的纯 self-attention 层以及 Stage 8 的 RTMSA 都会切换到 SGP，互注意力层保持 `WindowAttention`。

**注意（实现约定）**：
- `SGPWrapper` 期望输入/输出格式为 `[B, D, H, W, C]`（VRT 的统一边界），并保证输出 shape 与输入完全一致。
- `SGPBlock` 的内部卷积以 channel-first `[N, C, T]` 形式工作（`N = B*H*W`）。
- `LayerNorm` 使用了从 TriDet 复现的 3D-friendly 实现（支持 `[B, C, T]`），并且在 SGP 后、FFN 前将第二个 norm 替换为 `GroupNorm`（当 `use_sgp=True` 且该层为 self-only 时）。

示例：常见 shape conservation（单次前向）  
- 输入（VRT block）： `x.shape = [B, D, H, W, C] = [2, 8, 4, 4, 64]`  
- SGPWrapper 输出： `y.shape = [2, 8, 4, 4, 64]`（经单元测试与 smoke tests 验证）

## 6. 验收标准与 Smoke Tests（必须通过）

在把 SGP 作为替换提交/合并前，请确保以下最低验收标准能够自动化验证：

- 结构正确性（自动化检查）
  - Shape 守恒：随机输入 forward 输出 shape 与输入一致。  
  - Norm 替换：当 `use_sgp=True` 且层为 self-only 时，确认 `norm2` 为 `GroupNorm`。  
  - 单一残差路径：根据 `use_inner` 设置，确保没有双重残差（自动化断言 SGPBlock 内部不含 identity 时外层会有 residual）。

- 行为正确性（基本 smoke tests）
  - 基本前向/反向：随机输入可进行 forward + backward（loss.backward() 不抛异常）。  
  - 开/关一致性：在相同 seed 下，`use_sgp=False` 与 baseline 行为一致（数值上允许小浮点误差，但训练趋势不应异常）。  

- 可解释性/动机检验（建议）
  - 通道相关性 / 有效秩评估：在替换点比较 self-attn vs SGP 的中间特征，期待 SGP 的通道相关性更低或有效秩更高（用于验证 TriDet 的“辨识度提升”动机）。

### 快速运行示例（开发者 smoke test）

下面是一个最小的 Python snippet，用于快速验证 SGPWrapper 在当前仓库环境下的前向行为：

```bash
python - <<'PY'
import torch
from models.blocks.sgp import SGPWrapper

device = torch.device('cpu')
sgp = SGPWrapper(dim=64).to(device)
x = torch.randn(2, 8, 4, 4, 64).to(device)  # [B,D,H,W,C]
with torch.no_grad():
    y = sgp(x)
print('in', x.shape, 'out', y.shape)
PY
```

建议把上述片段嵌入 CI 的 smoke stage。

## 7. 调试、测试与 FAQ（常见问题）

### 7.1 已有测试覆盖（参考）
- `tests/models/test_sgp_integration.py`：覆盖 `SGPWrapper` 的输入校验、SGPBlock phi/psi 分支形态，以及 TMSA 中 norm2 的替换行为。运行整个测试文件可获得完整的集成回归信心。

### 7.2 常见问题与排查建议
- Q: 运行时报错 GroupNorm 的 num_groups 不能整除 channels。  
  - A: 我们会在构造时自动选择一个能整除 `dim` 的 `num_groups`（从 [32,16,8,4,2,1] 中选择）；如果你的 `dim` 为奇数或特殊值，请手动指定 `num_groups` 或调整 `dim`。  

- Q: 为什么看到“残差被叠加两次”？  
  - A: 这是典型的 `use_inner`/外层 residual 配置不一致导致的问题。解决办法：若使用 SGPBlock 的 `use_inner=True`（完整替换），确保外层不再对该子层做 `x + ...`。反之，若希望外层保留 residual，请用 `use_inner=False`。  

- Q: 如何验证自己实现的 φ/ψ 与图一致？  
  - A: 在 SGPBlock forward 的中间插桩，把 `phi.mean().item()`、`psi.mean().item()`、`fc_branch.norm()`、`convw.norm()` 等打印出来并与 TriDet 参考实现对比（相同输入下数值分布应为同量级）。

### 7.3 推荐实验矩阵（快速）
- Baseline: 原始 VRT（use_sgp=False）  
- SGP-operator: `use_sgp=True`，`use_inner=False`（只替换 self-attn 算子）  
- SGP-block: `use_sgp=True`，`use_inner=True`（完整替换块）  
对每种设置跑同样的训练 schedule，比较：loss 曲线、PSNR/SSIM、以及中间表征（通道相关性 / 有效秩）。

---

如需我把本文件再输出为 README-ready 的英文版、或把 smoke tests 加入 CI pipeline（例如 GitHub Actions 或 GitLab CI），我可以继续实施那些改动。 

## 6. 训练/推理注意事项

- **内存与算力**：SGP 避免了 QK^T/Softmax 的 O(N²) 计算，窗口长度较大时显存更友好；卷积分支在大核配置下会略增算力，需要根据显存预算调整 `sgp_k`。
- **兼容性**：SGPBlock 与标准 Transformer block 接口一致，梯度检查点、DropPath、LayerNorm 等常用组件在 `network_vrt.py` 中已做兼容处理，无需额外改动。
- **调参策略**：
  - 若追求更多局部纹理复原，可提高 `sgp_k` 以增大等效感受野。
  - 若显存紧张，可同时降低 `window_size` 与 `sgp_w`，或仅在部分 Stage 启用（通过自定义 `use_sgp` 传入 Stage 构造函数实现）。
  - 可以与 `mul_attn_ratio` 联动，仅对特定 Stage 的 self-only 残差组开启 SGP，形成「互注意力 + 卷积注意力」混合。

## 7. 调试与验证建议

1. **模块单测**：在 `models/sgp_vrt.py` 中添加随机张量单元测试，确认 `SGPBlock` 保持张量形状 `(B_, N, C)` 不变。
2. **已存在的测试（参考）**：本仓库包含丰富的 pytest 用例（`tests/models/test_sgp_integration.py`），覆盖：
   - `SGPWrapper` 只接受 5D 输入并保持输出形状；  
   - `SGPBlock` 的 phi/psi 分支计算与图结构一致；  
   - 当 `use_sgp=True` 时，`TMSA` 将 `norm2` 替换为 `GroupNorm`（并对 5D 输入做合适 reshape）；  
   - SGP 模式下参数梯度流与内存特性（基本 smoke tests 已通过）。
   直接运行相关测试可以得到形状/类型的即时打印信息，方便把示例值拷入文档。
2. **训练日志**：关注 `options` 中 `use_sgp` 开关对应的 WANDB/SwanLab 运行，比较 loss 曲线、PSNR/SSIM 等指标。
3. **性能 Profiling**：使用 `torch.cuda.profiler` 或 `nsys` 对比启用/关闭 SGP 的时延与显存，便于确定最佳 `sgp_w/sgp_k` 组合。

---

如需进一步扩展（例如引入可学习 `sgp_k` 或与多尺度 attention 混合），建议在本文件中补充实验记录与设计思路，保持文档与代码一致。

