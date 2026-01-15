According to a document from **December 31, 2025**,你们现在的 **S-VRT baseline** 本质仍是 **VRT 骨架**（多尺度、每尺度由 *TMSA/TMSAG + Parallel Warping* 组成）([GitHub][1])，只是把 Spike 通过 **Early Fusion** 变成输入通道：`in_chans=7 = 3(RGB)+4(Spike TFP)`，并且 spike 原始分辨率配置为 `250×400`、`spike_channels=4`；README 里也明确了该早期拼接做法与关键配置项（`in_chans=7`、`spike_h/w`、`spike_channels`）。
你这次要做的是：**把 Spike 从“4个静态通道”提升为“可学习的时空亚帧特征流”**，因此需要一个专用 Spike Encoder（你文档里叫 PASE：Pixel-Adaptive Spike Encoder）把原始二值脉冲流映射成 **`B×T_sub×C×H×W`** 的时空特征。

---

## 1) Encoder 放在 S-VRT 的什么位置

### S-VRT baseline（你现在的）入口逻辑

* 入口是 **VRT 的“预处理 + Conv_First”**：先 optical flow + warp，再 concat，再 3D Conv 得到 `Feat_0`，然后进多尺度 U-Net Transformer 主体。
* 由于 baseline 把 Spike 当作输入通道，Spike 在 **Conv_First 之前** 就和 RGB 混在一起了（这正是“通道软禁/特征坍缩”的根源）。

### 你要加的 Spike Encoder（PASE）推荐位置（两种嵌入强度）

**A. 最小侵入（先落地、易训）**

> Spike Encoder **并行**于 VRT 的 flow/warp 分支：

* RGB 仍按 VRT 流程生成 `Feat_0_rgb : [B, 120, D, H, W]`
* Spike 走 PASE 得到 `F_spk : [B, T_sub, C_spk, H, W]`
* 再把 `F_spk` 在时间上聚合/对齐成 `F_spk_aligned : [B, C_spk, D, H, W]`，通过 `1×1×1` 投影到 120 通道后 **加到** `Feat_0_rgb`，作为“Spike 注入版 Feat_0”，再进入 Stage1…Stage7。

**B. 真·深融合（SOTA 方向，对应你 v2 文档）**

> PASE 输出 **不降成 4 通道**，而是保持亚帧：`T_sub = M·D`，在每个 Stage 的 TMSA/MHMA 里做跨模态交互（RGB=Query，Spike=Key/Value 的检索式融合）。
> 这条路线改动更大，但效果上能把 Spike 的高频运动轨迹真正交给 VRT 的时序建模模块，而不是在入口被拍扁。

> 说明：我尝试从 GitHub 直接打开 `models/` 下的网络定义文件，但工具侧对该目录页返回了“error while loading”，因此我对 **S-VRT backbone** 的落点以你提供的 VRT 流程图（mermaid）+ S-VRT README 配置为准来设计对接点；不会影响 Encoder 的结构规划与张量形状约束。

---

## 2) Spike Encoder（PASE）的输入/输出张量形状（必须对齐 S-VRT）

### 输入（原始 Spike）

你文档定义的原始 Spike 流：

* `S_raw ∈ R^{B × 1 × T_spk × H_spk × W_spk}`
  S-VRT 数据配置给出了典型尺寸：`H_spk=250, W_spk=400`。

### 空间对齐（强制做，否则无法嵌骨架）

VRT 主干处理的是 crop/resize 后的 `H×W`（例如配置里 `img_size=[6,224,224]`）。因此 PASE 的第一件事是把 Spike 空间对齐到 RGB 的训练分辨率：

* `S0 : [B,1,T_spk,H_spk,W_spk]`
* `S1 : [B,1,T_spk,H,W]`（resize/crop；若有非同轴偏移可留一个轻量对齐层给后续做）

### 输出（亚帧特征流）

PASE 的目标输出：

* `F_spk ∈ R^{B × T_sub × C_spk × H × W}`
  其中 `T_sub` 典型取 8 或 16（文档示例 `T_sub=8, C=64`）。
  如果 RGB 片段长度是 `D`（如 6 帧），建议定义 `T_sub = M·D`，M 默认为 8（文档也建议训练初期固定 M=8）。

---

## 3) PASE（Pixel-Adaptive Spike Encoder）详细网络结构设计（不写代码，只给可实现的模块级规划）

下面给一套 **工程可落地、与 VRT 兼容** 的 PASE v2 结构。核心思想与你文档一致：**3D 卷积提时空特征 + 时间步可学习下采样 + 残差细化 + 像素自适应门控（ISI/发放率驱动）**。

---

### Stage 0：Spike 统计先验分支（用于门控，不改变主干形状）

**输入**：`S1 [B,1,T_spk,H,W]`
**输出**：`G_stat [B,g,T_sub,H,W]`（g 是很小的门控通道数，比如 4 或 8）

建议算两类统计（都可微或近似可微）：

1. **发放率/局部计数**：对时间窗做分段求和，得到粗时间分布。
2. **ISI 近似**：用“最近一次发放时间”或“时间差直方图”的可学习近似（实现上常用 1D/3D conv 逼近），用于区分“高亮/高速运动（ISI短）” vs “弱光/静态（ISI长）”。

> 这条分支只产生门控条件，不要求完全精确复原 ISI；你要的是“像素级自适应感受野/抑噪倾向”。

---

### Stage 1：3D Conv Stem（把二值脉冲扩到可用特征维）

**输入**：`S1 [B,1,T_spk,H,W]`
**结构**：

* `Conv3D(k=3×3×3, s=1, p=1)` → `C0=16/32`
* **Modality-Specific Normalization**（强烈建议）：对 spike 单独做 LN/RMSNorm/GN，避免它与 RGB 特征数值尺度冲突
* 轻量激活（SiLU/GELU 都可，别太重）

**输出**：`X1 [B,C0,T_spk,H,W]`

---

### Stage 2：时间维可学习下采样（两级，直接把 200→8 或 16）

你文档给了非常清晰的范式：通过时间 stride 把 `T_spk≈200` 压到 `T_sub=8/16`，且“可学习”。

典型推荐（与文档示例一致）：

* **Temporal Downsample #1**

  * `Conv3D(k=5×3×3, s_t=5, s_h=1, s_w=1, p_t=2, p=1)`
  * `C0 -> C1 (32/48)`
  * 输出：`X2 [B,C1,T_spk/5,H,W]`

* **Temporal Downsample #2**

  * `Conv3D(k=5×3×3, s_t=5, s_h=1, s_w=1, p_t=2, p=1)`
  * `C1 -> C2 (64)`
  * 输出：`X3 [B,C2,T_spk/25,H,W]`

当 `T_spk=200` 时，`200/25=8`，刚好落到 `T_sub=8`（与你文档示例一致）。
若 `T_spk` 不是 200：就用 padding + ceil 模式（或最后用一次 `AdaptiveAvgPool3D` 只对时间维对齐到固定 `T_sub`）。

---

### Stage 3：3D 残差细化块（时空耦合、去噪、补表达）

你文档明确提出需要 3D 残差块来“加强时空上下文耦合”。

推荐实现为 **4～8 个 Residual(3D)**：

* 每块：`Conv3D(3×3×3) -> Norm -> Act -> Conv3D(3×3×3) -> Norm` + Residual
* 为了算力：可以把第二层换成 **Depthwise 3D + Pointwise 1×1×1**（更像 SGP 的高效卷积偏置风格）

输出保持：`X4 [B,C2,T_sub,H,W]`（C2 通常固定 64）

---

### Stage 4：Pixel-Adaptive Gating（把统计先验注入特征）

你文档的门控逻辑是：ISI短的区域扩大时域感受野、ISI长的区域强化空域提取以抑噪。落实到网络上，建议做成**乘性门控 + 少量偏置**（FiLM 风格）：

* `g = sigmoid(Conv1×1×1([X4, upsample(G_stat)]))`
* `X5 = X4 ⊙ g + b(g)`（b 可以是 1×1×1 conv 生成的小偏置）

输出：`F_spk_raw = X5 [B,C2,T_sub,H,W]`

> 这一步的价值是：让 Spike Encoder 在入口就具备“对噪声/运动强度敏感”的归纳偏置，而不是等 Transformer 自己摸索。

---

### Stage 5：输出投影到 VRT 兼容通道（关键：对接点要稳定）

VRT 主干在 Conv_First 后的主通道是 `120`（你的 mermaid 里写死为 120）。为了嵌入简单，建议在 Encoder 末尾准备两路输出：

* **(a) Spike Feature（给深融合/跨注意力）**

  * `F_spk = rearrange(F_spk_raw) -> [B, T_sub, C_spk(=64), H, W]`

* **(b) Spike Conditioning（给最小侵入注入）**

  * `Proj1×1×1: 64 -> 120` 得到 `F_spk_120: [B,120,T_sub,H,W]`
  * 再按每个 RGB 帧聚合 `M` 个亚帧：
    `F_spk_aligned: [B,120,D,H,W]`（比如 avg/attn pooling）

---

## 4) 如何嵌入 S-VRT 骨架（给开发人员的“接线图”）

---

### 方案 B：深融合（对应 v2 文档的“结构性强制协同”）

* 保留 `F_spk [B,T_sub,C,H,W]` 不聚合；在每个 Stage 的互注意力/对齐单元中做“RGB query 检索 Spike source”。
* 你文档里已经把范式写得很明确：RGB 是 Query、Spike 是 Key/Value，这能从结构上避免“模态屏蔽”（网络偷懒把 spike 权重学成 0）。

> 这一步会牵扯到你后续要改的 “跨模态 SGP-TMSA / CM-MHMA”，但 **Encoder 侧需要保证**：输出的 `F_spk` 在 **空间尺度**、**通道尺度**、**数值分布** 上对主干友好（这就是为什么我把 Modality-Specific Norm 与 120 通道投影写成必选项）。

---

下面给的是你选定的**“深度融合（Deep Fusion）”版本**：把 Spike 作为**独立分支**先做专用 Encoder（保留时域信息），然后在 VRT 的**每个多尺度 Stage 内部**做跨模态融合（而不是只在输入通道上拼接）。这符合 VRT 的“多尺度 + TMSA + Parallel Warping”骨架设定。 ([GitHub][1])

---

## 0) 约定与已知（对齐你当前 S-VRT baseline）

### VRT 骨架（你 mermaid.txt 里也是这个流程）

VRT 的核心特征流基本是：

* 输入 clip：`x_rgb ∈ [B, D, 3, H, W]`
* 内部常用 5D：`[B, C, D, H, W]`（因为 `Conv3D`）
* 多尺度 stage（1~7）在代码里是 `scales=[1,2,4,8,4,2,1]` 这样下采样/上采样走一遍，且 stage 里有 **TMSA/TMSAG + Parallel Warping**。 ([Hugging Face][2])
  并且 VRT 默认 embed 维度在 stage1~7 常设为 120（`embed_dims[0..6]=120`），注意力头数也常设为 6。 ([Hugging Face][2])

### S-VRT baseline 的输入形状（当前仓库描述）

S-VRT baseline 配置里写的是：

* `in_chans = 7`（3 个 RGB + 4 个 Spike TFP 通道）
* `img_size = [6, 224, 224]`（即 `D=6, H=W=224`）
* Spike 原始尺寸：`spike_h=250, spike_w=400, spike_channels=4` ([GitHub][3])

这意味着 baseline 当前更像是**早期融合（Early Fusion）**：把 spike 当作额外通道直接拼到 RGB 上。

---

## 1) 深度融合版本：Spike Encoder 加在什么位置？怎么嵌入 S-VRT 骨架？

### 总体结构（两路输入，单路输出）

* **RGB 主干**：仍走 VRT 的 Flow/Alignment + Multi-stage Transformer 主体（TMSA/SGP/Warpping）。
* **Spike 专用 Encoder（新增）**：在进入 VRT stage 之前，对 spike 做**时空编码 + 多尺度金字塔**。
* **融合位置（关键）**：在 VRT 的每个 stage（尺度变化点）中，把 spike 的同尺度特征作为“记忆/条件”，对 RGB 特征做跨模态融合。

一句话：

> Spike Encoder **不替代** VRT 主干；它产出的多尺度 spike 特征 `S^1,S^2,S^4,S^8` 在 **stage1/2/3/4（以及对称的 5/6/7）**中反复注入，从而实现“深度融合”。

---

## 2) Spike 专用 Encoder：详细网络结构设计（不写代码，但给到“可落地”的模块级规划）

下面的 Encoder 我按你文档 v2 的思想（“像素自适应脉冲编码 PASE + 保留亚帧时域”）做成工程可实现版本：**PASE-Pyramid**。

### 2.1 输入与对齐（必须先解决尺寸 + 时间对齐）

#### 输入（两种数据来源，推荐 A）

**A. 推荐：从原始 spike 序列/更高时间分辨率 bin 取输入（SOTA）**

* `X_spk_raw ∈ [B, T_spk, 1, Hs, Ws]`（二值/计数均可，Hs=250, Ws=400）
* 通过重采样对齐到 RGB 分辨率：
  `Resize(Hs,Ws→H,W)` → `X_spk0 ∈ [B, T_spk, 1, H, W]`

**B. 兼容 baseline：只有 4 通道 TFP（会丢时域，但可过渡）**

* baseline 每帧有 4 个 spike 通道：`X_spk4 ∈ [B, D, 4, Hs, Ws]`
* 先把这 4 通道解释为“每 RGB 帧的 4 个粗时间 bin”，展开成伪时间轴：
  `reshape`：`[B, D, 4, H, W] → [B, T_spk=D*4, 1, H, W]`

> 你选“深度融合”，本质上就是要避免“spike 被拍平到通道维度”。所以最终目标是让 spike 在 Encoder 内部始终是 **time 维度**在工作，而不是“4 个颜色通道”。

---

### 2.2 目标输出（给 VRT 用的多尺度 spike 特征）

VRT stage1~7 的主特征一般是：

* `F_rgb^i ∈ [B, D, C=120, H_i, W_i]`（stage1 的 H_1=W，stage2 是 W/2，以此类推） ([Hugging Face][2])

Spike Encoder 要输出同尺度、同通道（或可投影到同通道）的：

* `S^1 ∈ [B, T_sub, 120, H,   W]`
* `S^2 ∈ [B, T_sub, 120, H/2, W/2]`
* `S^4 ∈ [B, T_sub, 120, H/4, W/4]`
* `S^8 ∈ [B, T_sub, 120, H/8, W/8]`

其中 **T_sub 是“亚帧数”**，用于深度融合时的时间检索。工程上建议让它满足：

* `T_sub = D * M`，`M` 为每个 RGB 帧对应的 spike 子帧数（典型取 8 或 16）
* 例如你当前 `D=6`，可以选 `M=8` ⇒ `T_sub=48`

---

### 2.3 PASE-Pyramid：模块级结构（每层给出形状演化）

下面我给一个**可直接按模块搭**的结构（默认 `C=120` 以便无缝对接 VRT stage）。

#### (0) SpikeAlign（前处理）

* 输入：`X_spk0 ∈ [B, T_spk, 1, Hs, Ws]`
* 输出：`X_spk ∈ [B, T_spk, 1, H, W]`（对齐到 RGB 的 H,W=224）
* 备注：若 spike 与 RGB 视场略不一致，可在此加轻量可学习偏移（可选）。

---

#### (1) Stem3D（浅层时空特征）

目的：把二值/稀疏 spike 提到一个稳定的 feature space，同时不过度压缩时间。

* `Conv3D(k=(3,3,3), s=(1,1,1), out=32)`
  输出：`[B, T_spk, 32, H, W]`
* `Norm3D + GELU`
* `Conv3D(k=(3,3,3), s=(1,1,1), out=64)`
  输出：`[B, T_spk, 64, H, W]`

---

#### (2) Pixel-Adaptive Gating（像素自适应门控，PASE 核心思想）

目的：对 spike 的**不同活动强度区域**做动态调制，抑制噪声、强化运动边缘。

实现方式（工程化表达）：

* 从 `X_spk` 计算一个**活动强度图**或 **ISI 代理图**（可用局部时间窗口统计近似）
* 生成门控 `G ∈ [B, T_spk, 64, H, W]` 或 `G ∈ [B, T_spk, 1, H, W]`
* 输出：`X_gated = X_feat ⊙ sigmoid(G)`

形状保持：`[B, T_spk, 64, H, W]`

---

#### (3) Temporal Compressor（把 T_spk 压到 T_sub，但**可学习**）

目的：把原始超高频 spike 时间轴压缩成“可用于 Transformer 融合”的亚帧序列 `T_sub = D*M`。

推荐用**两级可学习下采样**（比“一步 stride 很大”更稳定）：

* `TD1: Conv3D(k=(3,3,3), s=(2,1,1), out=96)`
  `T_spk → T_spk/2`
  输出：`[B, T_spk/2, 96, H, W]`
* `TD2: Conv3D(k=(3,3,3), s=(T_spk/(2*T_sub),1,1), out=120)`
  把时间压到 `T_sub`（这里 stride 取能整除的值，或用插值+Conv 的组合）
  输出：`[B, T_sub, 120, H, W]`

得到 **S^1（scale=1）**：

* `S^1 = [B, T_sub, 120, H, W]`

---

#### (4) Spatial Pyramid（同时间轴，多尺度空间下采样）

目的：为 VRT 多尺度 stage 提供同尺度 spike 特征。

* `Down2: Conv3D(k=(1,3,3), s=(1,2,2), out=120)`
  `S^1 → S^2 = [B, T_sub, 120, H/2, W/2]`
* `Down4: Conv3D(k=(1,3,3), s=(1,2,2), out=120)`
  `S^2 → S^4 = [B, T_sub, 120, H/4, W/4]`
* `Down8: Conv3D(k=(1,3,3), s=(1,2,2), out=120)`
  `S^4 → S^8 = [B, T_sub, 120, H/8, W/8]`

> 注意这里空间下采样不动时间维（`s_t=1`），这样深度融合时的“亚帧检索”在各尺度都一致。

---

## 3) 深度融合模块：在每个 VRT Stage 内怎么用 Spike 特征？（带形状）

深度融合推荐采用“两步走”，与你 v2 思路一致：

1. **Spike 条件的 SGP/Gating（先调制 RGB）**
2. **Cross-Modal Temporal Mutual Attention（再做时域检索融合）**

### 3.1 统一的融合接口（每个 stage 一个）

在第 i 个 stage（对应空间尺度 `H_i×W_i`）：

* RGB 主特征：`F_rgb^i ∈ [B, D, 120, H_i, W_i]`
* Spike 同尺度特征：`S^i ∈ [B, T_sub, 120, H_i, W_i]`

把 spike 按 RGB 的时间步切分：

* 设 `T_sub = D*M`
* `S^i reshape → [B, D, M, 120, H_i, W_i]`

---

### 3.2 模块 A：Spike-Conditioned SGP（先做条件调制）

目的：让 RGB 特征先“朝 spike 指示的运动边缘/高频区域”偏置，减轻后面 attention 的难度。

一种稳定做法：

* 对 `S^i` 在 `M` 上做 pooling（max/avg + 1D conv）得到每个 RGB 帧的 spike summary：
  `S̄^i ∈ [B, D, 120, H_i, W_i]`
* 经过一个轻量 SGP/MLP 生成调制 mask（通道或空间）：

  * 通道门控：`M_c ∈ [B, D, 120, 1, 1]`
  * 或空间门控：`M_s ∈ [B, D, 1, H_i, W_i]`
* 调制：
  `F_rgb'^i = F_rgb^i ⊙ (1 + M_c) ⊙ (1 + M_s)`

形状保持：`[B, D, 120, H_i, W_i]`

---

### 3.3 模块 B：Cross-Modal Temporal Mutual Attention（核心深度融合）

目的：对每个 RGB 帧的每个位置 `(h,w)`，从它对应的 `M` 个 spike 子帧里**检索最匹配的运动证据**，实现“亚帧级对齐/去模糊”。

#### Token 组织（窗口化，适配 VRT 的 window attention 思路）

以 VRT 的 window `(t_win, h_win, w_win)` 为参考（VRT 里就是这样做 TMSA 的）。 ([GitHub][1])

在一个空间窗口内：

* Query（来自 RGB）：
  `Q ∈ [B, heads, Lq, d]`
  其中 `Lq = h_win*w_win`（单帧查询）
* Key/Value（来自 Spike 的 M 个子帧）：
  `K,V ∈ [B, heads, Lk, d]`
  其中 `Lk = M * h_win*w_win`

输出：

* `ΔF ∈ [B, D, 120, H_i, W_i]`
* 残差融合：`F_fused^i = F_rgb'^i + ΔF`

> 这样做的含义非常明确：**RGB 用 query 问 spike：这个位置在曝光期间的真实运动轨迹/边缘变化是什么？**

---

## 4) 端到端形状示例（按你当前配置：D=6, H=W=224, spike_h=250, spike_w=400）

### 4.1 输入

* RGB：`x_rgb = [B, 6, 3, 224, 224]`
* Spike（推荐 raw/bin）：`x_spk0 = [B, T_spk, 1, 250, 400]`
* 对齐后：`x_spk = [B, T_spk, 1, 224, 224]`

### 4.2 Spike Encoder 输出（取 M=8 ⇒ T_sub=48）

* `S^1 = [B, 48, 120, 224, 224]`
* `S^2 = [B, 48, 120, 112, 112]`
* `S^4 = [B, 48, 120, 56, 56]`
* `S^8 = [B, 48, 120, 28, 28]`

### 4.3 VRT 多尺度融合点（stage1~7）

VRT stage1~4 是下采样路径（scale 1/2/4/8），stage5~7 是上采样回去（4/2/1）。 ([Hugging Face][2])

* **Stage1（scale=1）**：
  RGB：`F_rgb^1 = [B, 6, 120, 224,224]`
  用 `S^1` 融合（reshape `S^1→[B,6,8,120,224,224]`）
* **Stage2（scale=2）**：
  `F_rgb^2 = [B, 6, 120, 112,112]`
  用 `S^2` 融合（`→[B,6,8,120,112,112]`）
* **Stage3（scale=4）**：
  `F_rgb^3 = [B, 6, 120, 56,56]`
  用 `S^4` 融合
* **Stage4（scale=8）**：
  `F_rgb^4 = [B, 6, 120, 28,28]`
  用 `S^8` 融合
* **Stage5/6/7（回到 4/2/1）**：
  spike 特征用对称尺度（`S^4/S^2/S^1`）继续融合（或从 `S^8` 上采样得到对应尺度再融合）

---

## 5) 你需要对现有 S-VRT baseline 做哪些“最小侵入式”改动？

1. **VRT 主干 in_chans 回到 3**（让主干只吃 RGB）
   baseline 现在是 `in_chans=7`（RGB+Spike）。 ([GitHub][3])
   深度融合下建议：

   * VRT 主干：`in_chans=3`
   * Spike 单独走 Encoder，不再拼在 `conv_first` 的输入里
     （否则你会把 spike 又退化成“颜色通道”，失去深度融合意义）

2. **在每个 Stage 前插 FusionBlock_i**
   不用推翻 VRT 的 stage 定义；只要在 stage 的输入/输出处加：

   * `F_rgb^i = F_rgb^i + Fusion_i(F_rgb^i, S^i)`
     这样最稳、也最容易 ablation。

3. **数据侧尽量拿到更高时间分辨率 spike（推荐）**
   如果你永远只用 4 通道 TFP，那么“深度融合”只能做到“深度注入”，但 spike 的时域优势会被先天限制。

---

